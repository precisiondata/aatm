import time
import pandas as pd
from pathlib import Path
import sqlite3
import questionary
from rich.console import Console
from rich.progress import track
import shutil

from .data_models import ExpressionMetadata

console = Console()


def rate_limiter(n_docs: int, rate_limit: int, next_allowed_time: float) -> float:
    """Rate limit helper. Sleep as needed so we don't exceed the rate limit.

    Args:
        n_docs: Number of documents to process
        rate_limit: Rate limit in docs per minute
        next_allowed_time: Next allowed time

    """
    now = time.monotonic()
    if now < next_allowed_time:
        time.sleep(next_allowed_time - now)
        now = time.monotonic()
    # Reserve time for this batch
    next_allowed_time = max(next_allowed_time, now) + n_docs * 60.0 / rate_limit
    return next_allowed_time


def build_local_sqlite_vocab_database(vocab_dir: Path) -> None:
    if Path(".aatm/omop.db").exists():
        user_preference = questionary.select(
            "A vocabulary database already exists. What would you like to do?",
            choices=[
                "Skip",
                "Overwrite",
            ],
        ).ask()

        if user_preference == "Skip":
            return

    con = sqlite3.connect(".aatm/omop.db")
    list(vocab_dir.glob("*.csv"))[0].stem.lower()

    for file_path in track(
        list(vocab_dir.glob("*.csv")), description="Building vocabulary database..."
    ):
        table_name = file_path.stem.lower()
        if table_name == "source_to_concept_map":
            df = pd.read_csv(file_path, sep=",", dtype=str)
        else:
            df = pd.read_csv(file_path, sep="\t", dtype=str)
        df.to_sql(table_name, con, index=False, if_exists="replace")

    con.commit()
    con.close()


def build_mapping_datasets(standard_vocabularies: list[str]) -> None:
    import yaml
    from importlib.resources import files

    sql_commands_path = files("aatm").joinpath("sql_commands.yaml")
    sql_commands = yaml.safe_load(sql_commands_path.read_text(encoding="utf-8"))

    if len(standard_vocabularies) == 0:
        raise ValueError("No standard vocabularies provided")

    datasets_base_path = Path(".aatm/datasets")
    if datasets_base_path.exists() and len(list(datasets_base_path.glob("*.csv"))) > 0:
        user_preference = questionary.select(
            "The datasets for terminology mapping already exist. What would you like to do?",
            choices=[
                "Skip",
                "Overwrite",
            ],
        ).ask()

        if user_preference == "Skip":
            return

    datasets_base_path.mkdir(exist_ok=True, parents=True)

    con = sqlite3.connect(".aatm/omop.db")

    dfs = []

    for command_name in track(
        sql_commands.keys(), description="Building datasets for terminology mapping..."
    ):
        sql_prompt = sql_commands[command_name]["sql"]
        sql_prompt_formatted = sql_prompt.format(
            standard_vocabulary_list=tuple(standard_vocabularies)
        )
        sql_prompt_df = pd.read_sql(sql_prompt_formatted, con)
        sql_prompt_df.to_csv(datasets_base_path / f"{command_name}.csv", index=False)
        dfs.append(sql_prompt_df)


def build_local_vector_database(
    embedding_model_name: str,
    vector_db_dir: Path | None = None,
    rate_limit: int | None = None,
    batch_size: int = 100,
) -> None:
    """
    Build local vector database. By default, uses chromadb for vector database and creates/repair the database. It gives the option to skip this step if the database already exists, to repair the database, or to overwrite the database.

    Args:
        vector_db_dir: Path to directory containing vector database
        embedding_model_name: Name of embedding model
        rate_limit: Rate limit in docs per minute
        batch_size: Batch size
    """
    # lazy loading for performance
    import chromadb
    from .retrievers import CHROMADB_RETRIEVER_MODEL_REGISTRY as model_registry

    # check if vector database directory provided exists
    if vector_db_dir is not None and not vector_db_dir.exists():
        raise ValueError("vector_db_dir does not exist")

    if vector_db_dir is None:
        vector_db_dir = Path(model_registry[embedding_model_name]["chromadb_path"])

    # check user preference if vector db already exists
    if vector_db_dir.exists():
        user_preference = questionary.select(
            f"A vector database using the embedding model '{embedding_model_name}' database already exists. What would you like to do?",
            choices=["Skip", "Repair", "Overwrite"],
            default="Skip",
        ).ask()

        if user_preference == "Skip":
            return
        elif user_preference == "Overwrite":
            shutil.rmtree(vector_db_dir)
        elif user_preference == "Repair":
            # this function already checks for incomplete vector db and repairs it
            pass

    if rate_limit is None:
        rate_limit = model_registry[embedding_model_name].get("rate_limit", None)

    if rate_limit is not None and rate_limit <= 0:
        raise ValueError("rate_limit must be > 0")

    if rate_limit is not None:
        next_allowed_time = time.monotonic()

    console.print("Creating local vector database...")
    client = chromadb.PersistentClient(
        model_registry[embedding_model_name]["chromadb_path"]
        if vector_db_dir is None
        else vector_db_dir
    )
    collection = client.get_or_create_collection(
        model_registry[embedding_model_name]["collection_name"],
        embedding_function=model_registry[embedding_model_name]["embedding_function"](
            model=model_registry[embedding_model_name]["model_id"]
        ),
    )

    datasets_base_path = Path(".aatm/datasets")
    for dataset_path in datasets_base_path.glob("*.csv"):
        expression_origin = dataset_path.stem
        df = pd.read_csv(dataset_path, low_memory=False)
        df = df.drop_duplicates().dropna()
        for i in track(
            range(0, len(df), batch_size),
            description=f"Adding embeddings for {expression_origin}",
        ):
            records = df.iloc[i : i + batch_size].to_dict("records")
            records = [
                ExpressionMetadata(**record, expression_origin=expression_origin)
                for record in records
            ]
            pairs = [(r.expression_id, r) for r in records]

            seen = set()
            pairs = [(i, r) for (i, r) in pairs if (i not in seen and not seen.add(i))]

            ids = [i for i, _ in pairs]
            results = collection.get(ids=ids)
            found_ids = set(results["ids"])

            pairs = [(i, r) for (i, r) in pairs if i not in found_ids]
            if not pairs:
                continue

            if rate_limit is not None:
                rate_limiter(
                    n_docs=len(pairs),
                    rate_limit=rate_limit,
                    next_allowed_time=next_allowed_time,
                )

            collection.add(
                ids=[i for i, _ in pairs],
                documents=[r.expression for _, r in pairs],
                metadatas=[r.to_dict() for _, r in pairs],
            )
