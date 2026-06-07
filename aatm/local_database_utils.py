"""Utilities for building and managing local terminology-mapping data assets.

This module provides helper functions for constructing the local resources
required by the terminology-mapping workflow. These resources include a local
SQLite vocabulary database, derived CSV datasets used for terminology mapping,
and a local vector database for semantic retrieval.

The module also includes a simple rate-limiting helper for embedding generation
workloads and interactive prompts to prevent accidental overwrites of existing
artifacts.
"""

import time
from typing import Any, Hashable, List, Tuple
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
    """Apply a document-based rate limit and return the next allowed execution time.

    This helper ensures that document processing respects a maximum throughput
    expressed in documents per minute. If the current time is earlier than the
    next permitted execution time, the function sleeps until processing is
    allowed. It then reserves time for the current batch and returns the updated
    timestamp for the next permitted request.

    Args:
        n_docs: Number of documents that will be processed in the current batch.
        rate_limit: Maximum allowed processing rate in documents per minute.
        next_allowed_time: Monotonic timestamp representing when the next batch
            is allowed to start.

    Returns:
        The updated monotonic timestamp indicating when the next batch may be
        processed.

    Raises:
        ValueError: In contexts where the caller validates that the provided
            rate limit is invalid, such as non-positive values.

    Notes:
        This function uses ``time.monotonic()`` so that elapsed-time calculations
        are not affected by system clock changes.
    """
    now = time.monotonic()
    if now < next_allowed_time:
        time.sleep(next_allowed_time - now)
        now = time.monotonic()
    # Reserve time for this batch
    next_allowed_time = max(next_allowed_time, now) + n_docs * 60.0 / rate_limit
    return next_allowed_time


def build_local_sqlite_vocab_database(vocab_dir: Path) -> None:
    """Build a local SQLite vocabulary database from OMOP vocabulary files.

    This function reads vocabulary files from a directory, converts each file
    into a SQLite table, and stores the result in a local database file. If a
    database already exists, the user is prompted to either skip rebuilding it
    or overwrite the existing database.

    CSV parsing behavior depends on the table being imported. The
    ``source_to_concept_map`` file is read as comma-separated, while the other
    vocabulary files are read as tab-separated.

    Args:
        vocab_dir: Path to the directory containing the vocabulary files to be
            imported into the local SQLite database.

    Returns:
        None.

    Side Effects:
        Creates or overwrites the local SQLite database at ``.aatm/omop.db``.
        Prompts the user for confirmation when an existing database is found.
        Writes one table per vocabulary file into the database.

    Raises:
        FileNotFoundError: If the expected vocabulary files are not present in
            the provided directory.
        sqlite3.Error: If a database connection or write operation fails.
        pandas.errors.ParserError: If a vocabulary file cannot be parsed.
    """

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
    """Generate terminology-mapping datasets from SQL templates and a local database.

    This function loads SQL command templates from a packaged YAML file, formats
    them using the provided list of standard vocabularies, executes each query
    against the local OMOP SQLite database, and saves the resulting datasets as
    CSV files in the local datasets directory.

    If mapping datasets already exist, the user is prompted to either skip
    regeneration or overwrite the existing files.

    Args:
        standard_vocabularies: List of standard vocabulary names used to
            parameterize the SQL queries that generate the mapping datasets.

    Returns:
        None.

    Raises:
        ValueError: If no standard vocabularies are provided.
        FileNotFoundError: If the SQL command resource file cannot be located.
        sqlite3.Error: If an error occurs while querying the local SQLite
            database.
        yaml.YAMLError: If the SQL command YAML file cannot be parsed.

    Side Effects:
        Creates the ``.aatm/datasets`` directory when needed.
        Writes one CSV file per SQL command to the local datasets directory.
        Prompts the user before overwriting existing dataset files.
    """
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
    """Build or repair a local vector database for terminology mapping.

    This function creates a persistent vector database from the generated
    terminology-mapping datasets. It loads records from local CSV files,
    converts them into structured metadata objects, avoids duplicate or already
    indexed entries, and stores embeddings in a ChromaDB collection using the
    configured embedding model.

    If a vector database already exists, the user may choose to skip the
    operation, overwrite the existing database, or repair it. Optional
    rate-limiting can be applied to control embedding throughput for providers
    with request limits.

    Args:
        embedding_model_name: Registry key identifying the embedding model
            configuration to use.
        vector_db_dir: Optional path to the vector database directory. If not
            provided, the default path from the retriever model registry is used.
        rate_limit: Optional maximum number of documents to embed per minute. If
            not provided, the default value from the model registry is used.
        batch_size: Number of records to process and add to the vector database
            in each batch.

    Returns:
        None.

    Raises:
        ValueError: If ``vector_db_dir`` is provided but does not exist.
        ValueError: If ``rate_limit`` is provided and is not greater than zero.
        FileNotFoundError: If required dataset files are missing.
        sqlite3.Error: If dependent local resources are unavailable or invalid.
        Exception: If the vector database client, embedding function, or
            collection operations fail.

    Side Effects:
        Creates, repairs, or overwrites a local persistent vector database.
        Reads CSV datasets from ``.aatm/datasets``.
        Prompts the user before modifying an existing vector database.
        Generates embeddings and stores documents and metadata in the target
        collection.

    Notes:
        The function performs lazy imports for some dependencies to reduce
        startup overhead for workflows that do not require vector database
        creation.
    """
    # lazy loading for performance
    from aatm.registries.retrievers import (
        CHROMADB_RETRIEVER_MODEL_REGISTRY as model_registry,
    )

    # check if vector database directory provided exists
    if vector_db_dir is not None and not vector_db_dir.exists():
        raise ValueError("vector_db_dir does not exist")

    if vector_db_dir is None:
        vector_db_dir = Path(model_registry[embedding_model_name].chromadb_path)

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
        rate_limit = model_registry[embedding_model_name].rate_limit

    if rate_limit is not None and rate_limit <= 0:
        raise ValueError("rate_limit must be > 0")

    if rate_limit is not None:
        next_allowed_time = time.monotonic()

    # lazy import for performance
    import chromadb

    console.print("Creating local vector database...")
    client = chromadb.PersistentClient(
        model_registry[embedding_model_name].chromadb_path
        if vector_db_dir is None
        else vector_db_dir
    )
    collection = client.get_or_create_collection(
        model_registry[embedding_model_name].collection_name,
        embedding_function=model_registry[embedding_model_name].embedding_function_cls(
            model=model_registry[embedding_model_name].model_id
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
            records: list[dict[Hashable, Any]] = df.iloc[i : i + batch_size].to_dict(
                "records"
            )
            expression_metadatas: list[ExpressionMetadata] = [
                ExpressionMetadata(**record, expression_origin=expression_origin)  # type: ignore[arg-type]
                for record in records
            ]

            pairs = [(r.expression_id, r) for r in expression_metadatas]

            unique_pairs = list(set(pairs))

            ids = [i for i, _ in unique_pairs]
            assert all([identifier is not None for identifier in ids]), (
                f"Missing identifier in {expression_origin}"
            )
            results = collection.get(ids=ids)  # type: ignore[arg-type]
            found_ids = set(results["ids"])

            filtered_unique_pairs: List[Tuple[str, ExpressionMetadata]] = [
                (i, r)
                for (i, r) in unique_pairs
                if i not in found_ids and i is not None
            ]
            if not filtered_unique_pairs:
                continue

            if rate_limit is not None:
                rate_limiter(
                    n_docs=len(filtered_unique_pairs),
                    rate_limit=rate_limit,
                    next_allowed_time=next_allowed_time,
                )

            collection.add(
                ids=[i for i, _ in filtered_unique_pairs],
                documents=[r.expression for _, r in filtered_unique_pairs],  # type: ignore[misc]
                metadatas=[r.to_dict() for _, r in filtered_unique_pairs],
            )
