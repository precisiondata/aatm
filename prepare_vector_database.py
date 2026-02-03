from pathlib import Path
import time
import pandas as pd
import chromadb
from aatm.data_models import ExpressionMetadata
import dotenv
from tqdm import tqdm

# Custom modules
from aatm.retrievers import CHROMADB_RETRIEVER_MODEL_REGISTRY as model_registry

dotenv.load_dotenv()

model_name = "text-embedding-3-large"


# Rate limiting config
DOCS_PER_MIN = model_registry[model_name].get("rate_limit", None)
if DOCS_PER_MIN is not None and DOCS_PER_MIN <= 0:
    raise ValueError("DOCS_PER_MIN must be > 0")

next_allowed_time = time.monotonic()


def rate_limit(n_docs: int) -> None:
    """Sleep as needed so we don't exceed DOCS_PER_MIN."""
    global next_allowed_time
    now = time.monotonic()
    if now < next_allowed_time:
        time.sleep(next_allowed_time - now)
        now = time.monotonic()
    # Reserve time for this batch
    next_allowed_time = max(next_allowed_time, now) + n_docs * 60.0 / DOCS_PER_MIN


client = chromadb.PersistentClient(model_registry[model_name]["chromadb_path"])

collection = client.get_or_create_collection(
    model_registry[model_name]["collection_name"],
    embedding_function=model_registry[model_name]["embedding_function"](
        model=model_registry[model_name]["model_id"]
    ),
)


batch_size = 100  # max batch size = 100
datasets_base_path = Path("datasets")
for dataset_path in datasets_base_path.glob("*.csv"):
    expression_origin = dataset_path.stem
    df = pd.read_csv(dataset_path)
    df = df.drop_duplicates().dropna()
    for i in tqdm(
        range(0, len(df), batch_size),
        desc=f"Adding embeddings for {expression_origin}",
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

        if DOCS_PER_MIN is not None:
            rate_limit(n_docs=len(pairs))

        collection.add(
            ids=[i for i, _ in pairs],
            documents=[r.expression for _, r in pairs],
            metadatas=[r.to_dict() for _, r in pairs],
        )
