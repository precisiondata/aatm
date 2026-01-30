from pathlib import Path
import time
import pandas as pd
import chromadb
from aatm.data_models import ExpressionMetadata
import dotenv
from tqdm import tqdm

# Custom modules
from aatm.embedding_functions import GoogleEmbeddingFunction

dotenv.load_dotenv()

# Rate limiting config
DOCS_PER_MIN = 3_000  # GCP limits to 3,000 docs per minute
if DOCS_PER_MIN <= 0:
    raise ValueError("DOCS_PER_MIN must be > 0")

SECONDS_PER_DOC = 60.0 / DOCS_PER_MIN

next_allowed_time = time.monotonic()


def rate_limit(n_docs: int) -> None:
    """Sleep as needed so we don't exceed DOCS_PER_MIN."""
    global next_allowed_time
    now = time.monotonic()
    if now < next_allowed_time:
        time.sleep(next_allowed_time - now)
        now = time.monotonic()
    # Reserve time for this batch
    next_allowed_time = max(next_allowed_time, now) + n_docs * SECONDS_PER_DOC


client = chromadb.PersistentClient()

collection = client.get_or_create_collection(
    "expressions",
    embedding_function=GoogleEmbeddingFunction(model="gemini-embedding-001"),
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
        records_dict = [record.to_dict() for record in records]
        ids = [record.expression_id for record in records]

        # # check if all ids alerady exist in the database
        results = collection.get(ids=ids)

        found_metadatas = results["metadatas"][0]
        for idx, (new_metadata, old_metadata) in enumerate(
            zip(records_dict, found_metadatas)
        ):
            if new_metadata == old_metadata:
                records.pop(idx)
                ids.pop(idx)

        if len(records) > 0:
            # rate_limit(len(records))
            collection.update(
                ids=ids,
                metadatas=[record.to_dict() for record in records],
            )
