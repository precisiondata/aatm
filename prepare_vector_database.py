from pathlib import Path
import pandas as pd
import chromadb
from aatm.data_models import ExpressionMetadata
import dotenv
from tqdm import tqdm

# Custom modules
from aatm.embedding_functions import GoogleEmbeddingFunction

dotenv.load_dotenv()


client = chromadb.PersistentClient()

collection = client.get_or_create_collection(
    "expressions",
    # embedding_function=GoogleEmbeddingFunction(model="gemini-embedding-001"),
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
        ids = [record.expression_id for record in records]

        # check if all ids alerady exist in the database
        results = collection.get(ids=ids)
        ids = set(ids)

        found_ids = set(results["ids"])
        if len(found_ids) == len(ids):
            continue

        # remove ids that already exist in the database
        ids = [id for id in ids if id not in found_ids]
        records = [record for record in records if record.expression_id in ids]

        collection.add(
            ids=ids,
            documents=[record.expression for record in records],
            metadatas=[record.to_dict() for record in records],
        )
