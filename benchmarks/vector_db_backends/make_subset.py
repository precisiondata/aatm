"""Make a subset of a model's local index, small enough for a Pinecone free plan.

The subset keeps whole concepts: every expression (name and synonyms) of the
concepts it picks. It holds:

- the concepts of the #33 lab-results benchmark's gold codes (LOINC), so lab
  queries have their right answers in it;
- standard concepts drawn at random (``--seed``) until about ``--rows``
  expressions, which keeps the index's mix of domains and vocabularies, so
  domain and vocabulary filters are as selective as on the full index.

It writes the subset's expression IDs to ``.aatm/subsets/<model>.txt`` and copies the
same rows, without re-embedding, into a local ChromaDB database at
``.aatm/chroma_vector_dbs_subset/<model>``. Both backends can then be benchmarked on
the same data:

```bash
uv run python benchmarks/vector_db_backends/make_subset.py --model embeddinggemma-300M --rows 50000
PINECONE_INDEX_PREFIX=aatm-subset uv run aatm vector-db upload --to pinecone -e embeddinggemma-300M --ids-file .aatm/subsets/embeddinggemma-300M.txt
```

On Pinecone's free plan, writes are the limit: an upsert costs one write unit per KB,
and a 768-dimension record is about 16.6 KB as JSON, so 50,000 records cost about
0.85 million of the 2 million monthly write units.
"""

import argparse
from pathlib import Path

# pandas before the retrievers (torch): on some Windows setups, loading pyarrow after
# torch crashes the interpreter.
import numpy as np
import pandas as pd

from aatm.data_models import ExpressionMetadata
from aatm.local_database_utils import get_existing_collection, release_chromadb_clients
from aatm.registries.retrievers import CHROMADB_RETRIEVER_MODEL_REGISTRY
from aatm.vector_db import ChromaConnection
from aatm.vector_stores import ChromaVectorStore

HERE = Path(__file__).parent
LAB_CASES = HERE.parent / "concept_search" / "lab_results.csv"
DATASETS = Path(".aatm/datasets")
PAGE = 5000


def gold_codes() -> set[str]:
    cases = pd.read_csv(LAB_CASES, dtype=str, keep_default_na=False)
    return {
        code
        for gold, also in zip(cases.gold, cases.also_accept)
        for code in (gold, *also.split(";"))
        if code
    }


def load_rows() -> pd.DataFrame:
    """Read the mapping datasets the way the index builder does."""
    parts = []
    for path in sorted(DATASETS.glob("*.csv")):
        df = pd.read_csv(path, low_memory=False).drop_duplicates().dropna()
        df["expression_origin"] = path.stem
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def pick_concepts(rows: pd.DataFrame, target: int, seed: int) -> set:
    codes = gold_codes()
    gold = rows[
        (rows.std_vocabulary_id == "LOINC")
        & rows.std_vocabulary_code.astype(str).isin(codes)
    ].std_concept_id.unique()
    print(f"{len(gold)} concepts for the {len(codes)} lab benchmark codes")

    sizes = rows.groupby("std_concept_id").size()
    chosen = set(gold)
    n = int(sizes.loc[list(chosen)].sum())
    order = np.random.default_rng(seed).permutation(sizes.index.to_numpy())
    for concept in order:
        if n >= target:
            break
        if concept not in chosen:
            chosen.add(concept)
            n += int(sizes.loc[concept])
    return chosen


def expression_ids(rows: pd.DataFrame) -> list[str]:
    ids: dict[str, None] = {}
    for record in rows.to_dict("records"):
        metadata = ExpressionMetadata(**record)
        ids.setdefault(metadata.expression_id, None)  # type: ignore[arg-type]
    return list(ids)


def copy_subset(model: str, ids: list[str], out_dir: Path) -> int:
    """Copy the rows with ``ids`` from the model's local index to ``out_dir``."""
    spec = CHROMADB_RETRIEVER_MODEL_REGISTRY[model]
    source = ChromaConnection.local(str(Path(spec.chromadb_path).resolve())).client()
    full = get_existing_collection(source, spec.collection_name)
    if full is None:
        raise SystemExit(f"No local index for {model} at {spec.chromadb_path}")

    target = ChromaConnection.local(str(out_dir.resolve())).client()
    if get_existing_collection(target, spec.collection_name) is not None:
        target.delete_collection(spec.collection_name)
    store = ChromaVectorStore(
        target.create_collection(
            spec.collection_name, metadata=full.metadata, embedding_function=None
        )
    )
    copied = 0
    for start in range(0, len(ids), PAGE):
        page = full.get(
            ids=ids[start : start + PAGE],
            include=["embeddings", "documents", "metadatas"],
        )
        store.write(
            page["ids"], page["embeddings"], page["documents"], page["metadatas"]
        )
        copied += len(page["ids"])
    count = store.count()
    del full, source, target, store
    release_chromadb_clients()
    assert count == copied, (count, copied)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="embeddinggemma-300M")
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows()
    print(f"{len(rows)} rows in the datasets")
    concepts = pick_concepts(rows, args.rows, args.seed)
    subset = rows[rows.std_concept_id.isin(concepts)]
    ids = expression_ids(subset)
    print(f"{len(concepts)} concepts, {len(ids)} expressions")
    print(
        subset.groupby(["std_vocabulary_id", "std_domain_id"])
        .size()
        .sort_values(ascending=False)
        .head(12)
        .to_string()
    )

    ids_file = Path(".aatm/subsets") / f"{args.model}.txt"
    ids_file.parent.mkdir(parents=True, exist_ok=True)
    ids_file.write_text("\n".join(ids) + "\n")
    print(f"Wrote {ids_file}")

    out_dir = Path(".aatm/chroma_vector_dbs_subset") / args.model
    count = copy_subset(args.model, ids, out_dir)
    print(f"Copied {count} rows to {out_dir}")


if __name__ == "__main__":
    main()
