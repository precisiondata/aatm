"""Compare ChromaDB and Pinecone on the full expression index: query time and recall.

For each query and filter, every (backend, filter strategy) pair retrieves ``top_k``
records. Query embeddings are computed once and reused, so timings exclude the
embedding call. As in issue #45, the reference for a query is the ``top_k`` nearest
records found by *any* method (distances are the same on both backends, see
`aatm.vector_stores`), and recall@k is the share of the reference a method found.

Run from the repository root, with the same model's index in both databases (build it
with ``aatm init``, then copy it with ``aatm vector-db upload --to pinecone``):

```bash
uv run python benchmarks/vector_db_backends/run_benchmark.py --retriever text-embedding-3-small
```

``--chroma`` picks the ChromaDB side (``local``, the default, or ``cloud``). Writes one
row per query, filter and method to ``results/<retriever>.csv`` and prints a summary.
"""

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

# pandas before the retrievers (torch): on some Windows setups, loading pyarrow after
# torch crashes the interpreter.
import pandas as pd

from aatm.omop.coding import retriever_filter
from aatm.registries.retrievers import load_retriever

HERE = Path(__file__).parent
LAB_CASES = HERE.parent / "concept_search" / "lab_results.csv"

# Texts that are not lab results, so the Measurement filter fits them poorly.
OTHER_TEXTS = [
    "type 2 diabetes mellitus",
    "essential hypertension",
    "acute myocardial infarction",
    "metformin 500 mg tablet",
    "insulin pump",
    "appendectomy",
    "chronic obstructive pulmonary disease",
    "asthma exacerbation",
    "atrial fibrillation",
    "urinary tract infection",
]

FILTERS: dict[str, dict[str, Any] | None] = {
    "none": None,
    "Measurement": retriever_filter("Measurement", None),
    "Measurement + LOINC": retriever_filter("Measurement", ["LOINC"]),
    "Condition": retriever_filter("Condition", None),
}

STRATEGIES = {
    "chroma": ["native", "overfetch_fallback"],
    "pinecone": ["native", "overfetch_fallback"],
}


class CachedEmbeddings:
    """Return precomputed query embeddings, so timings exclude the embedding call."""

    def __init__(self, texts: list[str], embeddings: Any):
        self.by_text = dict(zip(texts, embeddings))

    def __call__(self, input: list[str]) -> list[Any]:
        return [self.by_text[text] for text in input]


def load_queries(n_lab: int) -> list[str]:
    lab = pd.read_csv(LAB_CASES)["text"].drop_duplicates().tolist()[:n_lab]
    return lab + OTHER_TEXTS


def run(args: argparse.Namespace) -> pd.DataFrame:
    queries = load_queries(args.lab_queries)
    retrievers = {
        "chroma": load_retriever(args.retriever, vector_db=args.chroma),
        "pinecone": load_retriever(args.retriever, vector_db="pinecone"),
    }
    # Both backends hold the same vectors: embed once, with the model the index
    # was built with, and reuse the embeddings everywhere.
    embedding_function = retrievers["chroma"].embedding_function
    embeddings = list(
        getattr(embedding_function, "embed_query", embedding_function)(input=queries)
    )
    for retriever in retrievers.values():
        retriever.embedding_function = CachedEmbeddings(queries, embeddings)

    rows = []
    for filter_name, where in FILTERS.items():
        for backend, retriever in retrievers.items():
            strategies = STRATEGIES[backend] if where else ["native"]
            for strategy in strategies:
                print(f"{backend} / {strategy} / {filter_name}...")
                for query in queries:
                    seconds = []
                    for _ in range(args.repeats):
                        started = time.perf_counter()
                        (hits,) = retriever.retrieve(
                            [query],
                            where=where,
                            top_k=args.top_k,
                            filter_strategy=strategy,
                        ).results
                        seconds.append(time.perf_counter() - started)
                    rows.append(
                        {
                            "backend": backend,
                            "strategy": strategy,
                            "filter": filter_name,
                            "query": query,
                            "seconds": statistics.median(seconds),
                            "ids": [h.expression_id for h in hits],
                            "distances": [h.distance for h in hits],
                        }
                    )

    df = pd.DataFrame(rows)
    df["recall"] = recall_against_union(df, args.top_k)
    return df


def recall_against_union(df: pd.DataFrame, top_k: int) -> pd.Series:
    """Recall@k of each row against the k nearest records any method found."""
    recall = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby(["filter", "query"]):
        nearest: dict[str, float] = {}
        for ids, distances in zip(group.ids, group.distances):
            for identifier, distance in zip(ids, distances):
                nearest[identifier] = min(distance, nearest.get(identifier, distance))
        reference = set(sorted(nearest, key=nearest.__getitem__)[:top_k])
        for index, ids in zip(group.index, group.ids):
            recall[index] = len(reference & set(ids)) / max(len(reference), 1)
    return recall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retriever", default="text-embedding-3-small")
    parser.add_argument("--chroma", default="local", choices=["local", "cloud"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--lab-queries", type=int, default=20)
    parser.add_argument(
        "--repeats", type=int, default=3, help="Runs per query; the median is kept."
    )
    args = parser.parse_args()

    df = run(args)
    out = HERE / "results" / f"{args.retriever}.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["filter", "backend", "strategy"], sort=False)
        .agg(
            median_s=("seconds", "median"),
            mean_s=("seconds", "mean"),
            recall_mean=("recall", "mean"),
            recall_worst=("recall", "min"),
        )
        .round(3)
    )
    print(summary.to_markdown())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
