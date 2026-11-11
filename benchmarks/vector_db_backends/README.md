# Vector database backends benchmark

Compares ChromaDB and Pinecone as the vector database of the expression index
([issue #74](https://github.com/precisiondata/aatm/issues/74)):

- **Query time**, unfiltered and with the domain and vocabulary filters the OMOP builder
  and the concept search agent use (`Measurement`, `Measurement + LOINC`, and
  `Condition`, which fits lab queries poorly).
- **Recall@k**, measured as in [#45](https://github.com/precisiondata/aatm/pull/45).
  Both indexes are approximate, so there is no exact answer to compare with. For each
  query, the reference is the `top_k` nearest records found by any method. Distances
  mean the same on both backends (squared Euclidean, see `aatm.vector_stores`), so the
  results can be pooled.

Each backend runs with its filter strategies. ChromaDB runs with `native` (its own
filtered query, which applies the filter before the search) and with
`overfetch_fallback` (the default). Pinecone runs with `native` (the default, which
filters during the search) and with `overfetch_fallback`.

## Run

From the repository root, with the same model's index in ChromaDB (local or Chroma
Cloud) and in Pinecone (needs `PINECONE_API_KEY` and `uv sync --extra pinecone`):

```bash
uv run aatm init --embedding-model text-embedding-3-small
uv run aatm vector-db upload --to pinecone -e text-embedding-3-small
uv run python benchmarks/vector_db_backends/run_benchmark.py --retriever text-embedding-3-small
```

The queries are the first 20 lab texts of the [#33 benchmark](../concept_search/README.md)
and 10 texts that are not lab results. Query embeddings are computed once, so timings
exclude the embedding call. Each query runs `--repeats` times (3 by default) and the
median is kept. Results go to `results/<retriever>.csv`, one row per query, filter and
method.

To compare **selection quality** on the #33 lab-results benchmark, run it once per
backend:

```bash
uv run python benchmarks/concept_search/run_benchmark.py --retriever text-embedding-3-small --vector-db local
uv run python benchmarks/concept_search/run_benchmark.py --retriever text-embedding-3-small --vector-db pinecone
```

## Results

Not run yet: this needs a Pinecone project holding the full index (about 2.5 million
expressions).
