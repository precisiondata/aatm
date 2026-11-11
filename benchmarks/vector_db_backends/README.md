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

### On a subset (Pinecone free plan)

The full index does not fit Pinecone's free plan. `make_subset.py` picks a subset that
does, writes its IDs, and copies the same rows into a local ChromaDB database, so both
backends hold identical data:

```bash
uv run python benchmarks/vector_db_backends/make_subset.py --model embeddinggemma-300M --rows 50000
PINECONE_INDEX_PREFIX=aatm-subset uv run aatm vector-db upload --to pinecone -e embeddinggemma-300M --ids-file .aatm/subsets/embeddinggemma-300M.txt
PINECONE_INDEX_PREFIX=aatm-subset uv run python benchmarks/vector_db_backends/run_benchmark.py --retriever embeddinggemma-300M --chroma-dir .aatm/chroma_vector_dbs_subset/embeddinggemma-300M --label subset-50k
```

## Results

### 50,000-expression subset (2026-11-11)

Setup: `embeddinggemma-300M` (768 dimensions, float32). The subset is 50,007 expressions
of 12,622 concepts: the 30 concepts of the #33 gold codes, plus standard concepts
drawn at random (seed 0). That keeps the full index's mix: SNOMED Condition 21%,
SNOMED Observation 20%, LOINC Measurement 14%, RxNorm Drug 13%, and so on. ChromaDB
1.5.9 ran locally; Pinecone was a serverless index on the free plan (`aws`,
`us-east-1`). `top_k=20`, 30 queries (20 lab texts and 10 others), 3 runs per query
with the median kept, and embeddings computed beforehand.

| filter | method | median s | mean s | recall@20 mean | worst query |
|---|---|---|---|---|---|
| none | ChromaDB | 0.031 | 0.032 | 0.983 | 0.85 |
| none | Pinecone | 0.150 | 0.150 | **0.998** | 0.95 |
| Measurement | ChromaDB `native` | 0.072 | 0.074 | 0.982 | 0.90 |
| Measurement | ChromaDB `overfetch_fallback` | 0.037 | 0.042 | 0.977 | 0.90 |
| Measurement | Pinecone `native` | 0.150 | 0.152 | **0.998** | 0.95 |
| Measurement | Pinecone `overfetch_fallback` | 0.172 | 0.190 | 0.968 | 0.75 |
| Measurement + LOINC | ChromaDB `native` | 0.079 | 0.082 | 0.992 | 0.90 |
| Measurement + LOINC | ChromaDB `overfetch_fallback` | 0.038 | 0.065 | 0.985 | 0.85 |
| Measurement + LOINC | Pinecone `native` | 0.150 | 0.150 | **0.997** | 0.95 |
| Measurement + LOINC | Pinecone `overfetch_fallback` | 0.174 | 0.239 | 0.950 | 0.55 |
| Condition | ChromaDB `native` | 0.073 | 0.074 | 1.000 | 1.00 |
| Condition | ChromaDB `overfetch_fallback` | 0.037 | 0.042 | 1.000 | 1.00 |
| Condition | Pinecone `native` | 0.150 | 0.151 | 1.000 | 1.00 |
| Condition | Pinecone `overfetch_fallback` | 0.172 | 0.195 | 0.987 | 0.85 |

What it shows:

- **Pinecone's filtered queries cost nothing extra**: 0.150 s with or without a filter.
  The time is the network round trip from the benchmark machine: a request that does
  no search (index stats) also took 0.150 s. A machine closer to the index's region
  would see less.
- **Pinecone's own filter (`native`) has the best recall**, 0.997 to 1.0, and is the
  right default for it. The over-fetch is slower and less accurate on Pinecone: its
  deep unfiltered queries (1,000 to 5,000 results) recall worse than a filtered query of
  20.
- **On 50,000 expressions, local ChromaDB is faster** (0.03 to 0.08 s) because it
  needs no network. This says little about the full index: there, ChromaDB's filtered
  query took about 5 s and the over-fetch fell back to it for some queries (#45),
  while Pinecone's latency should depend little on the index size. That remains to be
  measured on a paid plan.

Cost on the free plan: the upload of 50,007 records took 4.6 minutes and about 0.85
million write units (16.6 KB of JSON per record). Queries on an index this small cost
the minimum, 0.25 read units each.

### Full index

Not run yet: the full index (about 2.5 million expressions, about 40 GB of upsert
requests for `embeddinggemma-300M`) needs a paid Pinecone plan.
