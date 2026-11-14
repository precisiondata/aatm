# Vector database backends benchmark

Compares ChromaDB, Pinecone and PostgreSQL with pgvector as the vector database of the
expression index ([issue #74](https://github.com/precisiondata/aatm/issues/74),
[issue #111](https://github.com/precisiondata/aatm/issues/111)):

- **Query time**, unfiltered and with the domain and vocabulary filters the OMOP builder
  and the concept search agent use (`Measurement`, `Measurement + LOINC`, and
  `Condition`, which fits lab queries poorly).
- **Recall@k**, measured as in [#45](https://github.com/precisiondata/aatm/pull/45).
  With the `pgvector-exact` backend (PostgreSQL searching without its HNSW index), the
  reference for each query is its exact `top_k` nearest records. Without it, every index
  is approximate, so the reference is the `top_k` nearest records found by any method.
  Distances mean the same on every backend (squared Euclidean, see
  `aatm.vector_stores`), so the results can be pooled.

Each backend runs with its filter strategies. ChromaDB runs with `native` (its own
filtered query, which applies the filter before the search) and with
`overfetch_fallback` (the default). Pinecone runs with `native` (the default, which
filters during the search) and with `overfetch_fallback`. PostgreSQL runs the same
two, and `pgvector-exact` runs `native` only.

`--backends` picks the backends, among `chroma`, `pinecone`, `pgvector` and
`pgvector-exact` (default `chroma,pinecone`).

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

With PostgreSQL (needs `AATM_PG_DSN` and `uv sync --extra pgvector`; see
[PostgreSQL with pgvector](../../docs/getting-started/env.md#postgresql-with-pgvector)):

```bash
uv run aatm vector-db upload --to pgvector -e text-embedding-3-small
uv run python benchmarks/vector_db_backends/run_benchmark.py --retriever text-embedding-3-small --backends chroma,pgvector,pgvector-exact
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

### PostgreSQL with pgvector on the same subset (2026-11-14)

Setup: the same 50,007 expressions of `embeddinggemma-300M`, copied from the subset's
local ChromaDB database with `aatm vector-db upload --to pgvector` (16 s, plus 16 s to
build the HNSW and metadata indexes). PostgreSQL 17.12 with pgvector 0.8.7 ran in the
`pgvector/pgvector:pg17` Docker image on the same machine as ChromaDB 1.5.9, with
AATM's defaults (`ef_search=100`, `relaxed_order` iterative scans, HNSW `m=16`,
`ef_construction=64`). Same queries and settings as above. Here the reference is
**exact**: `pgvector-exact`, so recall is against the true 20 nearest records (above,
it was against the union of what the methods found).

| filter | method | median s | mean s | recall@20 mean | worst query |
|---|---|---|---|---|---|
| none | ChromaDB | 0.030 | 0.032 | 0.985 | 0.85 |
| none | pgvector | **0.004** | 0.004 | 0.990 | 0.90 |
| none | pgvector exact | 0.048 | 0.049 | 1.000 | 1.00 |
| Measurement | ChromaDB `native` | 0.070 | 0.073 | 0.990 | 0.90 |
| Measurement | ChromaDB `overfetch_fallback` | 0.036 | 0.041 | 0.985 | 0.90 |
| Measurement | pgvector `native` | 0.022 | 0.020 | **0.998** | 0.95 |
| Measurement | pgvector `overfetch_fallback` | 0.019 | 0.026 | 0.987 | 0.85 |
| Measurement | pgvector exact | 0.021 | 0.022 | 1.000 | 1.00 |
| Measurement + LOINC | ChromaDB `native` | 0.075 | 0.078 | 0.990 | 0.90 |
| Measurement + LOINC | ChromaDB `overfetch_fallback` | 0.037 | 0.064 | 0.983 | 0.85 |
| Measurement + LOINC | pgvector `native` | 0.017 | 0.018 | **1.000** | 1.00 |
| Measurement + LOINC | pgvector `overfetch_fallback` | 0.019 | 0.036 | 0.990 | 0.85 |
| Measurement + LOINC | pgvector exact | 0.017 | 0.017 | 1.000 | 1.00 |
| Condition | ChromaDB `native` | 0.080 | 0.077 | 1.000 | 1.00 |
| Condition | ChromaDB `overfetch_fallback` | 0.036 | 0.041 | 1.000 | 1.00 |
| Condition | pgvector `native` | 0.023 | 0.023 | 1.000 | 1.00 |
| Condition | pgvector `overfetch_fallback` | 0.018 | 0.021 | 1.000 | 1.00 |
| Condition | pgvector exact | 0.023 | 0.023 | 1.000 | 1.00 |

What it shows:

- **pgvector's own filter (`native`) is the right default**: 2 to 4 times faster than
  ChromaDB's filtered query and its over-fetch, with the best recall (0.998 to 1.0).
- On a subset this small, PostgreSQL answers the filtered queries in about the time of
  an exact search: when a filter matches a few thousand rows, its planner reads them
  through the metadata index and sorts them by distance, which is exact. On the full
  index, filters match hundreds of thousands of rows, and the HNSW index with
  iterative scans should take over. That remains to be measured.
- Unfiltered, the HNSW search takes 4 ms and finds 99% of the true neighbours.

Results: `results/embeddinggemma-300M--subset-50k-pgvector.csv`.

### Full index: ChromaDB and PostgreSQL with pgvector (2026-11-14)

Setup: the full `embeddinggemma-300M` index (768 dimensions, float32), about 2.2 million
expressions, in local ChromaDB 1.5.9 and in PostgreSQL 17.12 with pgvector 0.8.7 (the
container `aatm init` starts, on the same machine). The PostgreSQL table was built with
`aatm init --vector-db pgvector`: HNSW with pgvector's defaults (`m=16`,
`ef_construction=64`), searched with AATM's defaults (`ef_search=100`, `relaxed_order`
iterative scans). Same 30 queries, `top_k=20`, 3 runs per query with the median kept.
Recall is against the **exact** 20 nearest records (`pgvector-exact`, which searches
without the HNSW index). Pinecone was not run: the full index needs a paid plan.

```bash
uv run python benchmarks/vector_db_backends/run_benchmark.py --retriever embeddinggemma-300M --backends chroma,pgvector,pgvector-exact --label full
```

| filter | method | median s | mean s | recall@20 mean | worst query |
|---|---|---|---|---|---|
| none | ChromaDB | 0.030 | 0.031 | 0.972 | 0.70 |
| none | pgvector | **0.004** | 0.004 | 0.963 | 0.85 |
| none | pgvector exact | 1.932 | 1.942 | 1.000 | 1.00 |
| Measurement | ChromaDB `native` | 5.440 | 5.448 | 0.955 | 0.80 |
| Measurement | ChromaDB `overfetch_fallback` (default) | 0.050 | 0.458 | 0.953 | 0.75 |
| Measurement | pgvector `native` (default) | **0.005** | 0.017 | 0.877 | 0.30 |
| Measurement | pgvector `overfetch_fallback` | 0.044 | 0.075 | 0.918 | 0.30 |
| Measurement | pgvector exact | 0.579 | 0.579 | 1.000 | 1.00 |
| Measurement + LOINC | ChromaDB `native` | 4.634 | 4.625 | 0.947 | 0.60 |
| Measurement + LOINC | ChromaDB `overfetch_fallback` (default) | 0.050 | 0.887 | 0.952 | 0.60 |
| Measurement + LOINC | pgvector `native` (default) | 0.927 | 0.837 | **0.993** | 0.85 |
| Measurement + LOINC | pgvector `overfetch_fallback` | 0.043 | 0.236 | 0.952 | 0.70 |
| Measurement + LOINC | pgvector exact | 0.940 | 0.943 | 1.000 | 1.00 |
| Condition | ChromaDB `native` | 7.291 | 7.028 | 0.963 | 0.85 |
| Condition | ChromaDB `overfetch_fallback` (default) | 0.048 | 0.287 | 0.968 | 0.80 |
| Condition | pgvector `native` (default) | **0.007** | 0.009 | 0.955 | 0.75 |
| Condition | pgvector `overfetch_fallback` | 0.025 | 0.042 | 0.970 | 0.80 |
| Condition | pgvector exact | 0.605 | 0.608 | 1.000 | 1.00 |

How often each method's nearest record is the true nearest (top-1), and recall at 5:

| filter | ChromaDB default | pgvector `native` | pgvector `overfetch_fallback` |
|---|---|---|---|
| none | top-1 0.90, r@5 0.973 | top-1 1.00, r@5 0.973 | — |
| Measurement | top-1 0.97, r@5 0.920 | top-1 0.93, r@5 0.913 | top-1 1.00, r@5 0.960 |
| Measurement + LOINC | top-1 0.90, r@5 0.873 | top-1 1.00, r@5 0.987 | top-1 0.97, r@5 0.927 |
| Condition | top-1 0.93, r@5 0.960 | top-1 0.87, r@5 0.907 | top-1 0.90, r@5 0.960 |

What it shows:

- **ChromaDB's own filtered query takes 5 to 7 s** on the full index, as in #45, which is
  why its retriever over-fetches by default (about 50 ms). **pgvector filters during the
  search in 5 to 7 ms**, and its exact search takes 0.6 s with a filter and 1.9 s without.
- **pgvector's filtered search recalls less on broad filters.** With Measurement it finds
  0.877 of the true 20 nearest (ChromaDB's over-fetch 0.953), with Condition 0.955 (0.968).
  The worst cases are texts the filter fits poorly: for "metformin 500 mg tablet" the true
  nearest Measurement records rank #3,695 to #8,681 among all records and score about
  0.2, so missing them costs nothing for coding. But lab tests lose some too: "INR" in
  Measurement finds the right top results and 0.55 of the top 20.
- **The limit is the HNSW graph, not the search settings.** Raising `ef_search` from 100
  to 1000 (the maximum) lifts Measurement only to 0.918 at 27 ms; strict ordering is
  worse; keeping the 20 nearest of more candidates changes nothing. A denser graph
  helps: see below.
- **Narrow filters are exact on pgvector.** For Measurement + LOINC, PostgreSQL reads the
  matching rows through the metadata index and sorts them by distance: recall 0.993 in
  about 0.9 s, faster than ChromaDB's 4.6 s and more accurate than its over-fetch.
- Unfiltered, pgvector's HNSW search takes 4 ms with recall 0.963 (ChromaDB 30 ms, 0.972).

Selection quality on the #33 lab-results benchmark was not run here; it calls an LLM.

Results: `results/embeddinggemma-300M--full.csv`.

#### Denser HNSW graphs

The same table, with the HNSW index rebuilt by `create_pg_indexes` (16 GB of
`maintenance_work_mem` and 7 workers, from the container `aatm init` sizes), and the same
30 queries measured against the exact results above. Recall@20, with the worst query in
brackets, and the median time per query:

| HNSW graph | build | `ef_search` | none | Measurement | Measurement + LOINC | Condition |
|---|---|---|---|---|---|---|
| `m=16`, `ef_construction=64` (pgvector's default) | 35 min (64 MB, 2 workers) | 100 | 0.963 (0.85), 5 ms | 0.877 (0.30), 7 ms | 0.963 (0.70), 941 ms | 0.955 (0.75), 9 ms |
| | | 400 | 0.977 (0.85), 11 ms | 0.890 (0.30), 13 ms | 0.968 (0.70), 942 ms | 0.958 (0.75), 14 ms |
| `m=16`, `ef_construction=128` | 6.7 min | 100 | 0.982 (0.85), 8 ms | 0.908 (0.40), 11 ms | 0.980 (0.85), 929 ms | 0.945 (0.55), 12 ms |
| | | 400 | 0.988 (0.85), 13 ms | 0.922 (0.45), 14 ms | 0.992 (0.85), 925 ms | 0.970 (0.75), 14 ms |
| **`m=24`, `ef_construction=128`** | 8.7 min | 100 | **0.982** (0.85), 8 ms | **0.925** (0.45), 12 ms | 0.978 (0.85), 948 ms | **0.967** (0.80), 16 ms |
| | | 400 | 0.982 (0.85), 18 ms | 0.927 (0.45), 22 ms | 0.985 (0.85), 947 ms | 0.973 (0.80), 19 ms |
| ChromaDB, `overfetch_fallback` (for reference) | | | 0.972 (0.70), 30 ms | 0.953 (0.75), 50 ms | 0.952 (0.60), 50 ms | 0.968 (0.80), 48 ms |

- **`m=24`, `ef_construction=128` closes most of the gap** at the default `ef_search`:
  Measurement from 0.877 to 0.925 (ChromaDB 0.953), Condition from 0.955 to 0.967
  (0.968), unfiltered from 0.963 to 0.982 (0.972), still in 8 to 16 ms against ChromaDB's
  30 to 50 ms. Its index is no larger on disk (9.4 GB): with 768 dimensions, each vector
  fills most of a page either way.
- Raising `ef_search` to 400 on top adds little; the graph matters more than the search.
- Measurement's worst query is still "metformin 500 mg tablet", whose true matches rank
  thousands deep (see above).
- Measurement + LOINC is answered exactly either way (about 0.9 s); its recall below 1.0
  comes from ties between records at the same distance.
