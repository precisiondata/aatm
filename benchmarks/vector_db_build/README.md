# Vector database build benchmark

Times a full build of a local model's vector database from the mapping datasets
([issue #61](https://github.com/precisiondata/aatm/issues/61)). It checks that the
batching changes in [#60](https://github.com/precisiondata/aatm/pull/60) hold on a
full build, not only on the 40,000-row sample they were measured on.

`run_build.py` builds the database in a scratch directory, never in the project's
`.aatm/chroma_vector_dbs`, and records:
- build time and rows embedded;
- the batch setting calibration chose;
- the build profiler's per-stage totals;
- peak dedicated GPU memory and peak shared GPU memory (Windows);
- any warnings logged during the build;
- the top-10 search results for 130 queries: the 30 lab texts of
  `benchmarks/concept_search` plus 100 expressions sampled from the datasets.

With `--repair-sample N` it deletes N random rows and runs a repair. It tests whatever
`aatm` is importable, so two versions can be compared by changing `PYTHONPATH`.
`compare.py` prints the comparison.

## Run

After `aatm init` has built the mapping datasets:

```bash
python benchmarks/vector_db_build/run_build.py --label after --code-ref "$(git rev-parse --short HEAD)" --work-dir /path/to/scratch --datasets .aatm/datasets --out benchmarks/vector_db_build/results/embeddinggemma-300M__after.json --repair-sample 5000
```

```bash
python benchmarks/vector_db_build/compare.py benchmarks/vector_db_build/results/embeddinggemma-300M__before.json benchmarks/vector_db_build/results/embeddinggemma-300M__after.json
```

The scratch directory needs room for the datasets and one database: 12 GiB for
EmbeddingGemma-300M. The script deletes the database at the end unless `--keep-db` is
passed.

## Results (2026-11-10)

EmbeddingGemma-300M, float32 (its default), all four datasets: 2,458,830 rows. RTX 4090
(24 GB), Windows 11, PyTorch 2.11 + CUDA 12.8, sentence-transformers 5.5.1, ChromaDB
in a separate writer process. *Before* is `c153628`, `main` just before #60; *after*
is `main` with #60.

| | Before | After | Change |
|---|---|---|---|
| **Build time** | 37 min 46 s (2,266 s) | **36 min 6 s** (2,166 s) | **1.05× faster** |
| Rows/s, end to end | 1,085 | 1,135 | |
| Batch setting chosen | batch size 128 | token budget 4,096 | |
| **Encoding** (`embedding_function_call`) | 1,009 s | **678 s** | **1.49× faster** |
| Calibration | 17.1 s | 10.7 s | |
| ChromaDB adds (writer process) | 2,043 s | 2,006 s | |
| ChromaDB existing-ID lookups | 2,026 s | 1,988 s | |
| GPU waiting for the next prepared chunk | 1,186 s | 1,358 s | |
| GPU waiting for the writer queue | 16 s | 86 s | |
| **Peak dedicated GPU memory** (whole card) | 10.0 GiB | **3.2 GiB** | |
| Peak shared GPU memory | 0.24 GiB | 0.22 GiB | no spill |
| Database size | 11.92 GiB | 11.92 GiB | |
| Warnings (OOM backoff or other) | 0 | 0 | |

**Encoding is 1.49× faster, but the build is only 1.05× faster, because the build is
limited by ChromaDB.**
- **ChromaDB adds take about 2,000 s, close to the whole build in both versions.**
  The single writer process adds about 1,200 rows/s, however fast the GPU encodes.
- **Existing-ID lookups take another 2,000 s.** They run in the same writer process,
  in a second thread. Chunk preparation waits on them, so the GPU waits on chunk
  preparation for 1,358 s.
- **On a new database every lookup returns nothing**, so they're pure overhead
  on a fresh build. Encoding (678 s) now takes about a third of the build time.

**Search results are the same.**
- For 115 of the 130 queries, all ten distances match to within 1e-4.
- Top-1 agreement is 95.4% (124/130), and the top-10 overlap is 97.5%.
- Of the 6 different top results:
  - 4 are ties: two results with identical distances in a different order.
  - In the other 2, the *before* database's approximate index (HNSW) missed the
    query's own exact match (distance 0.0), which the *after* index found.
- The differences come from HNSW being approximate, and its graph differing from one
  build to the next, not from the embeddings. In float32 the embeddings are identical
  ([#60](https://github.com/precisiondata/aatm/pull/60)).

**Repair works.** After 5,000 random rows were deleted from the *after* database, a
repair embedded exactly those 5,000 again, in 90 s with no warnings; the row count went
back to 2,458,830. The repair calibrated again and chose a token budget of 8,192. On
this GPU, 4,096 and 8,192 are within 5% of each other in speed, so calibration can pick
either.

`results/` holds both runs in full, including every query's top-10 ids and distances.
