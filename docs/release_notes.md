# Release notes

## Unreleased

### New

- **PostgreSQL with pgvector** as a vector database (`AATM_VECTOR_DB=pgvector`,
  [#111](https://github.com/precisiondata/aatm/issues/111)), with the
  `aatm[pgvector]` extra. Each embedding model gets a table, with an HNSW index on the
  embeddings and a GIN index on the metadata, built after the load. Domain and
  vocabulary filters are applied during the search, with iterative index scans, so
  `PgVectorRetriever` needs no over-fetch. `aatm init --vector-db pgvector` builds the
  index, `aatm vector-db upload --to pgvector` copies a local one without re-embedding,
  and `aatm vector-db check --vector-db pgvector` reports the server, the pgvector
  version and each model's table. See
  [PostgreSQL with pgvector](getting-started/env.md#postgresql-with-pgvector).
- **`aatm init` asks where to build the search index**, when `--vector-db` is not
  given, preselecting `AATM_VECTOR_DB`. Each backend lists the models already built in
  it, and one that misses a setting or package is shown greyed out with what it needs.
  The embedding model question then marks the models already built there.
- **`aatm init` can start PostgreSQL for you.** When PostgreSQL is chosen and no server
  answers, it offers to start PostgreSQL with pgvector in Docker (container
  `aatm-pgvector`, reachable from this machine only) and to save `AATM_PG_DSN`, and
  optionally `AATM_VECTOR_DB=pgvector`, to `.env`. Or you set `AATM_PG_DSN` yourself.
- **Faster PostgreSQL index builds, with progress.** Building the search indexes shows a
  progress bar from PostgreSQL's own report, and warns when `maintenance_work_mem` is
  too small for the HNSW graph. `AATM_PG_BUILD_WORKERS` sets the parallel workers. The
  container `aatm init` starts gets shared memory, build memory and workers sized from
  Docker's resources instead of PostgreSQL's defaults (64 MB, 2 workers).
- **`aatm init` guides you through Docker and the container's size.** When Docker is not
  installed, not running or not usable, it says how to fix that on your platform, can
  open Docker Desktop for you, and checks again. It then shows the memory and CPUs the
  container can use (Docker's limits and the machine's free memory), suggests index
  build memory and workers, and lets you change them. An existing container with too
  little shared memory for them can be recreated, keeping its data.
- **`aatm vector-db sync`** records the indexes `.aatm/config.yaml` does not list, such
  as those built before 0.4.0, which `aatm vector-db check` now marks *not recorded*.
  `--default <model>` chooses which one retrievers use when none is given, and
  `--prune` drops records of indexes that are gone.
- **Denser HNSW indexes for PostgreSQL.** New pgvector indexes are built with `m=24` and
  `ef_construction=128` instead of pgvector's defaults (16 and 64). On the full
  `embeddinggemma-300M` index, filtered searches find 0.925 (Measurement) and 0.967
  (Condition) of the true 20 nearest records instead of 0.877 and 0.955, still in 8 to
  16 ms (ChromaDB's default: 0.953 and 0.968, in about 50 ms), for a build about a third
  longer and an index of the same size. The container `aatm init` starts may now give
  index builds up to 20 GB, which the 3,072-dimension models need. Existing indexes keep
  their settings until rebuilt. See `benchmarks/vector_db_backends`.
- **`AATM_PG_HNSW_M` and `AATM_PG_HNSW_EF_CONSTRUCTION`** set the density of the HNSW
  graph `aatm init` and `aatm vector-db upload` build (default 24 and 128), checked
  against pgvector's limits before the build starts.
- **An indexed vocabulary database.** `aatm init` indexes `.aatm/omop.db` (concept ids
  and codes, relationships, synonyms and ancestors), and adds the indexes a database
  you keep lacks; `aatm index-vocab` adds them on their own. On a full download with
  ICD-10-CM, a "Maps to" or hierarchy lookup took 0.7 to 10 s and now takes under a
  millisecond, which made the concept search agent several times faster. Lookups warn
  once about a database without them.
- **Coding to source vocabularies such as ICD-10-CM.**
  `ConceptSearchAgent.search(..., vocabularies=["ICD10CM"], standard=False)` finds a
  code of vocabularies whose concepts are not standard, for billing extracts or
  `*_source_concept_id` columns. Semantic search returns the ICD-10-CM names in the
  index and the codes that map to the standard concepts it finds, hierarchy follows
  the vocabulary's own parents and children, and the agent is asked for the most
  specific code the context supports: a code with children, such as an ICD-10-CM
  category, is sent back once with them listed. Standard searches are unchanged.
  On ACI-bench's 120 test encounters with ICD-10-CM annotations, extraction and
  coding with `gpt-5-mini` reach an exact-code F1 of 77.6 ± 1.1 over 5 runs; see
  `benchmarks/icd10cm_coding`.
- **`aatm init --rate-limit`, `--no-rate-limit` and `--chunk-size`** tune how fast the
  vector database is built. With an API model such as `gemini-embedding-001`, the build
  was held to the model's default of 3000 texts per minute in requests of 100 texts,
  about 18 hours for the default vocabularies. `--rate-limit` raises that limit to
  match your quota, `--no-rate-limit` removes it, and `--chunk-size` sets the texts
  sent per embeddings request.
- **Concurrent embeddings requests for API models**, with `aatm init --concurrency N`.
  Gemini and OpenAI embedding functions split each call into requests within the
  provider's limit on inputs per request (100 texts for Gemini, 2048 for OpenAI), so
  `--chunk-size` above 100 no longer fails with Gemini, and send up to `N` of them at
  a time. Requests go through the limiter of `aatm.llm.limits`: rate-limit (429),
  server and connection errors are retried with backoff instead of stopping the build,
  and limits set with `set_rate_limits("gemini", ...)` apply. The default chunk of an
  API model is 100 texts per concurrent request. With `gemini-embedding-001`,
  `--concurrency 4` embedded about 2.5 times as many texts per second as one request
  at a time.

### Fixes

- **`aatm init` no longer runs out of memory building the vocabulary database.** Each
  vocabulary file was read whole into memory before being written: on a full Athena
  download, `CONCEPT_RELATIONSHIP.csv` (2 GB) took about 17 GB and the process was
  killed. Files are now imported 500,000 rows at a time; the same download builds with
  a peak of 1.3 GB.

## 0.4.0

0.4.0 takes the notes-to-OMOP pipeline from a few notes in memory to full datasets:
resumable runs, batch pricing, sharding, a dry run with a cost estimate, a pilot
workspace and a command line for all of it. It also adds an agent that searches the
vocabularies, decision models, and two hosted vector databases.

### Upgrading from 0.3

Most code runs unchanged. Check these:

- **`mapper=` is now `coder=`** on `OmopTableBuilder` and `OmopExtractionPipeline`.
  `mapper=` still works with a `DeprecationWarning`, and will be removed in a later
  minor version.
- **The retriever defaults to the index `aatm init` built.** `TerminologyMapper()`,
  `load_retriever()`, `AgenticCoder(model)` and `aatm map` without a retriever use the
  embedding model recorded in `.aatm/config.yaml`. Naming a model whose index is not
  built raises `VectorDatabaseNotBuiltError`, which lists the indexes that are. Indexes
  built with 0.3 are not recorded: if you have several, pass `retriever=` or run
  `aatm init --embedding-model <model>` and choose *Skip* to record one.
- **Extraction concurrency.** `SimpleLLMExtractor` used to make one call at a time
  (`max_concurrency=1`). It now sets no cap of its own and relies on
  `aatm.llm.set_rate_limits`, with retries on rate-limit and transient errors for every
  call. A chunked run keeps 16 calls in flight with a hosted API and 4 with a local
  server unless you set `max_concurrency`.
- **Coded concepts are checked against the vocabulary.** A `map` result that is not a
  standard concept is stored as `0` and reported `unmapped`, and one of the wrong
  domain is reported `domain_mismatch`, whatever the coder says.
- **`pyarrow`** is a declared dependency (it was already installed through `datasets`).

### Notes to OMOP tables at scale

- **Chunked runs** with `pipeline.run(source, output_dir)`: notes are read in chunks
  from CSV, Parquet, DataFrames, lists or SQL (`notes_from_sql`), and tables are written
  as Parquet. A ledger records every extraction, so a run resumes where it stopped,
  retries failed notes, and never pays for a model call twice. Extraction streams across
  chunks, without waiting for each chunk to finish.
- **Rebuilds** with `pipeline.rebuild(output_dir)`: rebuild the tables from the stored
  extractions after editing an `omop` block or changing coder, with no model call.
- **Observability**: a progress line with throughput, time left, tokens and cost, and a
  summary per run broken down by stage, saved as JSON in `runs/`.
- **Batch extraction** with `batch=True` through the OpenAI, Anthropic and Gemini batch
  APIs, at about half the price. Pending jobs survive a stopped run.
- **Sharding** with `shard=(index, count)` and `run_sharded`, across processes,
  machines or scheduler jobs. Row ids are derived from the note id, so shards never
  clash.
- **`pipeline.plan(...)`**: a dry run that shows the settings a run would use, the notes
  left, an estimate of calls, tokens, cost and time from a pilot, and warnings with a
  fix.
- **`Pilot`**: versioned runs on a fixed sample, with `pilot.compare(...)` showing the
  rows, issues, mentions, codes and cost that changed between versions.
- **Command line**: `aatm omop run`, `rebuild`, `status`, `failures`, and
  `aatm omop pilot log` / `compare`, with options in a YAML file and exit codes for
  schedulers.

### Coding extracted text

- **Concept search agent**: `AgenticCoder` (OMOP tables) and `AgenticSelector`
  (terminology mapping) search the vocabularies with semantic and lexical search, code
  lookups, concept details and the hierarchy, and read the note's context. On a lab
  benchmark it coded 29 of 30 tests correctly, against 14 for an LLM selector.
- **Coding cache**: `cache=True` stores each coding result on disk and reuses it for
  identical inputs across chunks, runs and rebuilds. `aatm cache info` and
  `aatm cache clear` manage it.
- **Coder protocols**: `TextCoder`, `ContextualTextCoder` and `CacheableCoder` define
  what a coder of your own needs.

### Decision models

- `DecisionModel` answers closed-set questions with probabilities, with OpenAI, Gemini
  and Jev backends.
- `DecisionSelector` picks among retrieved candidates with one.
- `DecisionAttributeLabeller` labels or verifies entity attributes with one
  (`attribute_labeller=`, or `--labeller` on the command line).

### Setup and infrastructure

- **`aatm init --vocab-zip`** sets up the vocabularies from the link Athena emails, or
  the downloaded zip.
- **Chroma Cloud and Pinecone** as alternatives to a local ChromaDB index
  (`AATM_VECTOR_DB=cloud` or `pinecone`), with `aatm vector-db check` and
  `aatm vector-db upload`.
- **Faster retrieval and builds**: filtered queries over-fetch without the filter
  instead of using ChromaDB's slow filtered search; large queries are split into
  batches; local embeddings are batched by token budget and use free GPU memory.
- **Rate limits and retries** for every LLM call, set once with `set_rate_limits`, and
  `on_error="record"` to keep a batch going when one note fails.
- **JSON Schema for entity definitions**: completion and inline errors in your editor.

### Fixes

- `TerminologyMapper.map` and `amap` wrote their results to the mapper's own folder
  and ignored the `output_dir` argument. They now write to the folder given.
- `pipeline.plan()` estimated the notes left from counts alone, so new notes next to an
  earlier run could show as "0 to extract". It now matches note ids with the run's
  ledger when they are cheap to read (lists, DataFrames, Parquet), and
  `RunPlan.by_note_id` says which method was used. A sharded plan counts the notes of
  its shard exactly.
- `Pilot.compare` with the same version twice crashed; it now says so.
- `aatm amap`, which was never implemented, is hidden from the command line help.

### Documentation

- Versioned documentation (latest, previous and dev).
- A [command line reference](reference/cli.md) generated from the CLI, with every
  command, argument and option; clearer `--help` texts for `init`, `map`, `search-ui`
  and `serve`.
- New pages: [Start here](getting-started/index.md),
  [Set up the vocabularies](getting-started/setup.md), a
  [notes-to-OMOP quick start](getting-started/quick_start_omop.md), a
  [high-level guide](getting-started/high_level.md) and a
  [low-level guide](getting-started/low_level.md). The OMOP guide is split into
  [the tables](getting-started/omop_tables.md),
  [coding extracted text](getting-started/coding_text.md) and
  [processing large datasets](getting-started/large_datasets.md).
