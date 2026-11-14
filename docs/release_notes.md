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
