# Environment variables

Some components depend on external APIs. Create a `.env` file in your project root when needed.

Example:

```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

AATM loads environment variables automatically with `python-dotenv`.

The [information extraction](quick_start_extraction.md) pipeline needs the key of the provider you use: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY`. Local OpenAI-compatible servers such as Ollama usually need none. `OPENAI_API_KEY` and `GOOGLE_API_KEY` are also used by the OpenAI- and Gemini-based mapping components.

[Decision models](../reference/decision_models.md) use `OPENAI_API_KEY` (`openai/gpt-6-luna`), `GOOGLE_API_KEY` (`gemini/gemini-decision-1`) or `OPENROUTER_API_KEY` (`openrouter/typesafe/jev-1.13`). They send the state they decide on, e.g. a sentence of a note, to that provider. Check its data processing terms before sending protected health information.

## Optional settings

| Variable | Default | Purpose |
|---|---|---|
| `AATM_LOG_LEVEL` | `INFO` | Log level used by the CLI, the API and the search UI: `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. Same as the CLI option `--log-level`. |
| `STREAMLIT_SERVER_FILE_WATCHER_TYPE` | unset (watcher off) | `aatm search-ui` disables Streamlit's file watcher unless this is set, e.g. to `auto` while editing the search UI. |

## Logging in Python code

Importing `aatm` does not print anything. To see AATM's log messages in a notebook or script:

```python
import logging
import aatm

aatm.configure_logging()               # INFO and above, on stderr
aatm.configure_logging(logging.DEBUG)  # everything
logging.getLogger("aatm.data_models.extractors").setLevel(logging.DEBUG)  # one module only
```

## Vector database

By default, AATM builds one ChromaDB vector database per embedding model on this machine, under `.aatm/chroma_vector_dbs/`. Three other backends let a team build the index once and share it, let machines without the disk space or GPU for a local index run retrieval, or filter faster:

- [Chroma Cloud](https://trychroma.com/), with `AATM_VECTOR_DB=cloud`.
- [Pinecone](https://www.pinecone.io/), with `AATM_VECTOR_DB=pinecone` (see [below](#pinecone)).
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector), on your machine or a server you run, with `AATM_VECTOR_DB=pgvector` (see [below](#postgresql-with-pgvector)).

Switching is configuration only: retrievers, `aatm map`, `aatm search-ui`, `aatm serve`, the concept search agent and `load_retriever()` all use the configured backend.

### Chroma Cloud

Set:

| Variable | Default | Purpose |
|---|---|---|
| `AATM_VECTOR_DB` | `local` | `local`, `cloud`, `pinecone` or `pgvector`. `aatm init --vector-db` overrides it for the build. |
| `CHROMA_API_KEY` | unset | Chroma Cloud API key. Required in cloud mode. Never logged. |
| `CHROMA_TENANT` | unset | Chroma Cloud tenant. Optional when the API key is scoped to one database. |
| `CHROMA_DATABASE` | unset | Chroma Cloud database. Optional when the API key is scoped to one database. |
| `CHROMA_CLOUD_HOST` | `api.trychroma.com` | Chroma Cloud host. |

```env
AATM_VECTOR_DB=cloud
CHROMA_API_KEY=your_chroma_api_key
CHROMA_DATABASE=aatm
```

In cloud mode, one database holds every model's index, so each embedding model gets its own collection, named `expressions-<model>` (e.g. `expressions-qwen3-06B`). Retrievers, `aatm map`, the search UI and the API then query Chroma Cloud with the same filters as locally. Embeddings for local models are still computed on the machine that runs the build.

Useful commands:

```bash
aatm vector-db check                       # show the configured location and test the connection
aatm init --vector-db cloud                # build the index directly in Chroma Cloud
aatm vector-db upload -e embeddinggemma-300M  # copy an existing local index, without re-embedding
```

`aatm vector-db upload` skips rows already in the cloud, so an interrupted upload resumes when run again. Cloud writes are retried with backoff on rate limits, server errors and network failures.

### Pinecone

Install the extra, then set the variables below:

```bash
pip install "aatm[pinecone]"   # or: uv sync --extra pinecone
```

| Variable | Default | Purpose |
|---|---|---|
| `AATM_VECTOR_DB` | `local` | Set to `pinecone`. |
| `PINECONE_API_KEY` | unset | Pinecone API key. Required. Never logged. |
| `PINECONE_INDEX_PREFIX` | `aatm-expressions` | Prefix of the index names. |
| `PINECONE_CLOUD` | `aws` | Cloud of new serverless indexes. |
| `PINECONE_REGION` | `us-east-1` | Region of new serverless indexes. |

```env
AATM_VECTOR_DB=pinecone
PINECONE_API_KEY=your_pinecone_api_key
```

Every Pinecone index has a fixed dimension, so each embedding model gets its own serverless index, named `<prefix>-<model>` and made valid for Pinecone: lowercase `a-z`, `0-9` and `-`, at most 45 characters (e.g. `aatm-expressions-qwen3-06b`). `aatm init` creates it with the dimension of the model's embeddings.

```bash
aatm vector-db check --vector-db pinecone                     # list the models' indexes
aatm init --vector-db pinecone                                # build the index in Pinecone
aatm vector-db upload --to pinecone -e embeddinggemma-300M    # copy a local index, without re-embedding
```

On Pinecone's free plan the full index does not fit: writes cost one write unit per KB of request, and a 768-dimension record is about 16.6 KB, so the 2 million monthly write units cover roughly 120,000 records. `--ids-file` uploads only the rows a file lists, one ID per line; `benchmarks/vector_db_backends/make_subset.py` makes such a subset. Set `PINECONE_INDEX_PREFIX` (e.g. `aatm-subset`) so the subset does not take the full index's name.

How the index maps to Pinecone:

- **Scores.** The indexes use the `euclidean` metric, whose score is the squared Euclidean distance, the same as ChromaDB's default `l2` distance. A record has the same `distance` on both backends, so the confidence score (`1 - distance`), selectors and thresholds work unchanged.
- **Filters.** Domain and vocabulary filters (`retriever_filter`) are translated to Pinecone's syntax and applied **during** the vector search. So by default the Pinecone retriever passes them straight to Pinecone (`filter_strategy="native"`) instead of over-fetching. The over-fetch strategies still work, with query sizes capped at Pinecone's maximum `top_k` of 10,000. See [Metadata filters](concepts.md#metadata-filters).
- **Records.** Pinecone has no document field and no null metadata. The expression text is stored in the metadata (as `expression`), and `None` values are left out. The embedding precision of local models is stored as an index tag (`aatm_embedding_dtype`), and queries are encoded in it.
- **Writes** use `upsert`, which is idempotent, in requests within Pinecone's limits (1,000 records and 2 MB per request). Every request is retried with backoff on rate limits, server errors and network failures. A repair (`aatm init`, *Repair*) or a rerun of `aatm vector-db upload` only writes the missing records. Newly written records take a moment to become visible, so a repair right after a build may embed a few records again, which is harmless.

### PostgreSQL with pgvector

[pgvector](https://github.com/pgvector/pgvector) adds vector search to PostgreSQL. Use it to keep the index next to OMOP data you already host in Postgres, to run fast filtered searches on your own machine, or to share one index across a team on a server you run.

Install the extra:

```bash
pip install "aatm[pgvector]"   # or: uv sync --extra pgvector
```

The quickest local server is the `pgvector/pgvector` Docker image. This one accepts connections from this machine only, without a password:

```bash
docker run -d --name aatm-pgvector --shm-size=1g \
  -e POSTGRES_HOST_AUTH_METHOD=trust -e POSTGRES_DB=aatm \
  -p 127.0.0.1:5432:5432 -v aatm-pgvector:/var/lib/postgresql/data \
  pgvector/pgvector:pg17
```

`--shm-size` gives the HNSW index build enough shared memory; Docker's 64 MB default makes large parallel builds fail. For a server other people can reach, set a password (`POSTGRES_PASSWORD`) instead of `trust`.

Then set:

| Variable | Default | Purpose |
|---|---|---|
| `AATM_VECTOR_DB` | `local` | Set to `pgvector`. |
| `AATM_PG_DSN` | unset | Connection string, e.g. `postgresql://postgres@localhost:5432/aatm`. If unset, libpq's defaults and the standard `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER` and `PGPASSWORD` variables apply. A password in it is never shown. |
| `AATM_PG_TABLE_PREFIX` | `aatm_expressions` | Prefix of the table names. |
| `AATM_PG_EF_SEARCH` | `100` | `hnsw.ef_search`: candidates the search keeps, from 1 to 1000. Higher finds more true neighbours, more slowly. Each query uses at least its `top_k`. |
| `AATM_PG_ITERATIVE_SCAN` | `relaxed_order` | `hnsw.iterative_scan`: `relaxed_order`, `strict_order` or `off`. How a filtered search keeps scanning until it has `top_k` matches (pgvector 0.8+). |
| `AATM_PG_MAINTENANCE_WORK_MEM` | server setting | Memory for building the HNSW index, e.g. `4GB`. More memory builds it faster. |

```env
AATM_VECTOR_DB=pgvector
AATM_PG_DSN=postgresql://postgres@localhost:5432/aatm
```

Prefer `PGPASSWORD` or a [`.pgpass` file](https://www.postgresql.org/docs/current/libpq-pgpass.html) to a password in `AATM_PG_DSN`.

Without `AATM_PG_DSN`, AATM connects where libpq would: `localhost:5432`, to a database named after your user. `aatm init` names the server it will use, and if the connection fails it says where it tried and how to set `AATM_PG_DSN` or start a local server. AATM waits up to 10 seconds per address for the server; set `connect_timeout` in `AATM_PG_DSN`, or `PGCONNECT_TIMEOUT`, to change it.

```bash
aatm vector-db check --vector-db pgvector                     # server, pgvector version, the models' tables
aatm init --vector-db pgvector                                # build the index in PostgreSQL
aatm vector-db upload --to pgvector -e embeddinggemma-300M    # copy a local index, without re-embedding
```

How the index maps to PostgreSQL:

- **Tables.** Each embedding model gets its own table, named `<prefix>_<model>` and made a safe identifier (e.g. `aatm_expressions_qwen3_06b`), with columns `id`, `embedding vector(<dimension>)`, `document` and `metadata jsonb`. The table `aatm_index_metadata` records each index's dimension and the embedding precision of local models, and queries are encoded in it. `aatm init` installs the `vector` extension if the database lacks it, which needs the right to create extensions.
- **Search indexes.** An HNSW index on the embeddings and a GIN index on the metadata are built once the rows are loaded, which is much faster than updating them row by row. On the full index (about 2.5 million expressions) this takes a while; raise `AATM_PG_MAINTENANCE_WORK_MEM` to speed it up. HNSW indexes `vector` columns of up to 2,000 dimensions, so larger embeddings (`qwen3-4B`, `gemini-embedding-001`, `text-embedding-3-large`) are indexed in half precision (`halfvec`, up to 4,000); their distances still come from the full-precision vectors.
- **Scores.** pgvector's Euclidean distance is squared, so a record has the same `distance` as in ChromaDB and Pinecone, and the confidence score (`1 - distance`), selectors and thresholds work unchanged.
- **Filters.** Domain and vocabulary filters (`retriever_filter`) become `WHERE` clauses on the metadata, applied **during** the vector search, so the retriever passes them straight to PostgreSQL by default (`filter_strategy="native"`). Iterative index scans keep a filtered search going until it has `top_k` matches; PostgreSQL may also choose the metadata index and an exact search when few rows match. See [Metadata filters](concepts.md#metadata-filters).
- **Writes** go through a staging table with `COPY`, then into the table. A first write skips IDs already stored, and a retry replaces them, so a repair (`aatm init`, *Repair*) or a rerun of `aatm vector-db upload` only writes the missing records and never duplicates one.

### Choosing a backend

| | Local ChromaDB | Chroma Cloud | Pinecone | PostgreSQL + pgvector |
|---|---|---|---|---|
| Setup | None | API key | API key and `aatm[pinecone]` | A PostgreSQL server with pgvector and `aatm[pgvector]` |
| Shared across a team | No | Yes | Yes | Yes, on a server others can reach |
| Disk and memory on this machine | Several GB per model | None | None | Several GB per model, on the server |
| Filtered queries (domain, vocabulary) | Over-fetch by default, because ChromaDB's own filtered query takes seconds on a full index | Same as local | Filtered during the search | Filtered during the search |
| Cost | Free | Usage-based | Usage-based | Free locally; your server otherwise |

Embeddings of local models are computed on the machine that runs the build, whichever backend stores them. To compare query time and recall on your own index, see `benchmarks/vector_db_backends`.

## Device selection

| Variable | Values | Effect |
|---|---|---|
| `AATM_DEVICE` | `cuda`, `mps`, `cpu` | Forces the device local models run on. If unset, AATM uses CUDA, then MPS, then CPU, whichever PyTorch can see first. AATM raises an error if the requested device is not available. |

At `INFO` level, AATM logs the device local models run on (see the logging settings above).

## Embedding precision

| Variable | Values | Effect |
|---|---|---|
| `AATM_EMBEDDING_DTYPE` | `float32`, `bfloat16`, `float16` | Precision local embedding models run in when building the vector database. If unset, AATM uses the precision the model's checkpoint is stored in (`bfloat16` for Qwen3 Embedding, `float32` for EmbeddingGemma), or `float32` if this machine cannot run it. Half precision (`bfloat16` or `float16`) needs an NVIDIA GPU; it roughly halves GPU memory use and encodes faster. `bfloat16` needs an Ampere or newer GPU, and EmbeddingGemma does not support `float16`. `aatm init --embedding-dtype` overrides this variable. |

Half precision changes the embeddings slightly, so AATM never mixes precisions in one vector database. It records the precision in the database and:

- refuses to repair a database with a different precision than the one requested. Choose **Overwrite** to rebuild it instead;
- when no precision is requested, repairs a database in the precision it was built with;
- refuses to repair a database built before AATM recorded precision, since its precision cannot be checked;
- encodes search queries in the database's precision, falling back to `float32` with a warning if this machine cannot run it.
