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

## Vector database location (Chroma Cloud)

By default, AATM builds one ChromaDB vector database per embedding model on this machine, under `.aatm/chroma_vector_dbs/`. To keep the index in [Chroma Cloud](https://trychroma.com/) instead, which lets a team build it once and share it, set:

| Variable | Default | Purpose |
|---|---|---|
| `AATM_VECTOR_DB` | `local` | `local` or `cloud`. `aatm init --vector-db` overrides it for the build. |
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
