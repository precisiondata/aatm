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
