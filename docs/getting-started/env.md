# Environment variables

Some components depend on external APIs. Create a `.env` file in your project root when needed.

Example:

```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
```

AATM loads environment variables automatically with `python-dotenv`.

`OPENAI_API_KEY` is required by the [information extraction](quick_start_extraction.md) pipeline and by OpenAI-based components.

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