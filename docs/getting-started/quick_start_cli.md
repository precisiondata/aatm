# Quick start: medical coding from the command line

This page maps a CSV of local codes and descriptions to standard OMOP concepts with
`aatm map`, without writing Python.

**Before you start:** run `aatm init` ([Set up the vocabularies](setup.md)). `aatm map`
searches the index it builds.

!!! tip "Every command"
    | Command | Does |
    |---|---|
    | `aatm init` | Sets up the vocabularies and the search index ([Set up the vocabularies](setup.md)) |
    | `aatm doctor` | Checks whether local models use your GPU ([Installation](installation.md#gpu-acceleration-optional)) |
    | `aatm map` | Maps a CSV of terms to standard concepts (this page) |
    | `aatm omop run`, `rebuild`, `status`, `failures`, `pilot` | Clinical notes to OMOP tables ([Quick start](quick_start_omop.md), [High-level guide](high_level.md)) |
    | `aatm cache info`, `clear` | The coding cache of OMOP runs |
    | `aatm vector-db check`, `upload` | Where the search index is, and copying it to Chroma Cloud or Pinecone ([Environment variables](env.md#vector-database-local-chroma-cloud-or-pinecone)) |
    | `aatm search-ui` | An interactive search over the vocabularies (below) |
    | `aatm serve` | A REST API for mapping |

    `aatm <command> --help` lists the options of each.

---

## 1. Prepare your input CSV

The input follows the OMOP `SOURCE_TO_CONCEPT_MAP` layout, with these columns:

- `source_code`
- `source_concept_id`
- `source_vocabulary_id`
- `source_code_description`: the text that is mapped
- `valid_start_date`
- `valid_end_date`
- `invalid_reason`

```csv title="data/source_to_concept_map.csv"
source_code,source_concept_id,source_vocabulary_id,source_code_description,valid_start_date,valid_end_date,invalid_reason
A01,,LOCAL,"Dor no peito",2020-01-01,2099-12-31,
B02,,LOCAL,"Diabetes mellitus tipo 2",2020-01-01,2099-12-31,
```

## 2. Map it

```bash
aatm map --input-file data/source_to_concept_map.csv --output-dir output --limit-to 20
```

This searches the index `aatm init` built and keeps the closest candidate, on the first
20 rows. The result is `output/mapped_source_concepts.csv`, with the chosen concept's
id, vocabulary, code and domain, and a confidence score for each row.

The closest candidate is often wrong for short, abbreviated or non-English terms. Add a
translator and let an LLM choose among the candidates:

```bash
aatm map \
  --input-file data/source_to_concept_map.csv \
  --output-dir output \
  --translator-id gemini-2.5-flash \
  --reranker-id bm25-reranker \
  --selector-id gpt-5-mini \
  --batch-size 50
```

| Option | Values | Default |
|---|---|---|
| `--translator-id` | `empty-translator`, `gemini-2.5-flash` | no translation |
| `--retriever-id` | An embedding model whose index you built, e.g. `text-embedding-3-small` | the one `aatm init` built |
| `--reranker-id` | `bm25-reranker`, `qwen3-reranker-0.6b`, `qwen3-reranker-4b`, `qwen3-reranker-8b` | no reranking |
| `--selector-id` | `first-result-selector`; an LLM such as `gpt-5-mini` or `gemini-2.5-flash`; the agent, `agent-gpt-5-mini`; a decision model, `decision-openrouter/typesafe/jev-1.13` | `first-result-selector` |
| `--batch-size` | Terms per batch | 100 |
| `--rate-limit` | Terms per minute | no limit |
| `--limit-to` | Map only the first N rows | every row |

The full list of selectors is in the
[Python quick start](quick_start_python_api.md#2-choose-the-components).

## 3. Save the configuration

For runs you repeat, put the options in a file and pass it with `--task-config-path`
(`-t`):

=== "YAML"

    ```yaml title="task.yaml"
    input_file: data/source_to_concept_map.csv
    output_dir: output
    translator_id: gemini-2.5-flash
    reranker_id: bm25-reranker
    selector_id: gpt-5-mini
    batch_size: 50
    ```

=== "JSON"

    ```json title="task.json"
    {
      "input_file": "data/source_to_concept_map.csv",
      "output_dir": "output",
      "translator_id": "gemini-2.5-flash",
      "reranker_id": "bm25-reranker",
      "selector_id": "gpt-5-mini",
      "batch_size": 50
    }
    ```

```bash
aatm map --task-config-path task.yaml
```

---

## 4. Explore the vocabularies in the search UI

```bash
aatm search-ui
```

This opens a Streamlit app where you type a term and see the candidates the search
index returns, which helps choose a retriever and a `top_k`. Streamlit's file watcher
is turned off; set `STREAMLIT_SERVER_FILE_WATCHER_TYPE=auto` to turn it back on while
editing the UI.

![Streamlit search UI screenshot.](../assets/search-ui-screenshot.png)
/// caption
Streamlit search UI screenshot.
///

---

## Troubleshooting

| You see | Fix |
|---|---|
| `No vector database for '<model>'` | That index is not built: use one `aatm vector-db check` lists, leave `--retriever-id` out, or build it with `aatm init --embedding-model <model>` |
| `aatm map` says the task config file does not exist | Check the path given to `--task-config-path` |
| The CSV fails validation | Check the required columns above |
| An LLM component fails with an authentication error | Check the provider's key in `.env` ([Environment variables](env.md)) |

**More or fewer log messages.** Every command accepts a global `--log-level` (`-l`),
placed before the command name, or the `AATM_LOG_LEVEL` environment variable. Logs go
to standard error:

```bash
aatm --log-level DEBUG map --task-config-path task.yaml
```

**Output redirected to a file looks garbled.** The CLI writes UTF-8 to files and pipes
(e.g. `aatm serve > server.log`), so open such files as UTF-8.
