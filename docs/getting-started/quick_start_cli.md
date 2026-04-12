# CLI Quick Start

This page shows how to use **AATM entirely from the command line**.

It is focused on the actual CLI workflow exposed by the package: `init`, `map`, and `search-ui`.

---

## What you can do from the CLI

With the CLI, you can:

- initialize the local AATM environment
- build the local SQLite vocabulary database
- build the mapping datasets
- build the local vector database
- run terminology mapping jobs
- launch the interactive search UI

The main commands are:

- `aatm init`
- `aatm map`
- `aatm search-ui`

These commands are defined directly in the CLI entrypoint. fileciteturn4file0

---

## 1. Install AATM

Install the package in your environment:

```bash
pip install aatm
```

For local development:

```bash
pip install -e .
```

Or with `uv`:

```bash
uv sync
```

---

## 2. Set your environment variables

Some CLI workflows use external APIs. Create a `.env` file in your project root when needed.

Example:

```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
```

AATM loads environment variables automatically when the CLI starts. fileciteturn4file0

---

## 3. Prepare your OMOP vocabularies directory

Before running `aatm init`, download the OMOP vocabularies you want to use and place them in a directory.

By default, the CLI expects:

```text
./vocabularies
```

If you do not use that location, you can point the CLI to a different directory with `--vocab-dir`. The CLI validates this path during initialization. fileciteturn4file0

---

## 4. Initialize everything from the CLI

The `init` command is the main CLI setup workflow.

It does all of the following for you:

- creates the local `.aatm` helper directory
- ensures `.aatm` is added to `.gitignore`
- builds the local OMOP SQLite database
- lets you choose an embedding model
- lets you choose the standard vocabularies
- builds the mapping datasets
- builds the local vector database

That means you do **not** need to call Python setup functions manually for the normal setup flow. fileciteturn4file0

### Simplest setup

```bash
aatm init
```

This uses the default vocab directory and interactively asks you to choose the embedding model and standard vocabularies. fileciteturn4file0

### Setup with a custom vocab directory

```bash
aatm init --vocab-dir ./my_vocabularies
```

### Setup with an explicit embedding model

```bash
aatm init --embedding-model embeddinggemma-300M
```

### Setup with explicit standard vocabularies

```bash
aatm init --standard-vocabs LOINC --standard-vocabs SNOMED --standard-vocabs RxNorm
```

### Fully explicit setup

```bash
aatm init \
  --vocab-dir ./vocabularies \
  --embedding-model embeddinggemma-300M \
  --standard-vocabs LOINC \
  --standard-vocabs SNOMED \
  --standard-vocabs RxNorm
```

### Supported embedding models

The CLI supports these embedding models during initialization:

- `qwen3-06B`
- `qwen3-4B`
- `gemini-embedding-001`
- `embeddinggemma-300M`
- `text-embedding-3-small`
- `text-embedding-3-large`

These are the supported embedding model names hardcoded in the CLI. fileciteturn4file0

### Supported standard vocabularies

The CLI setup flow supports these standard vocabularies:

- `LOINC`
- `SNOMED`
- `RxNorm`

These are also defined in the CLI entrypoint. fileciteturn4file0

---

## 5. Prepare your input CSV

After initialization, prepare the CSV you want to map.

The mapper expects an OMOP-style `SOURCE_TO_CONCEPT_MAP` input structure, including these columns:

- `source_code`
- `source_concept_id`
- `source_vocabulary_id`
- `source_code_description`
- `valid_start_date`
- `valid_end_date`
- `invalid_reason`

Example:

```csv
source_code,source_concept_id,source_vocabulary_id,source_code_description,valid_start_date,valid_end_date,invalid_reason
A01,,LOCAL,"Dor no peito",2020-01-01,2099-12-31,
B02,,LOCAL,"Diabetes mellitus tipo 2",2020-01-01,2099-12-31,
```

---

## 6. Run mapping directly from the CLI

The `map` command runs a terminology mapping task. You can use it in two ways:

- with a task config file
- with explicit CLI options

Both paths are supported directly by the CLI implementation. fileciteturn4file0

### Option A: run from explicit CLI arguments

This is the most direct fully-CLI workflow.

```bash
aatm map \
  --input-file data/source_to_concept_map.csv \
  --output-dir output \
  --translator-id empty-translator \
  --retriever-id embeddinggemma-300M \
  --reranker-id bm25-reranker \
  --selector-id first-result-selector \
  --batch-size 100
```

### Recommended first mapping run

A good first run is:

- `empty-translator`
- `embeddinggemma-300M`
- `bm25-reranker`
- `first-result-selector`

Example:

```bash
aatm map \
  --input-file data/source_to_concept_map.csv \
  --output-dir output \
  --translator-id empty-translator \
  --retriever-id embeddinggemma-300M \
  --reranker-id bm25-reranker \
  --selector-id first-result-selector
```

### Run a small test job

Use `--limit-to` when you want to test with only a few rows.

```bash
aatm map \
  --input-file data/source_to_concept_map.csv \
  --output-dir output \
  --translator-id empty-translator \
  --retriever-id embeddinggemma-300M \
  --reranker-id bm25-reranker \
  --selector-id first-result-selector \
  --limit-to 20
```

### Apply rate limiting

If needed, you can also pass a rate limit:

```bash
aatm map \
  --input-file data/source_to_concept_map.csv \
  --output-dir output \
  --translator-id gemini-2.5-flash \
  --retriever-id embeddinggemma-300M \
  --reranker-id bm25-reranker \
  --selector-id first-result-selector \
  --batch-size 50 \
  --rate-limit 100
```

The CLI accepts all of these options directly. fileciteturn4file0

---

## 7. Run mapping from a config file

The other CLI workflow is to store the mapping task in a config file and pass it with `--task-config-path`.

### Example command

```bash
aatm map --task-config-path task.yaml
```

When you do this, the CLI loads the task config file and runs the mapping task from it. fileciteturn4file0

### Example task config

```yaml
input_file: data/source_to_concept_map.csv
output_dir: output
translator_id: empty-translator
retriever_id: embeddinggemma-300M
reranker_id: bm25-reranker
selector_id: first-result-selector
batch_size: 100
rate_limit: null
limit_to: null
```

This is useful when you want reproducible runs or reusable task definitions.

---

## 8. Launch the search UI from the CLI

You can also launch the Streamlit-based search interface directly from the CLI:

```bash
aatm search-ui
```

The CLI resolves the packaged `search_ui.py` file and launches it through Streamlit using the current Python interpreter. fileciteturn4file0

---

## 9. What gets created locally

After a normal CLI setup and mapping workflow, you will typically have local artifacts such as:

```text
.aatm/
├── omop.db
├── datasets/
└── chroma_vector_dbs/
```

And your mapped output will typically be written to:

```text
output/mapped_source_concepts.csv
```

The CLI `init` command is responsible for creating the local helper resources, and the `map` command runs the terminology mapping workflow. fileciteturn4file0

---

## 10. Useful component IDs for CLI runs

### Translators

- `empty-translator`
- `gemini-2.5-flash`

### Retrievers

- `qwen3-06B`
- `qwen3-4B`
- `gemini-embedding-001`
- `embeddinggemma-300M`
- `text-embedding-3-small`
- `text-embedding-3-large`

### Rerankers

- `bm25-reranker`
- `qwen3-reranker-0.6b`
- `qwen3-reranker-4b`
- `qwen3-reranker-8b`

### Selectors

- `first-result-selector`
- `gpt-5.2`
- `gpt-5`
- `gpt-5-mini`
- `gpt-5-nano`
- `gemini-3-pro-preview`
- `gemini-3-flash-preview`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `gemini-2.5-pro`

These identifiers match the registries used by the mapper workflow you invoke from the CLI.

---

## End-to-end CLI example

Here is a full CLI-only path.

### Step 1: initialize

```bash
aatm init \
  --vocab-dir ./vocabularies \
  --embedding-model embeddinggemma-300M \
  --standard-vocabs LOINC \
  --standard-vocabs SNOMED \
  --standard-vocabs RxNorm
```

### Step 2: run mapping

```bash
aatm map \
  --input-file data/source_to_concept_map.csv \
  --output-dir output \
  --translator-id empty-translator \
  --retriever-id embeddinggemma-300M \
  --reranker-id bm25-reranker \
  --selector-id first-result-selector \
  --batch-size 100
```

### Step 3: inspect output

```bash
ls output
```

You should see the mapped CSV file there.

### Step 4: optionally launch the search UI

```bash
aatm search-ui
```

---

## Troubleshooting

### `aatm init` says the vocabulary directory does not exist

Make sure your OMOP vocabulary files are present in `./vocabularies`, or pass the correct path with:

```bash
aatm init --vocab-dir /path/to/vocabularies
```

The CLI explicitly checks whether the provided directory exists. fileciteturn4file0

### `aatm init` says the embedding model is unsupported

Use one of the supported model names listed in this page. The CLI validates the model name before continuing. fileciteturn4file0

### `aatm map` says the task config file does not exist

Check the path you passed to:

```bash
aatm map --task-config-path task.yaml
```

The CLI verifies that the config file exists before loading it. fileciteturn4file0

### Retrieval fails at mapping time

Make sure you ran `aatm init` first and successfully built the local vector database for the retriever you want to use.

### API-backed components fail

Check that your `.env` file contains the required API keys.

---

## Next steps

After this CLI quick start, the most useful next pages are usually:

- Python API quick start
- task configuration reference
- component registry reference
- search UI
- advanced customization
