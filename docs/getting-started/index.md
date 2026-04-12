# Getting Started

Welcome to **AATM** — the **Any-to-Any Terminology Mapper**.

AATM helps you map source terminology expressions to standardized concepts through a modular pipeline composed of:

- **translation**
- **retrieval**
- **reranking**
- **selection**

It is designed so you can start simple and progressively swap components as needed.

---

## What AATM does

At a high level, AATM takes source concepts such as:

- local clinical terms
- source vocabulary descriptions
- free-text expressions

and maps them to standardized concepts using a pipeline like this:

```text
source concept
  -> translator
  -> retriever
  -> reranker
  -> selector
  -> mapped concept
```

The default workflow is centered on OMOP-style terminology mapping.

---

## Installation

Install the package in your environment.

```bash
pip install aatm
```

If you are developing locally:

```bash
pip install -e .
```

If you use `uv`:

```bash
uv sync
```

---

## Environment variables

Some components depend on external APIs. Create a `.env` file in your project root when needed.

Example:

```env
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
```

AATM loads environment variables automatically with `python-dotenv`.

---

## Project artifacts created locally

AATM uses a local `.aatm/` directory to store generated assets such as:

```text
.aatm/
├── omop.db
├── datasets/
└── chroma_vector_dbs/
```

Typical outputs are written to:

```text
output/
```

---

## Basic concepts

### Translator

Translators convert input text into English before retrieval.

Examples:

- `EmptyTranslator`
- `GeminiTranslator`
- `OpenAITranslator`

Use `EmptyTranslator` when your inputs are already in the desired language.

### Retriever

Retrievers fetch candidate concepts from a vector database.

Main implementation:

- `ChromaDBRetriever`

### Reranker

Rerankers reorder retrieved candidates.

Examples:

- `BM25Reranker`
- `Qwen3Reranker`

### Selector

Selectors choose the final mapped concept from the retrieved candidates.

Examples:

- `FirstResultSelector`
- `OpenAILLMSelector`
- `GeminiLLMSelector`

---

## Quick start

### 1. Load a translator, retriever, reranker, and selector

```python
from aatm.registries.translators import load_translator
from aatm.registries.retrievers import load_retriever
from aatm.registries.rerankers import load_reranker
from aatm.registries.selectors import load_selector

translator = load_translator("empty-translator")
retriever = load_retriever("embeddinggemma-300M")
reranker = load_reranker("bm25-reranker")
selector = load_selector("first-result-selector")
```

### 2. Build a mapper

```python
from aatm.terminology_mapper import TerminologyMapper

mapper = TerminologyMapper(
    translator=translator,
    retriever=retriever,
    reranker=reranker,
    selector=selector,
    batch_size=100,
)
```

### 3. Map a CSV file

```python
results_df = mapper.map("source_to_concept_map.csv")
print(results_df.head())
```

This runs the full pipeline and writes the output to:

```text
output/mapped_source_concepts.csv
```

---

## Expected input format

The CSV input is expected to follow the OMOP `SOURCE_TO_CONCEPT_MAP` structure.

Required columns:

- `source_code`
- `source_concept_id`
- `source_vocabulary_id`
- `source_code_description`
- `valid_start_date`
- `valid_end_date`
- `invalid_reason`

### Example

```csv
source_code,source_concept_id,source_vocabulary_id,source_code_description,valid_start_date,valid_end_date,invalid_reason
A01,,LOCAL,"Dor no peito",2020-01-01,2099-12-31,
B02,,LOCAL,"Diabetes mellitus tipo 2",2020-01-01,2099-12-31,
```

---

## Using custom column names

If your file does not use the expected OMOP column names, provide a column mapping.

```python
mapper = TerminologyMapper(
    column_mapping={
        "code": "source_code",
        "description": "source_code_description",
        "vocabulary": "source_vocabulary_id",
        "start_date": "valid_start_date",
        "end_date": "valid_end_date",
        "reason": "invalid_reason",
        "concept_id": "source_concept_id",
    }
)
```

Then call:

```python
results_df = mapper.map("my_input.csv")
```

---

## Building the local databases

Before retrieval works, you usually need to build local assets.

### 1. Build the local SQLite vocabulary database

```python
from pathlib import Path
from aatm.builders import build_local_sqlite_vocab_database

build_local_sqlite_vocab_database(Path("/path/to/omop_vocab_files"))
```

This creates:

```text
.aatm/omop.db
```

### 2. Build terminology-mapping datasets

```python
from aatm.builders import build_mapping_datasets

build_mapping_datasets(
    standard_vocabularies=["SNOMED", "LOINC", "RxNorm"]
)
```

This creates CSV datasets under:

```text
.aatm/datasets/
```

### 3. Build the local vector database

```python
from aatm.builders import build_local_vector_database

build_local_vector_database(
    embedding_model_name="embeddinggemma-300M",
    batch_size=100,
)
```

This creates a ChromaDB vector store for retrieval.

---

## Minimal end-to-end example

```python
from aatm.terminology_mapper import TerminologyMapper
from aatm.registries.translators import load_translator
from aatm.registries.retrievers import load_retriever
from aatm.registries.selectors import load_selector
from aatm.registries.rerankers import load_reranker

mapper = TerminologyMapper(
    translator=load_translator("empty-translator"),
    retriever=load_retriever("embeddinggemma-300M"),
    reranker=load_reranker("bm25-reranker"),
    selector=load_selector("first-result-selector"),
    batch_size=32,
)

df = mapper.map("source_to_concept_map.csv")
print(df.head())
```

---

## Using LLM-based components

You can replace the default rule-based stages with LLM-backed components.

### OpenAI selector

```python
selector = load_selector("gpt-5-mini")
```

### Gemini selector

```python
selector = load_selector("gemini-2.5-flash")
```

### Gemini translator

```python
translator = load_translator("gemini-2.5-flash")
```

### Qwen reranker

```python
reranker = load_reranker("qwen3-reranker-0.6b")
```

These components may require API keys, model downloads, or additional runtime resources.

---

## Supported registered components

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

---

## Pipeline composition

AATM components are pipeline-compatible and use `|` composition.

For example:

```python
selected_results = translated_batch | retriever | reranker | selector
```

This makes it easy to replace or combine stages.

---

## Streamlit search UI

AATM also includes a lightweight Streamlit-based search interface for exploring retrieval results interactively.

Typical features include:

- search by expression
- retriever selection from the registry
- ranked candidate inspection
- concept metadata display
- confidence score display

If your project exposes a Streamlit entry point, you can run it with something like:

```bash
streamlit run path/to/search_ui.py
```

---

## Logging

AATM includes centralized logging utilities so that modules share a consistent format.

Typical usage:

```python
from aatm.logs import configure_logging, get_logger
import logging

configure_logging(level=logging.INFO)
logger = get_logger(__name__)

logger.info("AATM initialized")
```

---

## Common workflow

A typical setup looks like this:

1. Prepare OMOP vocabulary files
2. Build the local SQLite database
3. Build derived mapping datasets
4. Build a vector database for your chosen embedding model
5. Configure translator, retriever, reranker, and selector
6. Run `TerminologyMapper.map(...)`
7. Review the generated CSV output

---

## Troubleshooting

### Missing API key

If you use Gemini or OpenAI components and get authentication errors, verify your `.env` file contains the correct key.

### Missing vector database

If retrieval fails, make sure you already built the local ChromaDB store for the selected retriever model.

### Invalid input columns

If mapping fails on CSV loading, verify that your file contains the required OMOP columns or provide `column_mapping`.

### Existing local artifacts

Some build steps prompt before overwriting existing assets. This is expected behavior and helps protect generated data.

---

## Next steps

After the initial setup, the best next pages are usually:

- **Configuration**
- **Pipeline components**
- **Registries**
- **CLI usage**
- **Streamlit UI**
- **Advanced customization**
- **API reference**
