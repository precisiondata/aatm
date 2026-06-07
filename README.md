# Any-to-Any Terminology Mapping

**Any-to-Any Terminology Mapping** is an open-source Python framework designed to facilitate terminology mapping tasks. The library organizes this process in a modular and extensible way to support multiple use cases and incorporate new techniques as they emerge.

In simple terms, mapping a new expression to a specific terminology involves considering many possible expressions, retrieving the best candidate target terms, and selecting them manually, which can be effortful and time-consuming.

AATM leverages the [OMOP vocabularies](https://athena.ohdsi.org/search-terms/start) to facilitate this task. These vocabularies reflect large, community-driven mapping efforts that connect many different health-related terminologies and classifications worldwide and organize them around standard terminologies, which serve as the central connecting nodes in the system. As these mapping efforts continue, healthcare-related concepts become increasingly well represented in these vocabularies, creating a virtuous cycle and increasing the chances of finding a strong correspondence for a new unmapped expression.

<p align="center">
  <img src="https://github.com/precisiondata/aatm/blob/main/docs/assets/std_vocabs_diagram.png?raw=true" alt="Standard vocabularies">
</p>

To accomplish this, AATM organizes the mapping process into very simple steps: 

- **translation**, which can be optional; 
- **retrieval**, which explores what is available from prior mapping efforts; and
- **selection**, which connects a standard concept to the new expression being mapped. 

Once this connection is made, every link associated with that standard concept becomes immediately available, enabling mapping to many different terminologies and classifications that are already connected to that concept, effectively breaking down barriers to interoperability in healthcare.

![Terminology mapping pipeline](https://github.com/precisiondata/aatm/blob/main/docs/assets/aatm-pipeline-diagram.jpg?raw=true)


## Documentation

The full documentation is available at: https://precisiondata.github.io/aatm

## Contributing

If you want to contribute to this project, please refer to our [Contributing guide](./CONTRIBUTING.md)

## Installation


Install the package in your environment.

```bash
pip install aatm
```

If you want to build from the source, clone the repository and install it locally:

```bash
git clone https://github.com/precisiondata/aatm.git
pip install -e . 
```

```bash
git clone https://github.com/precisiondata/aatm.git
uv sync
```

If you plan to run language models locally with CUDA acceleration, you must also install CUDA extensions for PyTorch. 

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

```bash
uv pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

## 1. Prepare your OMOP vocabularies directory

Before running `aatm init`, download the OMOP vocabularies you want to use and place them in a directory. You can find them at https://athena.ohdsi.org/vocabulary/list

By default, the CLI expects it at the root directory:

```text
./vocabularies
```

If you do not use that location, you can point the CLI to a different directory with the option `--vocab-dir` or `-vd`. The CLI validates this path during initialization.

---

## 2. Run the initialization command

The `init` command is the main CLI setup workflow.

It does all of the following for you:

- creates the local `.aatm` helper directory where the local databases and aatm config files will be stored
- ensures `.aatm` is added to `.gitignore`
- builds the local OMOP SQLite database
- lets you choose an embedding model
- lets you choose the standard vocabularies
- builds the mapping datasets
- builds the local vector database

That means you do **not** need to call Python setup functions manually for the normal setup flow. At the end, you will be ready to run terminology mapping tasks.

### Simplest setup

```bash
aatm init
```

This uses the default vocab directory and interactively asks you to choose the embedding model, standard vocabularies and other options.

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

## 3. Prepare your input CSV

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

## 4. Run mapping directly from the CLI

The `map` command runs a terminology mapping task. You can use it in two ways:

- with a task config file
- with explicit CLI options

Both paths are supported directly by the CLI implementation.

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

#### Run a small test job

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

#### Apply rate limiting

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

The CLI accepts all of these options directly.
