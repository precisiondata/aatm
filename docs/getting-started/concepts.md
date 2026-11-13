# Basic concepts

## TerminologyMapper

`TerminologyMapper` is the main orchestration class of AATM.

It coordinates the full mapping workflow by combining the pipeline components into an end-to-end process. In practice, it is responsible for:

- loading source concepts from an input file
- optionally translating source descriptions
- retrieving candidate standard concepts
- optionally reranking retrieved results
- selecting the final mapped concept
- returning the results as a DataFrame
- writing the mapped output to disk

It is the main entry point when you want to run terminology mapping in batch mode.

Typical usage:

```python
from aatm.terminology_mapper import TerminologyMapper

mapper = TerminologyMapper(
    translator="empty-translator",
    retriever="embeddinggemma-300M",   # optional: the index aatm init built
    reranker="bm25-reranker",
    selector="gpt-5-mini",
    batch_size=100,
)

results_df = mapper.map(file_path="source_to_concept_map.csv")
results_df = mapper.map(["chest pain", "cough"], save_to_disk=False)
```

Each component can be given as a registry id, as above, or as an object. A configured
mapper is also a **coder** for OMOP tables (see [below](#notes-to-omop-tables)).

## Translator

Translators convert input text into English before retrieval.

Examples:

- `EmptyTranslator`
- `GeminiTranslator`
- `OpenAITranslator`

Use `EmptyTranslator` when your inputs are already in the desired language.

## Retriever

Retrievers fetch candidate concepts from a vector database.

Implementations, both returned by `load_retriever()` depending on `AATM_VECTOR_DB`
(see [Environment variables](env.md#vector-database-local-chroma-cloud-or-pinecone)):

- `ChromaDBRetriever`: a local or Chroma Cloud collection.
- `PineconeRetriever`: a Pinecone index.

`aatm init` and `aatm vector-db upload` record the embedding model they built, the vector
database and the build date in `.aatm/config.yaml`. Without a model name,
`load_retriever()`, `TerminologyMapper()`, `AgenticCoder(model_id)` and `aatm map` use
the most recently built one for the configured vector database (and log which), or,
with nothing recorded, the only built one. `aatm vector-db check` shows that default.

A retriever only opens an index that is built: naming a model whose index does not
exist or is empty raises `VectorDatabaseNotBuiltError` instead of searching an empty
collection. The message lists the indexes that are built and how to build the one
asked for:

```text
No vector database for 'text-embedding-3-large' (.aatm/chroma_vector_dbs/text-embedding-3-large not found).
Built and ready:
  - text-embedding-3-small  local  .aatm/chroma_vector_dbs/text-embedding-3-small  (2,512,331 expressions)
Use retriever="text-embedding-3-small", or build it with: aatm init --embedding-model text-embedding-3-large
```

Both share their filtering logic (`VectorStoreRetriever`), and distances mean the same
on both, so selectors and thresholds do not depend on the backend.

### Metadata filters

A retriever can restrict its candidates with a `where` filter, written in ChromaDB's
syntax, which AATM translates for Pinecone. For example, to restrict them to
one domain and some vocabularies. The OMOP builder does this for every `map` source:

```python
from aatm.omop.coding import retriever_filter

where = retriever_filter("Measurement", ["LOINC"])
# {"$and": [{"std_domain_id": "Measurement"}, {"std_vocabulary_id": {"$in": ["LOINC"]}}]}
results = retriever(["hemoglobin"], where=where)
```

ChromaDB applies a filter before the vector search, which takes seconds per query on a
full index. So by default `ChromaDBRetriever` queries **without** the filter for more
results than it needs, keeps the ones that match, and asks ChromaDB's filtered query
only when too few match. `filter_strategy` chooses the method:

| `filter_strategy` | What it does | Results |
|---|---|---|
| `"overfetch_fallback"` (default) | Unfiltered query for `max(top_k * overfetch_factor, min_overfetch)` results, then for `max_overfetch` results, then ChromaDB's filtered query, for each query still short of `top_k` matches | Always `top_k` results; about as fast as no filter when the over-fetch finds enough |
| `"overfetch"` | The same, without the filtered query at the end | Fast, but may return fewer than `top_k` results for rare filters |
| `"native"` (or `"chroma"`) | Passes the filter to the vector database | ChromaDB: exact, slow on large indexes. Pinecone: the default, filtered during the search |

Set it when you load the retriever, or per call:

```python
retriever = load_retriever(
    "text-embedding-3-small",
    filter_strategy="overfetch_fallback",  # default
    overfetch_factor=20,                   # defaults
    min_overfetch=1000,
    max_overfetch=5000,
)
retriever(["hemoglobin"], where=where, filter_strategy="chroma")
```

Both methods are approximate vector searches, so their results can differ slightly. The
over-fetch can miss matches that rank deep in the unfiltered results, which happens when
the filter fits the query poorly (a lab test searched in the Condition domain). Raise
`min_overfetch` and `max_overfetch` for such filters. ChromaDB's filtered search can miss
close matches too: on a full index it sometimes returned none of the 20 nearest
Measurement records for a lab test.

The over-fetch evaluates equality, `$eq`, `$in`, `$and` and `$or`, which covers every
filter `retriever_filter` builds. The default strategy hands any other filter to
ChromaDB; `"overfetch"` raises `UnsupportedFilterError` for it.

Pinecone applies filters during the vector search, so `PineconeRetriever` defaults to
`filter_strategy="native"` and needs no over-fetch. Filters are translated to
Pinecone's syntax: `{"field": value}` becomes `{"field": {"$eq": value}}`, and `$ne`,
`$gt`, `$gte`, `$lt`, `$lte`, `$in` and `$nin` keep their names.

## Reranker

Rerankers reorder retrieved candidates.

Examples:

- `BM25Reranker`
- `Qwen3Reranker`

## Selector

Selectors choose the final mapped concept from the retrieved candidates.

| Selector | Chooses by |
|---|---|
| `FirstResultSelector` | The closest candidate (default) |
| `OpenAILLMSelector`, `GeminiLLMSelector` | An LLM, among the candidates |
| `DecisionSelector` | A [decision model](../reference/decision_models.md), with probabilities |
| `AgenticSelector` | An agent that starts from the candidates and searches the vocabularies with tools when none fits |

## Information extraction

The extraction pipeline turns clinical text into structured entities that can later be mapped.

- **Entity definitions** (`EntityDefinitions`) describe what to extract. They are loaded from YAML, either predefined (`condition_occurrence`, `drug_exposure`, `ecog_performance_status`, `lab_results`, `smoking_status`, `family_history`, `medical_history`, `device_exposure`) or your own. An entity can require a parent, such as a start date that belongs to a condition, and can define attributes: labels from a fixed list assigned to each mention, such as the result of a test. See [Writing entity definitions](entity_definitions.md).
- **Extractor** (`SimpleLLMExtractor`) sends each document to an LLM as numbered lines. The model returns each mention with an id, its exact text, its line and the ids of its parents.
- **Validation** (`validate_extracted_entity`) keeps only mentions that exist on the given line, checks keys, parent links and attribute labels, and records why every rejected mention failed. A wrong label on an optional attribute is reported without dropping the mention.
- **Records** (`EntityValidationResult.to_records()`) nest validated entities under their parents, ready to become table rows.
- **Attribute labeller** (`DecisionAttributeLabeller`, optional) labels or verifies attributes with a [decision model](../reference/decision_models.md) instead of the extraction call.

See the [Information extraction quick start](quick_start_extraction.md).

## Notes to OMOP tables

The OMOP pipeline adds three pieces after extraction:

- **`omop` blocks** in the entity definitions say which CDM table an entity fills and where each column's value comes from: the mention's text, a child, an attribute's label, the note's context, a fixed code, a code lookup, or the coder. They are never sent to the model. See [From notes to OMOP tables](omop_tables.md).
- **Coder** (`TextCoder`): turns extracted text into a standard concept id, for `map` sources. A configured `TerminologyMapper`, the concept search agent (`AgenticCoder`), or your own. Results can be cached on disk across runs. See [Coding extracted text](coding_text.md).
- **Table builder** (`OmopTableBuilder`): builds the rows from the validated entities, checks every concept against the vocabulary database (standard, right domain), and records each row's provenance and every gap as an issue.

`OmopExtractionPipeline` chains the extractor and the builder. Calling it builds tables in memory; its `run` method processes a dataset of any size in chunks, with a **ledger** (`ledger.db`) of every extraction, so a run resumes, retries failed notes, and can be **rebuilt** after a mapping change without calling the model again. `OmopRun` reads a run's output. See [Processing large datasets](large_datasets.md).

A **`ClinicalNote`** is a note's text with the values that come with it: `note_id`, `person_id`, `note_date`, `visit_occurrence_id`, `visit_detail_id`, `provider_id`.
