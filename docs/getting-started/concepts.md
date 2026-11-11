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
from aatm.registries.translators import load_translator
from aatm.registries.retrievers import load_retriever
from aatm.registries.rerankers import load_reranker
from aatm.registries.selectors import load_selector

mapper = TerminologyMapper(
    translator=load_translator("empty-translator"),
    retriever=load_retriever("embeddinggemma-300M"),
    reranker=load_reranker("bm25-reranker"),
    selector=load_selector("first-result-selector"),
    batch_size=100,
)

results_df = mapper.map("source_to_concept_map.csv")
```

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

Examples:

- `FirstResultSelector`
- `OpenAILLMSelector`
- `GeminiLLMSelector`

## Information extraction

The extraction pipeline turns clinical text into structured entities that can later be mapped.

- **Entity definitions** (`EntityDefinitions`) describe what to extract. They are loaded from YAML, either predefined (`condition_occurrence`, `drug_exposure`, `ecog_performance_status`, `lab_results`, `smoking_status`, `family_history`, `medical_history`, `device_exposure`) or your own. An entity can require a parent, such as a start date that belongs to a condition, and can define attributes: labels from a fixed list assigned to each mention, such as the result of a test. See [Writing entity definitions](entity_definitions.md).
- **Extractor** (`SimpleLLMExtractor`) sends each document to an LLM as numbered lines. The model returns each mention with an id, its exact text, its line and the ids of its parents.
- **Validation** (`validate_extracted_entity`) keeps only mentions that exist on the given line, checks keys, parent links and attribute labels, and records why every rejected mention failed. A wrong label on an optional attribute is reported without dropping the mention.
- **Records** (`EntityValidationResult.to_records()`) nest validated entities under their parents, ready to become table rows.

See the [Information extraction quick start](quick_start_extraction.md).
