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

Main implementation:

- `ChromaDBRetriever`

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

- **Entity definitions** (`EntityDefinitions`) describe what to extract. They are loaded from YAML, either predefined (`condition_occurrence`, `drug_exposure`, `ecog_performance_status`, `lab_results`, `smoking_status`, `family_history`) or your own. An entity can require a parent, such as a start date that belongs to a condition, and can define attributes: labels from a fixed list assigned to each mention, such as the result of a test. See [Writing entity definitions](entity_definitions.md).
- **Extractor** (`SimpleLLMExtractor`) sends each document to an LLM as numbered lines. The model returns each mention with an id, its exact text, its line and the ids of its parents.
- **Validation** (`validate_extracted_entity`) keeps only mentions that exist on the given line, checks keys, parent links and attribute labels, and records why every rejected mention failed. A wrong label on an optional attribute is reported without dropping the mention.
- **Records** (`EntityValidationResult.to_records()`) nest validated entities under their parents, ready to become table rows.

See the [Information extraction quick start](quick_start_extraction.md).
