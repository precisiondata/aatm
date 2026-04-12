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
