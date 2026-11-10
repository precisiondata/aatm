"""Coding extracted text with the terminology mapper.

`OmopTableBuilder` codes ``{from: ..., map: ...}`` sources through a `TextCoder`.
`TerminologyMapperCoder` adapts a configured `aatm.terminology_mapper.TerminologyMapper`
(translator, retriever, reranker and selector), so extraction and terminology mapping
share one pipeline:

```python
from aatm.terminology_mapper import TerminologyMapper
from aatm.omop import OmopTableBuilder

mapper = TerminologyMapper(retriever="text-embedding-3-small")
builder = OmopTableBuilder(definitions, mapper=mapper)  # wrapped automatically
```

`AgenticCoder` codes each text with an LLM agent that searches the vocabularies with
tools, given the text's context (related values such as the unit, and the sentence of
the note); see `aatm.concept_search`:

```python
from aatm.omop import AgenticCoder

builder = OmopTableBuilder(
    definitions, mapper=AgenticCoder("openai/gpt-5-mini", retriever="text-embedding-3-small")
)
```

The same agent can be the selector of a `TerminologyMapper`
(`aatm.concept_search.AgenticSelector`); the builder then passes it the context too.
"""

import json
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence

from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover
    from aatm.concept_search.agent import CodingContext, ConceptSearchResult


class CodedText(BaseModel):
    """The standard concept a text was mapped to.

    Attributes:
        text: The text that was mapped.
        concept_id: The selected standard concept, or ``None`` when nothing was
            selected.
        concept_name: Its name.
        vocabulary_id: Its vocabulary, e.g. ``"LOINC"``.
        concept_code: Its code in that vocabulary.
        domain_id: Its domain, e.g. ``"Measurement"``.
        score: The mapper's confidence, from 0 to 1 (``1 - distance`` for the default
            selector, a confidence level for the agent), when available.
        standard: Whether the concept is standard. ``None`` when unknown.
        rationale: Why the concept was chosen, for coders that explain (the agent).
        trace: The searches made to find it, for coders that search (the agent).
        usage: Tool calls, LLM calls, tokens and seconds spent, when reported. For a
            cached result, what was spent when it was first coded.
        error: The error that stopped the coder, if any. Such results are not cached.
        cached: Whether the result came from a `aatm.omop.CodingCache`, rather than
            being coded in this run.
    """

    text: str
    concept_id: Optional[int] = None
    concept_name: Optional[str] = None
    vocabulary_id: Optional[str] = None
    concept_code: Optional[str] = None
    domain_id: Optional[str] = None
    score: Optional[float] = None
    standard: Optional[bool] = None
    rationale: Optional[str] = None
    trace: Optional[list[dict[str, Any]]] = None
    usage: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    cached: bool = False


class TextCoder(Protocol):
    """Maps texts to standard concepts, optionally restricted to a domain and vocabularies.

    A coder with ``uses_context = True`` also accepts ``contexts``: one
    `aatm.concept_search.CodingContext` per text (related values, the sentence of the
    note, the target column). The builder then codes each distinct text *in its
    context* once, rather than each distinct text once.

    To be cached (`aatm.omop.CachedCoder`), a coder also describes itself with
    ``cache_identity()``: JSON-compatible values covering everything that changes its
    results (models, settings, prompts, index).
    """

    def code(
        self,
        texts: Sequence[str],
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
    ) -> list[CodedText]:
        """Return one `CodedText` per text, in order."""
        ...


def uses_context(coder: Any) -> bool:
    """Whether a coder (or a selector) reads the context of the texts."""
    return bool(getattr(coder, "uses_context", False))


def retriever_filter(
    domain: Optional[str], vocabularies: Optional[Sequence[str]]
) -> Optional[dict[str, Any]]:
    """Build the ChromaDB metadata filter that restricts retrieval to a domain and vocabularies."""
    conditions: list[dict[str, Any]] = []
    if domain is not None:
        conditions.append({"std_domain_id": domain})
    if vocabularies:
        conditions.append({"std_vocabulary_id": {"$in": list(vocabularies)}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class TerminologyMapperCoder:
    """`TextCoder` that runs texts through a `TerminologyMapper`.

    Domain and vocabularies become a filter on the mapper's retriever, so only
    matching standard concepts are candidates. When the mapper's selector reads
    context (`aatm.concept_search.AgenticSelector`), the contexts are passed on.
    """

    def __init__(self, mapper: Any):
        self.mapper = mapper

    @property
    def uses_context(self) -> bool:
        return uses_context(getattr(self.mapper, "selector", None))

    def cache_identity(self) -> dict[str, Any]:
        """The mapper's components and settings, for `aatm.omop.CachedCoder`."""
        from aatm.omop.cache import describe_component

        return {
            "coder": "TerminologyMapperCoder",
            "mapper": describe_component(self.mapper),
        }

    def code(
        self,
        texts: Sequence[str],
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
        contexts: Optional[Sequence["CodingContext"]] = None,
    ) -> list[CodedText]:
        if not texts:
            return []
        kwargs: dict[str, Any] = {}
        if contexts is not None and self.uses_context:
            kwargs["contexts"] = list(contexts)
        mapped = self.mapper.map(
            expressions=list(texts),
            return_as="mapped_source_concepts",
            save_to_disk=False,
            where=retriever_filter(domain, vocabularies),
            **kwargs,
        )
        return [
            CodedText(
                text=text,
                concept_id=int(m.target_concept_id) if m.target_concept_id else None,
                # MappedSourceConcept has no concept name; the builder adds it from
                # the vocabulary database.
                vocabulary_id=getattr(
                    m.target_vocabulary_id, "value", m.target_vocabulary_id
                ),
                concept_code=m.target_vocabulary_code,
                domain_id=m.domain_id,
                score=m.confidence_score,
                # The mapper's index holds standard concepts only.
                standard=True if m.target_concept_id else None,
                rationale=getattr(m, "mapping_rationale", None),
                trace=_load_trace(getattr(m, "mapping_trace", None)),
            )
            for text, m in zip(texts, mapped)
        ]


def _load_trace(trace: Optional[str]) -> Optional[list[dict[str, Any]]]:
    return json.loads(trace) if trace else None


class AgenticCoder:
    """`TextCoder` that codes each text with `aatm.concept_search.ConceptSearchAgent`.

    The agent gets each text with its context (related values such as the value and
    unit of a lab result, the line of the note, the target column) and searches the
    vocabularies with tools until it finds the standard concept or concludes there is
    none. See `aatm.concept_search` for the tools and the confidence scale of
    ``score``.

    Args:
        model_id: An `aatm.llm` model id, e.g. ``"openai/gpt-5-mini"``, or a
            `aatm.llm.StructuredLLM`.
        retriever: The retriever for semantic search: a registry name such as
            ``"text-embedding-3-small"`` (the embedding model chosen in
            ``aatm init``) or a `aatm.retrievers.BaseRetriever`. ``None`` leaves
            only the vocabulary tools.
        vocabulary: The vocabulary database (path or
            `aatm.concept_search.VocabularySearch`) for lexical search, code
            lookups, concept details and hierarchy. Defaults to ``.aatm/omop.db``.
        max_tool_calls: Tool calls per text, after the initial search.
        top_k: Results per search.
        settings: Generation settings (`aatm.llm.LLMSettings`), e.g.
            ``{"temperature": 0}`` on models that accept it.
        max_concurrency: Texts searched at the same time.
    """

    uses_context = True

    def __init__(
        self,
        model_id: Any,
        retriever: Any = None,
        vocabulary: Any = None,
        max_tool_calls: int = 8,
        top_k: int = 10,
        settings: Any = None,
        max_concurrency: int = 8,
    ):
        from aatm.concept_search.agent import ConceptSearchAgent
        from aatm.concept_search.tools import ConceptSearchTools
        from aatm.omop.vocabulary import DEFAULT_VOCABULARY_DB

        if isinstance(retriever, str):
            from aatm.registries.retrievers import load_retriever

            retriever = load_retriever(retriever)
        if vocabulary is None:
            vocabulary = DEFAULT_VOCABULARY_DB
        self.agent = ConceptSearchAgent(
            model_id,
            ConceptSearchTools(retriever, vocabulary, top_k=top_k),
            max_tool_calls=max_tool_calls,
            settings=settings,
            top_k=top_k,
        )
        self.max_concurrency = max_concurrency

    def cache_identity(self) -> dict[str, Any]:
        """The agent's model, settings, tools and prompts, for `aatm.omop.CachedCoder`.

        ``max_concurrency`` is left out: it does not change the results.
        """
        from aatm.omop.cache import describe_component

        return {"coder": "AgenticCoder", "agent": describe_component(self.agent)}

    def code(
        self,
        texts: Sequence[str],
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
        contexts: Optional[Sequence["CodingContext"]] = None,
    ) -> list[CodedText]:
        from aatm.concept_search.agent import CodingContext, run_sync

        if not texts:
            return []
        if contexts is None:
            contexts = [CodingContext(text=t) for t in texts]
        if len(contexts) != len(texts):
            raise ValueError(f"Got {len(contexts)} contexts for {len(texts)} texts")
        results = run_sync(
            self.agent.search_many(
                [(c, None) for c in contexts],
                domain,
                vocabularies,
                max_concurrency=self.max_concurrency,
            )
        )
        return [coded_text(text, r) for text, r in zip(texts, results)]


def coded_text(text: str, result: "ConceptSearchResult") -> CodedText:
    """A `CodedText` from an agent result."""
    concept = result.concept
    return CodedText(
        text=text,
        concept_id=concept.concept_id if concept else None,
        concept_name=concept.concept_name if concept else None,
        vocabulary_id=concept.vocabulary_id if concept else None,
        concept_code=concept.concept_code if concept else None,
        domain_id=concept.domain_id if concept else None,
        score=result.score,
        # The agent only answers with standard concepts.
        standard=True if concept else None,
        rationale=result.rationale,
        trace=result.trace,
        usage=result.usage(),
        error=result.error,
    )
