"""The concept search agent as a selector of the terminology-mapping pipeline.

`AgenticSelector` replaces the selector of a `aatm.terminology_mapper.TerminologyMapper`.
It starts from the candidates the mapper's retriever (and reranker) found, and the agent
searches further when none of them fits:

```python
from aatm.concept_search import AgenticSelector
from aatm.terminology_mapper import TerminologyMapper

mapper = TerminologyMapper(
    retriever="text-embedding-3-small",
    selector=AgenticSelector("openai/gpt-5-mini"),
)
df = mapper.map(file_path="source_concepts.csv")
df[["source_code_description", "target_concept_id", "confidence_score", "mapping_rationale"]]
```

The selector uses the mapper's retriever for its own searches, and the vocabulary
database built by ``aatm init`` (``.aatm/omop.db``) for lexical search, code lookups,
concept details and hierarchy. For source concepts, the source code and vocabulary are
given to the agent as context.
"""

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from aatm.concept_search.agent import (
    CodingContext,
    ConceptSearchAgent,
    ConceptSearchResult,
    run_sync,
)
from aatm.concept_search.tools import (
    Candidate,
    ConceptSearchTools,
    VocabularySearch,
    candidates_from_expressions,
)
from aatm.data_models import (
    EmptySelectionMetadata,
    RetrieverResults,
    SelectedExpressionMetadata,
    SelectorResults,
)
from aatm.data_models.data_models import ExpressionOrigin, StandardVocabulary
from aatm.llm import LLMSettings, StructuredLLM
from aatm.omop.vocabulary import DEFAULT_VOCABULARY_DB
from aatm.selectors import BaseSelector

INDEXED_VOCABULARIES = [v.value for v in StandardVocabulary]


def scope_from_where(
    where: Optional[dict[str, Any]],
) -> tuple[Optional[str], Optional[list[str]]]:
    """The domain and vocabularies of a retriever filter built by `retriever_filter`.

    Raises:
        ValueError: For filters on other fields, which the agent's tools cannot apply.
    """
    if not where:
        return None, None
    conditions = where["$and"] if set(where) == {"$and"} else [where]
    domain: Optional[str] = None
    vocabularies: Optional[list[str]] = None
    for condition in conditions:
        for key, value in condition.items():
            if key == "std_domain_id" and isinstance(value, str):
                domain = value
            elif key == "std_domain_id" and isinstance(value, dict) and "$eq" in value:
                domain = value["$eq"]
            elif key == "std_vocabulary_id" and isinstance(value, str):
                vocabularies = [value]
            elif (
                key == "std_vocabulary_id"
                and isinstance(value, dict)
                and "$in" in value
            ):
                vocabularies = list(value["$in"])
            else:
                raise ValueError(
                    f"AgenticSelector understands filters on std_domain_id and "
                    f"std_vocabulary_id only, got {condition!r}"
                )
    return domain, vocabularies


def source_context(source_concept: Any, text: str) -> CodingContext:
    """The context of a `SourceConcept`: its code and vocabulary, and the original text."""
    related: dict[str, str] = {}
    for field in ("source_code", "source_vocabulary_id"):
        value = getattr(source_concept, field, None)
        if value and str(value).lower() not in ("nan", "none"):
            related[field] = str(value)
    original = getattr(source_concept, "source_code_description", None)
    if original and original != text:
        related["original_text"] = original
    return CodingContext(text=text, related=related)


class AgenticSelector(BaseSelector):
    """Selects concepts with `ConceptSearchAgent`, searching beyond the retrieved candidates.

    Args:
        model_id: An `aatm.llm` model id, e.g. ``"openai/gpt-5-mini"``, or a
            `StructuredLLM`.
        retriever: The retriever for the agent's semantic searches. Defaults to the
            retriever of the `TerminologyMapper` the selector is used in.
        vocabulary: Path to the vocabulary database, or a `VocabularySearch`. Defaults
            to ``.aatm/omop.db`` when it exists; without it, the agent only has
            semantic search.
        domain: Only select concepts of this domain. A domain in the mapper's
            ``where`` filter takes precedence.
        vocabularies: Only select concepts of these vocabularies. Defaults to the
            vocabularies of the mapper's index (SNOMED, RxNorm, LOINC).
        max_tool_calls: See `ConceptSearchAgent`.
        top_k: Results per search.
        settings: See `ConceptSearchAgent`.
        max_concurrency: Texts searched at the same time.
    """

    uses_context = True

    def __init__(
        self,
        model_id: str | StructuredLLM,
        retriever: Any = None,
        vocabulary: VocabularySearch | str | Path | None = None,
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
        max_tool_calls: int = 8,
        top_k: int = 10,
        settings: Optional[LLMSettings | dict[str, Any]] = None,
        max_concurrency: int = 8,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        unknown = set(vocabularies or []) - set(INDEXED_VOCABULARIES)
        if unknown:
            raise ValueError(
                f"The terminology mapper only maps to {INDEXED_VOCABULARIES}, not "
                f"{sorted(unknown)}"
            )
        if vocabulary is None and DEFAULT_VOCABULARY_DB.exists():
            vocabulary = DEFAULT_VOCABULARY_DB
        self.model_id = model_id
        self.vocabulary = vocabulary
        self.domain = domain
        self.vocabularies = list(vocabularies) if vocabularies else INDEXED_VOCABULARIES
        self.max_tool_calls = max_tool_calls
        self.top_k = top_k
        self.settings = settings
        self.max_concurrency = max_concurrency
        self.agent: Optional[ConceptSearchAgent] = None
        if retriever is not None or vocabulary is not None:
            self.bind(retriever)

    def bind(self, retriever: Any) -> None:
        """Set the retriever of the agent's semantic searches (if none is set yet)."""
        if self.agent is not None and self.agent.tools.retriever is not None:
            return
        tools = ConceptSearchTools(retriever, self.vocabulary, top_k=self.top_k)
        if self.agent is None:
            self.agent = ConceptSearchAgent(
                self.model_id,
                tools,
                max_tool_calls=self.max_tool_calls,
                settings=self.settings,
                top_k=self.top_k,
            )
        else:
            self.agent.tools = tools

    def select(  # type: ignore[override]
        self,
        results: RetrieverResults,
        contexts: Optional[Sequence[CodingContext]] = None,
        where: Optional[dict[str, Any]] = None,
    ) -> SelectorResults:
        """Select a concept for each query.

        Args:
            results: The retrieved candidates of each query.
            contexts: The context of each query. Defaults to the query alone.
            where: The retriever filter of this call; its domain and vocabularies
                restrict the agent's searches too.
        """
        if self.agent is None:
            raise ValueError(
                "AgenticSelector has no retriever or vocabulary: use it in a "
                "TerminologyMapper, or pass retriever=..."
            )
        domain, vocabularies = scope_from_where(where)
        domain = domain or self.domain
        if vocabularies:
            vocabularies = [v for v in vocabularies if v in self.vocabularies]
        else:
            vocabularies = self.vocabularies
        if contexts is None:
            contexts = [CodingContext(text=q) for q in results.queries]
        if len(contexts) != len(results.queries):
            raise ValueError(
                f"Got {len(contexts)} contexts for {len(results.queries)} queries"
            )
        items = [
            (context, candidates_from_expressions(expressions))
            for context, expressions in zip(contexts, results.results)
        ]
        found: list[ConceptSearchResult] = run_sync(
            self.agent.search_many(
                items, domain, vocabularies, max_concurrency=self.max_concurrency
            )
        )
        return SelectorResults(
            results=[
                selection(result, expressions)
                for result, expressions in zip(found, results.results)
            ],
            queries=results.queries,
        )


def selection(
    result: ConceptSearchResult, expressions: Sequence[Any]
) -> SelectedExpressionMetadata | EmptySelectionMetadata:
    """A selector result from an agent result.

    The retrieved expression of the chosen concept is used when there is one;
    otherwise the concept's name, with ``result_list_index = -1``. The distance is
    ``1 - score``, so the mapper's ``confidence_score`` is the agent's score.
    """
    review: dict[str, Any] = {
        "rationale": result.rationale,
        "trace": json.dumps(result.trace, ensure_ascii=False),
        "usage": result.usage(),
    }
    concept: Optional[Candidate] = result.concept
    if concept is None:
        return EmptySelectionMetadata(**review)
    distance = 1 - result.score if result.score is not None else None
    for index, expression in enumerate(expressions):
        if (
            expression.std_concept_id
            and int(expression.std_concept_id) == concept.concept_id
        ):
            data = expression.model_dump(exclude={"distance", "expression_id"})
            return SelectedExpressionMetadata(
                **data, distance=distance, result_list_index=index, **review
            )
    return SelectedExpressionMetadata(
        expression=concept.concept_name,
        expression_concept_id=str(concept.concept_id),
        expression_origin=ExpressionOrigin.STANDARD_CONCEPT,
        std_concept_id=str(concept.concept_id),
        std_concept_name=concept.concept_name,
        std_vocabulary_id=StandardVocabulary(concept.vocabulary_id),
        std_vocabulary_code=concept.concept_code,
        std_domain_id=concept.domain_id,
        distance=distance,
        result_list_index=-1,
        **review,
    )
