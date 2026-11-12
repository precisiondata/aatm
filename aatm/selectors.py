"""
Define selector abstractions and concrete selection strategies for retrieval results.

This module provides selector components used in the terminology-mapping
pipeline to choose the best candidate from a set of retrieved expressions. It
includes a simple rule-based selector that picks the first result under a
distance threshold, as well as LLM-based selectors backed by OpenAI and Gemini
models, and a selector backed by decision models (`DecisionSelector`).

Selectors are designed to operate as pipeline stages that consume
``RetrieverResults`` and return structured ``SelectorResults`` objects.
"""

from enum import Enum
from typing import Optional, Any, Dict, List, Tuple
import dotenv
from abc import ABC, abstractmethod
from aatm.llm.gemini import gemini_client
from google.genai import types

from openai import OpenAI

# Custom modules
from aatm.data_models import (
    EmptySelectionMetadata,
    RetrieverResults,
    SelectedExpressionMetadata,
    SelectedResult,
    SelectorResults,
)
from aatm.debug import DebugMode, get_debug_mode
from aatm.logs import get_logger
from aatm.pipeline import PipelineBaseClass
from aatm.prompt_helpers import format_prompt

# Load environment variables
dotenv.load_dotenv()

logger = get_logger(__name__)
debug_mode = get_debug_mode()


class BaseSelector(PipelineBaseClass, ABC):
    """Define the abstract interface for selector pipeline components.

    This base class establishes the contract for selectors that choose a final
    candidate from retrieval results. Subclasses must implement ``select()``,
    while the base ``__call__()`` method provides pipeline-compatible
    invocation.
    """

    @abstractmethod
    def select(self, results: RetrieverResults) -> SelectorResults:
        """Select the best candidate result for each query.

        Subclasses must implement this method to examine retrieval results and
        return a structured selection output for each query.

        Args:
            results: Retrieval results containing the original queries and their
                candidate matches.

        Returns:
            A ``SelectorResults`` instance containing the selected result for
                each query.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        pass

    def __call__(self, results: RetrieverResults) -> SelectorResults:
        """Invoke the selector on retrieval results.

        This method makes selector instances directly callable and delegates to
        ``select()``.

        Args:
            results: Retrieval results to process.

        Returns:
            A ``SelectorResults`` instance produced by the selector.
        """
        return self.select(results)


class FirstResultSelector(BaseSelector):
    """Select the first retrieved result that satisfies a distance threshold.

    This selector implements a simple heuristic strategy: it chooses the first
    candidate in each result list if its distance is below the configured
    threshold. If no candidate is available or the first candidate does not
    satisfy the threshold, an empty selection is returned.
    """

    def __init__(
        self, selection_threshold: float = float("inf"), *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the first-result selector.

        Args:
            selection_threshold: Maximum allowed distance for accepting the
                first retrieved result. If the first result has a larger
                distance, no selection is made.
            *args: Additional positional arguments reserved for compatibility.
            **kwargs: Additional keyword arguments reserved for compatibility.

        Returns:
            None.
        """
        self.selection_threshold = selection_threshold

    def select(self, results: RetrieverResults) -> SelectorResults:
        """Select the first valid result for each query.

        This method inspects the first candidate in each query result list and
        returns it as the selected expression when its distance is below the
        configured threshold. If no candidates are available or the threshold
        is not met, an empty selection is returned.

        Args:
            results: Retrieval results containing candidate expressions for
                each query.

        Returns:
            A ``SelectorResults`` object containing either a selected
                expression or an empty selection for each query.
        """
        selected_results: List[SelectedExpressionMetadata | EmptySelectionMetadata] = []
        for idx, r in enumerate(results.results):
            try:
                if (
                    r[0].distance is not None
                    and r[0].distance < self.selection_threshold
                ):
                    selected_results.append(
                        SelectedExpressionMetadata(
                            **r[0].model_dump(), result_list_index=0
                        )
                    )
                else:
                    selected_results.append(EmptySelectionMetadata())
            except IndexError:
                selected_results.append(EmptySelectionMetadata())
                print(f"No results for query: {results.queries[idx]}. Results: {r}")
        return SelectorResults(results=selected_results, queries=results.queries)


class OpenAILLMModels(Enum):
    """Enumerate the supported OpenAI model identifiers for selection.

    This enumeration defines the OpenAI model names that can be used by the
    OpenAI-based selector implementation.
    """

    GPT_52 = "gpt-5.2"
    GPT_5 = "gpt-5"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_MINI = "gpt-5-mini"


class OpenAILLMSelector(BaseSelector):
    """Select results using an OpenAI language model.

    This selector formats each query and its retrieved candidate expressions
    into a prompt, asks an OpenAI model to choose the semantically closest
    expression, and validates the structured JSON response against the expected
    selection schema.

    If the model returns no selection or selects an expression that is not
    present in the candidate list, the selector produces an empty selection.
    """

    def __init__(
        self,
        model_id: str,
        prompt_template: List[Dict[str, str]] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI-based selector.

        This constructor resolves the configured model identifier, creates an
        OpenAI client, and sets the prompt template used to ask the model to
        choose the best candidate expression.

        Args:
            model_id: Name of the supported OpenAI model variant to use for
                selection.
            prompt_template: Optional structured prompt template. If not
                provided, a default system-and-user prompt is used.
            *args: Additional positional arguments reserved for compatibility.
            **kwargs: Additional keyword arguments reserved for compatibility.

        Returns:
            None.

        Raises:
            ValueError: If ``model_id`` is not a valid member of
                ``OpenAILLMModels``.
        """
        self.model_id = OpenAILLMModels(model_id).value
        self.client = OpenAI()

        if prompt_template is None:
            self.prompt_template = [
                {
                    "role": "system",
                    "content": "Select the expression from the list of expressions that is semantically closest to the query. You are mapping a non-standard concept to a standard concept and the list of expressions are already mapped to standard concepts. Your response should be a JSON object and include the expression_id of the selected expression. If no expression matches the query, return '{\"expression_id\": null}'.\n\nJSON object format: {json_format}",
                },
                {
                    "role": "user",
                    "content": "Query: '{query}'\n\nExpressions: {expressions}",
                },
            ]
        else:
            self.prompt_template = prompt_template

    def select(self, results: RetrieverResults) -> SelectorResults:
        """Select the best candidate for each query using an OpenAI model.

        This method formats a prompt for each query, sends it to the configured
        OpenAI model for structured response parsing, validates the selected
        expression identifier, and returns either the matched candidate or an
        empty selection.

        Args:
            results: Retrieval results containing candidate expressions for
                each query.

        Returns:
            A ``SelectorResults`` object containing the selected candidate or
                an empty selection for each query.

        Raises:
            AssertionError: If the parsed response is not a ``SelectedResult``
                instance.
        """
        assert self.prompt_template is not None, "Prompt template is not set."
        selector_results = SelectorResults(results=[], queries=results.queries)
        for query_id, query in enumerate(results.queries):
            prompt = format_prompt(
                self.prompt_template,
                {
                    "json_format": SelectedResult.model_json_schema(),
                    "query": query.capitalize(),  # avoid case sensitivity for some queries like 'cough' and 'Cough'
                    "expressions": [
                        r.to_prompt_object() for r in results.results[query_id]
                    ],
                },
            )

            if debug_mode == DebugMode.OPENAI_LLM_SELECTOR:
                logger.debug(prompt)

            response = self.client.responses.parse(
                model=self.model_id,
                input=prompt,  # type: ignore[arg-type]
                text_format=SelectedResult,
            )

            if debug_mode == DebugMode.OPENAI_LLM_SELECTOR:
                logger.debug(response)
                logger.debug(response.output_parsed)

            selected_result: SelectedResult | None = response.output_parsed
            assert isinstance(selected_result, SelectedResult), (
                f"Expected SelectedResult object from OpenAI, but got {type(selected_result)}"
            )

            # case where no expression is selected
            if selected_result.expression_id is None:
                selector_results.results.append(EmptySelectionMetadata())
                continue

            # case where expression is selected but it is not in the results
            results_expression_ids = set(
                [r.expression_id for r in results.results[query_id]]
            )

            if selected_result.expression_id not in results_expression_ids:
                selector_results.results.append(EmptySelectionMetadata())
                continue

            # case where expression is selected and it is in the results
            for result_idx, r in enumerate(results.results[query_id]):
                if r.expression_id == selected_result.expression_id:
                    selector_results.results.append(
                        SelectedExpressionMetadata(
                            **r.model_dump(), result_list_index=result_idx
                        )
                    )
                    break

        return selector_results


class GeminiLLMModels(Enum):
    """Enumerate the supported Gemini model identifiers for selection.

    This enumeration defines the Gemini model names that can be used by the
    Gemini-based selector implementation.
    """

    GEMINI_3_PRO_PREVIEW = "gemini-3-pro-preview"
    GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_5_PRO = "gemini-2.5-pro"


class GeminiLLMSelector(BaseSelector):
    """Select results using a Gemini language model.

    This selector formats each query and its retrieved candidate expressions
    into a prompt, sends the prompt to a Gemini model configured for JSON
    output, and validates the returned structured selection against the
    expected schema.

    If the model returns no selection or selects an expression that is not
    present in the candidate list, the selector produces an empty selection.
    """

    def __init__(
        self,
        model_id: str,
        prompt_template: Optional[List[Dict[str, str]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the Gemini-based selector.

        This constructor resolves the configured Gemini model identifier,
        creates a Gemini client, and sets the prompt template used to ask the
        model to choose the best candidate expression.

        Args:
            model_id: Name of the supported Gemini model variant to use for
                selection.
            prompt_template: Optional structured prompt template. If not
                provided, a default system-and-user prompt is used.
            *args: Additional positional arguments reserved for compatibility.
            **kwargs: Additional keyword arguments reserved for compatibility.

        Returns:
            None.

        Raises:
            ValueError: If ``model_id`` is not a valid member of
                ``GeminiLLMModels``.
        """
        self.model_id = GeminiLLMModels(model_id).value
        self.client = gemini_client()

        if prompt_template is None:
            self.prompt_template = [
                {
                    "role": "system",
                    "content": "Select the expression from the list of expressions that is semantically closest to the query. You are mapping a non-standard concept to a standard concept and the list of expressions are already mapped to standard concepts. Your response should be a JSON object and include the expression_id of the selected expression. If no expression matches the query, return '{\"expression_id\": null}'.\n\nJSON object format: {json_format}",
                },
                {
                    "role": "user",
                    "content": "Query: '{query}'\n\nExpressions: {expressions}",
                },
            ]
        else:
            self.prompt_template = prompt_template

    def select(self, results: RetrieverResults) -> SelectorResults:
        """Select the best candidate for each query using a Gemini model.

        This method formats a prompt for each query, converts it into Gemini
        content objects, requests a JSON response constrained by the selection
        schema, validates the returned expression identifier, and returns
        either the matched candidate or an empty selection.

        Args:
            results: Retrieval results containing candidate expressions for
                each query.

        Returns:
            A ``SelectorResults`` object containing the selected candidate or
                an empty selection for each query.

        Raises:
            AssertionError: If the parsed response is not a ``SelectedResult``
                instance.
        """
        selector_results = SelectorResults(results=[], queries=results.queries)
        assert self.prompt_template is not None, "Prompt template not set"
        for query_id, query in enumerate(results.queries):
            prompt = format_prompt(
                self.prompt_template,
                {
                    "json_format": SelectedResult.model_json_schema(),
                    "query": query.capitalize(),  # avoid case sensitivity for some queries like 'cough' and 'Cough'
                    "expressions": [
                        r.to_prompt_object() for r in results.results[query_id]
                    ],
                },
            )

            gemini_formatted_prompt = []
            for msg in prompt:
                gemini_formatted_prompt.append(types.UserContent(msg["content"]))

            if debug_mode == DebugMode.OPENAI_LLM_SELECTOR:
                logger.debug(prompt)

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=gemini_formatted_prompt,  # type: ignore[arg-type]
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SelectedResult,
                ),
            )

            if debug_mode == DebugMode.GEMINI_LLM_SELECTOR:
                logger.debug(response)
                logger.debug(response.text)
                assert isinstance(response.text, str), (
                    "Expected response text to be a string"
                )
                logger.debug(SelectedResult.model_validate_json(response.text))

            selected_result: SelectedResult = SelectedResult.model_validate_json(
                response.text  # type: ignore[arg-type]
            )
            assert isinstance(selected_result, SelectedResult), (
                f"Expected SelectedResult object from OpenAI, but got {type(selected_result)}"
            )

            # case where no expression is selected
            if selected_result.expression_id is None:
                selector_results.results.append(EmptySelectionMetadata())
                continue

            # case where expression is selected but it is not in the results
            results_expression_ids = set(
                [r.expression_id for r in results.results[query_id]]
            )

            if selected_result.expression_id not in results_expression_ids:
                selector_results.results.append(EmptySelectionMetadata())
                continue

            # case where expression is selected and it is in the results
            for result_idx, r in enumerate(results.results[query_id]):
                if r.expression_id == selected_result.expression_id:
                    selector_results.results.append(
                        SelectedExpressionMetadata(
                            **r.model_dump(), result_list_index=result_idx
                        )
                    )
                    break

        return selector_results


# --- Decision models ------------------------------------------------------------------

NONE_OPTION = "none"
"""The option of `DecisionSelector`'s question meaning that no candidate fits."""

DECISION_SELECTOR_INSTRUCTIONS = (
    "The state is a clinical term from a source record. Which candidate standard "
    "concept means the same as the term? Answer 'none' if no candidate means the "
    "same thing: a broader, narrower or related concept is not the same."
)


def _candidate_description(expressions: List[Any]) -> str:
    """How a candidate concept is shown to the model: name, code, domain, matches."""
    first = expressions[0]
    vocabulary = getattr(first.std_vocabulary_id, "value", first.std_vocabulary_id)
    details = [
        f"{vocabulary} {first.std_vocabulary_code or ''}".strip() if vocabulary else "",
        first.std_domain_id or "",
    ]
    text = first.std_concept_name or first.expression or ""
    if any(details):
        text += " (" + ", ".join(d for d in details if d) + ")"
    matched = [
        e.expression
        for e in expressions
        if e.expression and e.expression != first.std_concept_name
    ]
    if matched:
        text += "; matched: " + "; ".join(repr(m) for m in dict.fromkeys(matched))
    return text


class DecisionSelector(BaseSelector):
    """Select candidates with a decision model (`aatm.llm.DecisionModel`).

    For each query, the selector asks the model one choice question about the
    query text, whose options are the top-k retrieved candidate concepts and
    ``none``. Retrieved expressions of the same standard concept (synonyms) make one
    option, so their probability is not split between them.

    The probability of the chosen concept is the result's score: the mapper's
    ``confidence_score`` and the OMOP builder's ``score``, so ``min_score: 0.8`` keeps
    the mappings the model gives at least 80 %. ``llm/`` decision models give no
    probabilities: their selections have **no score** (``confidence_score`` is
    ``None``), and a ``min_score`` rejects all of them.

    ```python
    from aatm.terminology_mapper import TerminologyMapper

    mapper = TerminologyMapper(selector="decision-openrouter/typesafe/jev-1.13")
    ```

    **Data governance:** the query texts and candidate names are sent to the
    decision provider.

    Args:
        model: A decision model id for `aatm.llm.create_decision_model`, e.g.
            ``"openrouter/typesafe/jev-1.13"``, or a `DecisionModel`. Pin a version.
        top_k: Candidates (distinct concepts) offered to the model, at most 254.
        instructions: The question asked about each query.
        max_concurrency: Queries decided at the same time.
        model_kwargs: Passed to `create_decision_model` (``api_key``, ``base_url``,
            ``settings``) when `model` is an id.
    """

    def __init__(
        self,
        model: Any,
        top_k: int = 10,
        instructions: str = DECISION_SELECTOR_INSTRUCTIONS,
        max_concurrency: int = 8,
        model_kwargs: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        from aatm.llm.decisions import MAX_CHOICES, DecisionModel

        if not 1 <= top_k <= MAX_CHOICES - 1:
            raise ValueError(f"top_k must be between 1 and {MAX_CHOICES - 1}")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if isinstance(model, str):
            from aatm.llm import create_decision_model

            model = create_decision_model(model, **(model_kwargs or {}))
        elif model_kwargs:
            raise ValueError("model_kwargs only applies when model is a model id")
        if not isinstance(model, DecisionModel):
            raise TypeError(
                "model must be a decision model id or a DecisionModel, got "
                f"{type(model).__name__}"
            )
        self.decision_model = model
        self.model_id: str = model.model_id
        self.top_k = top_k
        self.instructions = instructions
        self.max_concurrency = max_concurrency
        if not self.has_probabilities:
            logger.warning(
                "%s gives no probabilities: DecisionSelector's selections have no "
                "confidence score, and a min_score rejects all of them.",
                self.model_id,
            )

    @property
    def has_probabilities(self) -> bool:
        """Whether the model's answers have probabilities (not ``llm/`` models)."""
        from aatm.llm.decisions import has_probabilities

        return has_probabilities(self.decision_model)

    def cache_identity(self) -> Dict[str, Any]:
        """The model and the question, for `aatm.omop.CachedCoder`.

        ``max_concurrency`` is left out: it does not change the results.
        """
        from aatm.llm.decisions import model_identity

        return {
            "selector": "DecisionSelector",
            "top_k": self.top_k,
            "instructions": self.instructions,
            **model_identity(self.decision_model),
        }

    def question(self, candidates: List[List[Any]]) -> Any:
        """The choice question for a query's candidates, grouped by concept."""
        from aatm.llm.decisions import Option, Question

        options = [
            Option(value=f"c{i + 1}", description=_candidate_description(group))
            for i, group in enumerate(candidates)
        ]
        options.append(Option(value=NONE_OPTION, description="None of the candidates"))
        return Question.choice("concept", self.instructions, options)

    def candidates(self, expressions: List[Any]) -> List[List[Tuple[int, Any]]]:
        """The top-k distinct concepts, each with its ``(index, expression)`` pairs."""
        groups: Dict[str, List[Tuple[int, Any]]] = {}
        for index, expression in enumerate(expressions):
            key = expression.std_concept_id or f"expression:{index}"
            if key not in groups:
                if len(groups) == self.top_k:
                    continue
                groups[key] = []
            groups[key].append((index, expression))
        return list(groups.values())

    async def _select_one(
        self, query: str, expressions: List[Any], semaphore: Any
    ) -> SelectedExpressionMetadata | EmptySelectionMetadata:
        import json

        from aatm.llm.decisions import DecisionError

        candidates = self.candidates(expressions)
        if not candidates:
            return EmptySelectionMetadata()
        question = self.question([[e for _, e in group] for group in candidates])
        try:
            async with semaphore:
                decision = await self.decision_model.decide({"text": query}, [question])
            answer = decision[question.name]
        except (DecisionError, KeyError) as e:
            logger.warning("No decision for %r: %s", query, e)
            return EmptySelectionMetadata(rationale=f"No decision: {e}")
        trace = json.dumps(
            [
                {
                    "model": decision.model,
                    "answer": answer.value,
                    "probabilities": answer.probabilities,
                    "refused": answer.refused,
                }
            ],
            ensure_ascii=False,
        )
        if answer.refused or answer.value not in question.values[:-1]:
            return EmptySelectionMetadata(trace=trace)
        position = question.values.index(str(answer.value))
        index, expression = candidates[position][0]
        probability = answer.probability
        data = expression.model_dump(exclude={"distance", "expression_id"})
        return SelectedExpressionMetadata(
            **data,
            # The mapper's confidence_score is 1 - distance: the probability.
            distance=1 - probability if probability is not None else None,
            result_list_index=index,
            trace=trace,
        )

    def select(self, results: RetrieverResults) -> SelectorResults:
        """Select a candidate for each query, or none."""
        import asyncio

        from aatm.concept_search.agent import run_sync

        async def select_all() -> List[Any]:
            semaphore = asyncio.Semaphore(self.max_concurrency)
            return await asyncio.gather(
                *(
                    self._select_one(query, list(expressions), semaphore)
                    for query, expressions in zip(results.queries, results.results)
                )
            )

        return SelectorResults(results=run_sync(select_all()), queries=results.queries)
