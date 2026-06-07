"""
Define selector abstractions and concrete selection strategies for retrieval results.

This module provides selector components used in the terminology-mapping
pipeline to choose the best candidate from a set of retrieved expressions. It
includes a simple rule-based selector that picks the first result under a
distance threshold, as well as LLM-based selectors backed by OpenAI and Gemini
models.

Selectors are designed to operate as pipeline stages that consume
``RetrieverResults`` and return structured ``SelectorResults`` objects.
"""

from enum import Enum
from typing import Optional, Any, Dict, List
import dotenv
from abc import ABC, abstractmethod
from google import genai
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
        self.client = genai.Client()

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
