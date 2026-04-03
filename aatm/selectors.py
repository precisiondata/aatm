from enum import Enum
from typing import Dict, List
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
    @abstractmethod
    def select(self, results: RetrieverResults) -> SelectorResults:
        pass

    def __call__(self, results: RetrieverResults) -> SelectorResults:
        return self.select(results)


class FirstResultSelector(BaseSelector):
    def __init__(self, selection_threshold: float = float("inf"), *args, **kwargs):
        self.selection_threshold = selection_threshold

    def select(self, results: RetrieverResults) -> SelectorResults:
        selected_results = []
        for idx, r in enumerate(results.results):
            try:
                if r[0].distance < self.selection_threshold:
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
    GPT_52 = "gpt-5.2"
    GPT_5 = "gpt-5"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_MINI = "gpt-5-mini"


class OpenAILLMSelector(BaseSelector):
    def __init__(
        self,
        model_id: str,
        prompt_template: List[Dict[str, str]] = None,
        *args,
        **kwargs,
    ):
        self.model_id = OpenAILLMModels(model_id).value
        self.prompt_template = prompt_template
        self.client = OpenAI()

        if self.prompt_template is None:
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

    def select(self, results: RetrieverResults) -> SelectorResults:
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
                input=prompt,
                text_format=SelectedResult,
            )

            if debug_mode == DebugMode.OPENAI_LLM_SELECTOR:
                logger.debug(response)
                logger.debug(response.output_parsed)

            selected_result: SelectedResult = response.output_parsed
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
    GEMINI_3_PRO_PREVIEW = "gemini-3-pro-preview"
    GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_5_PRO = "gemini-2.5-pro"


class GeminiLLMSelector(BaseSelector):
    def __init__(
        self,
        model_id: str,
        prompt_template: List[Dict[str, str]] = None,
        *args,
        **kwargs,
    ):
        self.model_id = GeminiLLMModels(model_id).value
        self.client = genai.Client()
        self.prompt_template = prompt_template

        if self.prompt_template is None:
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

    def select(self, results: RetrieverResults) -> SelectorResults:
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

            gemini_formatted_prompt = []
            for msg in prompt:
                gemini_formatted_prompt.append(types.UserContent(msg["content"]))

            if debug_mode == DebugMode.OPENAI_LLM_SELECTOR:
                logger.debug(prompt)

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=gemini_formatted_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SelectedResult,
                ),
            )

            if debug_mode == DebugMode.GEMINI_LLM_SELECTOR:
                logger.debug(response)
                logger.debug(response.text)
                logger.debug(SelectedResult.model_validate_json(response.text))

            selected_result: SelectedResult = SelectedResult.model_validate_json(
                response.text
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


SELECTOR_REGISTRY = {
    "first": FirstResultSelector,
    "gpt-5.2": OpenAILLMSelector,
    GeminiLLMModels.GEMINI_3_PRO_PREVIEW.value: GeminiLLMSelector,
    GeminiLLMModels.GEMINI_3_FLASH_PREVIEW.value: GeminiLLMSelector,
    GeminiLLMModels.GEMINI_2_5_FLASH.value: GeminiLLMSelector,
    GeminiLLMModels.GEMINI_2_5_FLASH_LITE.value: GeminiLLMSelector,
    GeminiLLMModels.GEMINI_2_5_PRO.value: GeminiLLMSelector,
}


def load_selector(name: str, *args, **kwargs):
    return SELECTOR_REGISTRY[name](name, *args, **kwargs)
