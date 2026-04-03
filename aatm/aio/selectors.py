import asyncio
from typing import List
import dotenv
from openai import AsyncOpenAI
from google.genai import types

from aatm.data_models import (
    EmptySelectionMetadata,
    RetrieverResults,
    SelectedExpressionMetadata,
    SelectedResult,
    SelectorResults,
)
from aatm.prompt_helpers import format_prompt
from aatm.selectors import OpenAILLMSelector, GeminiLLMSelector
from aatm.debug import DebugMode, get_debug_mode
from aatm.logs import get_logger

# Load environment variables
dotenv.load_dotenv()

logger = get_logger(__name__)
debug_mode = get_debug_mode()


class AsyncOpenAILLMSelector(OpenAILLMSelector):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.client = AsyncOpenAI()

    async def select(self, results: RetrieverResults) -> SelectorResults:
        selector_results = SelectorResults(results=[], queries=results.queries)
        async with asyncio.TaskGroup() as tg:
            tasks = []
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

                tasks.append(
                    tg.create_task(
                        self.client.responses.parse(
                            model=self.model_id,
                            input=prompt,
                            text_format=SelectedResult,
                        )
                    )
                )

        responses = [t.result() for t in tasks]

        for response, (query_id, query) in zip(responses, enumerate(results.queries)):
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


class AsyncGeminiLLMSelector(GeminiLLMSelector):
    async def select(self, results: RetrieverResults) -> SelectorResults:
        selector_results = SelectorResults(results=[], queries=results.queries)
        async with asyncio.TaskGroup() as tg:
            tasks = []
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

                if debug_mode == DebugMode.GEMINI_LLM_SELECTOR:
                    logger.debug(prompt)
                    logger.debug(gemini_formatted_prompt)

                tasks.append(
                    tg.create_task(
                        self.client.aio.models.generate_content(
                            model=self.model_id,
                            contents=gemini_formatted_prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=SelectedResult,
                            ),
                        )
                    )
                )

        responses: List[types.GenerateContentResponse] = [t.result() for t in tasks]
        for response, (query_id, query) in zip(responses, enumerate(results.queries)):
            selected_result: SelectedResult = SelectedResult.model_validate_json(
                response.text
            )

            if debug_mode == DebugMode.GEMINI_LLM_SELECTOR:
                logger.debug(response)
                logger.debug(response.text)
                logger.debug(SelectedResult.model_validate_json(response.text))

            assert isinstance(selected_result, SelectedResult), (
                f"Expected SelectedResult object from Gemini, but got {type(selected_result)}"
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
