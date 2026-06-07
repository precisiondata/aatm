import asyncio
import json
from google.genai import types
from typing import List

from openai import AsyncOpenAI

# Custom modules
from aatm.data_models import Translation
from aatm.prompt_helpers import format_prompt
from aatm.translators import BaseTranslator, GeminiTranslator, OpenAITranslator


class EmptyTranslator(BaseTranslator):
    async def translate(self, texts: List[str]) -> List[Translation]:  # type: ignore[override]
        return [Translation(text=t) for t in texts]


class AsyncGeminiTranslator(GeminiTranslator):
    async def translate(self, texts: List[str]) -> List[Translation]:  # type: ignore[override]
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(
                    self.client.aio.models.generate_content(
                        model=self.model,
                        contents=self.prompt_template.format(text=t),
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=Translation,
                        ),
                    )
                )
                for t in texts
            ]

        results = [t.result() for t in tasks]

        processed_results = []
        for result, t in zip(results, texts):
            try:
                if result.text is None:
                    raise ValueError("Gemini API returned null response.")
                processed_results.append(Translation(**json.loads(result.text)))
            except Exception as e:
                print(
                    f"Error while processing text '{t}' and response '{result}': {e}. Original text was maintained."
                )
                processed_results.append(Translation(text=t))
        return processed_results


class AsyncOpenAITranslator(OpenAITranslator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = AsyncOpenAI()

    async def translate(self, texts):
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(
                    self.client.responses.parse(
                        model=self.model_id,
                        input=format_prompt(self.prompt_template, {"text": t}),
                        text_format=Translation,
                    )
                )
                for t in texts
            ]

        results = [t.result() for t in tasks]

        processed_results = []
        for result, t in zip(results, texts):
            try:
                processed_results.append(result.output_parsed)
            except Exception as e:
                print(
                    f"Error while processing text '{t}' and response '{result}': {e}. Original text was maintained."
                )
                processed_results.append(Translation(text=t))
        return processed_results
