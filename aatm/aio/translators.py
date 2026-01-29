import asyncio
import json
from google.genai import types
from typing import List

# Custom modules
from aatm.data_models import Translation
from aatm.translators import GeminiTranslator


class AsyncGeminiTranslator(GeminiTranslator):
    async def __call__(self, text: str | List[str]) -> List[Translation]:
        if isinstance(text, str):
            text = [text]

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
                for t in text
            ]

        results = [t.result() for t in tasks]
        results = [json.loads(r.text) for r in results]
        return [Translation(**r) for r in results]
