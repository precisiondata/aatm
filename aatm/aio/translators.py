import asyncio
import json
from google.genai import types
from typing import List

# Custom modules
from aatm.data_models import Translation
from aatm.translators import GeminiTranslator


class AsyncGeminiTranslator(GeminiTranslator):
    async def translate(self, texts: List[str]) -> List[Translation]:
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
                processed_results.append(Translation(**json.loads(result.text)))
            except Exception as e:
                print(
                    f"Error while processing text '{t}' and response '{result}': {e}. Original text was maintained."
                )
                processed_results.append(Translation(text=t))
        return processed_results
