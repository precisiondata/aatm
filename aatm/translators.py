import os
import json
import google.genai as genai
from google.genai import types
import dotenv
from abc import ABC, abstractmethod
from typing import List

# Custom modules
from aatm.data_models import SourceConcept, Translation
from aatm.pipeline import PipelineBaseClass

# Load environment variables
dotenv.load_dotenv()


class BaseTranslator(PipelineBaseClass, ABC):
    @abstractmethod
    def translate(self, texts: List[str]) -> Translation:
        pass

    def __call__(
        self, text: str | List[str] | List[SourceConcept]
    ) -> List[Translation]:
        if isinstance(text, str):
            text = [text]

        if isinstance(text, list) and isinstance(text[0], SourceConcept):
            text = [t.source_code_description for t in text]

        assert isinstance(text, list) and isinstance(text[0], str), (
            f"text must be a string, a list of strings, or a list of SourceConcept objects. Got {text}."
        )

        return self.translate(text)


class GeminiTranslator(BaseTranslator):
    def __init__(self, model: str, prompt_template: str = None, *args, **kwargs):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = model
        self.prompt_template = prompt_template

        if self.prompt_template is None:
            self.prompt_template = 'Translate the following text into English: "{text}"'

    def translate(self, texts: List[str]) -> List[Translation]:
        results = []
        for t in texts:
            response = self.client.models.generate_content(
                model=self.model,
                contents=self.prompt_template.format(text=t),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=Translation,
                ),
            )
            results.append(Translation(**json.loads(response.text)))
        return results
