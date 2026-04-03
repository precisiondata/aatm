import os
import json
import google.genai as genai
from google.genai import types
import dotenv
from abc import ABC, abstractmethod
from typing import List

from openai import OpenAI

# Custom modules
from aatm.data_models import SourceConcept, Translation
from aatm.pipeline import PipelineBaseClass
from aatm.prompt_helpers import format_prompt
from aatm.selectors import OpenAILLMModels

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


class EmptyTranslator(BaseTranslator):
    def translate(self, texts: List[str]) -> List[Translation]:
        return [Translation(text=t) for t in texts]


class GeminiTranslator(BaseTranslator):
    def __init__(
        self,
        model: str,
        prompt_template: str = None,
        n_retries: int = 3,
        *args,
        **kwargs,
    ):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = model
        self.prompt_template = prompt_template
        self.n_retries = n_retries

        if self.prompt_template is None:
            self.prompt_template = 'Translate the following text into English: "{text}"'

    def translate(self, texts: List[str]) -> List[Translation]:
        results = []
        for t in texts:
            n_retries = 0
            while n_retries < self.n_retries:
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=self.prompt_template.format(text=t),
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=Translation,
                        ),
                    )
                    results.append(Translation(**json.loads(response.text)))
                    break
                except Exception as e:
                    n_retries += 1
                    print(
                        f"Error while processing text '{t}' (type: {type(t)}): {e}. Retrying... ({n_retries}/{self.n_retries})"
                    )
                    print(f'Malformed response: "{response}"')
                    if n_retries == self.n_retries:
                        results.append(Translation(text=t))
        return results


class OpenAITranslator(BaseTranslator):
    def __init__(
        self,
        model: str,
        prompt_template: str = None,
        n_retries: int = 3,
        *args,
        **kwargs,
    ):
        self.model_id = OpenAILLMModels(model).value
        self.prompt_template = prompt_template
        self.client = OpenAI()
        self.n_retries = n_retries

        if self.prompt_template is None:
            self.prompt_template = [
                {
                    "role": "user",
                    "content": 'Translate the following text into English: "{text}"',
                }
            ]

    def translate(self, texts: List[str]) -> List[Translation]:
        results = []
        for t in texts:
            n_retries = 0
            while n_retries < self.n_retries:
                try:
                    response = self.client.responses.parse(
                        model=self.model_id,
                        input=format_prompt(self.prompt_template, {"text": t}),
                        text_format=Translation,
                    )
                    results.append(response.output_parsed)
                    break
                except Exception as e:
                    n_retries += 1
                    print(
                        f"Error while processing text '{t}' (type: {type(t)}): {e}. Retrying... ({n_retries}/{self.n_retries})"
                    )
                    print(f'Malformed response: "{response}"')
                    if n_retries == self.n_retries:
                        results.append(Translation(text=t))
        return results
