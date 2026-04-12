"""
Define translator abstractions and concrete translation implementations.

This module provides pipeline-compatible translator components used to convert
source text into English before downstream terminology mapping steps. It
includes a base translator interface, a pass-through translator, and concrete
implementations backed by Gemini and OpenAI models.

Translators accept raw strings or source concept objects and return structured
``Translation`` objects so they can be composed with retrieval, reranking, and
selection stages in the mapping pipeline.
"""

import os
import json
import google.genai as genai
from google.genai import types
import dotenv
from abc import ABC, abstractmethod
from typing import Any, List

from openai import OpenAI

# Custom modules
from aatm.data_models import SourceConcept, Translation
from aatm.pipeline import PipelineBaseClass
from aatm.prompt_helpers import format_prompt
from aatm.selectors import OpenAILLMModels

# Load environment variables
dotenv.load_dotenv()


class BaseTranslator(PipelineBaseClass, ABC):
    """Define the abstract interface for translation pipeline components.

    This base class establishes the contract for translators that transform one
    or more input texts into structured ``Translation`` objects. It also
    provides a flexible ``__call__()`` implementation that normalizes supported
    input types before delegating to ``translate()``.
    """

    @abstractmethod
    def translate(self, texts: List[str]) -> Translation:
        """Translate one or more input texts.

        Subclasses must implement this method to perform the actual translation
        step and return structured translation outputs.

        Args:
            texts: A list of input strings to translate.

        Returns:
            One or more translation results corresponding to the input texts.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        pass

    def __call__(
        self, text: str | List[str] | List[SourceConcept]
    ) -> List[Translation]:
        """Normalize input values and translate them.

        This method allows translator instances to be called directly with a
        single string, a list of strings, or a list of ``SourceConcept``
        objects. Supported inputs are converted into a list of strings before
        being passed to ``translate()``.

        Args:
            text: Input text or texts to translate. Supported values are a
                single string, a list of strings, or a list of
                ``SourceConcept`` objects.

        Returns:
            A list of ``Translation`` objects corresponding to the normalized
            input texts.

        Raises:
            AssertionError: If the input is not one of the supported formats or
                does not contain valid strings.
        """
        if isinstance(text, str):
            text = [text]

        if isinstance(text, list) and isinstance(text[0], SourceConcept):
            text = [t.source_code_description for t in text]

        assert isinstance(text, list) and isinstance(text[0], str), (
            f"text must be a string, a list of strings, or a list of SourceConcept objects. Got {text}."
        )

        return self.translate(text)


class EmptyTranslator(BaseTranslator):
    """Return the original texts without performing translation.

    This translator acts as a no-op component by wrapping each input string in a
    ``Translation`` object unchanged. It is useful when translation is not
    needed but a translator stage is still required by the pipeline.
    """

    def translate(self, texts: List[str]) -> List[Translation]:
        """Wrap input texts as unchanged translation results.

        Args:
            texts: A list of input strings.

        Returns:
            A list of ``Translation`` objects whose text content matches the
            original input strings.
        """
        return [Translation(text=t) for t in texts]


class GeminiTranslator(BaseTranslator):
    """Translate texts using a Gemini model.

    This translator sends each input text to a Gemini model with a prompt that
    requests translation into English and expects a structured JSON response
    matching the ``Translation`` schema. It retries failed requests up to a
    configurable limit and falls back to the original text when all retries
    fail.
    """

    def __init__(
        self,
        model: str,
        prompt_template: str = None,
        n_retries: int = 3,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the Gemini-based translator.

        This constructor creates a Gemini client, stores the selected model
        name, configures the prompt template used for translation, and defines
        the retry behavior for failed requests.

        Args:
            model: Identifier of the Gemini model to use for translation.
            prompt_template: Optional prompt template containing a ``{text}``
                placeholder. If not provided, a default translation prompt is
                used.
            n_retries: Maximum number of attempts for each text before falling
                back to the original input.
            *args: Additional positional arguments reserved for compatibility.
            **kwargs: Additional keyword arguments reserved for compatibility.

        Returns:
            None.
        """
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = model
        self.prompt_template = prompt_template
        self.n_retries = n_retries

        if self.prompt_template is None:
            self.prompt_template = 'Translate the following text into English: "{text}"'

    def translate(self, texts: List[str]) -> List[Translation]:
        """Translate a list of texts into English using Gemini.

        This method processes each input text individually, sending it to the
        configured Gemini model and validating the structured JSON response as a
        ``Translation`` object. If a request fails, it is retried up to the
        configured limit. After all retries are exhausted, the original text is
        returned as a fallback translation.

        Args:
            texts: A list of input strings to translate.

        Returns:
            A list of ``Translation`` objects corresponding to the input texts.
        """
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
    """Translate texts using an OpenAI model.

    This translator formats each input text with a prompt template, sends it to
    an OpenAI model using structured response parsing, and returns the parsed
    ``Translation`` objects. Failed requests are retried up to a configurable
    limit, with fallback to the original text when all attempts fail.
    """

    def __init__(
        self,
        model: str,
        prompt_template: str = None,
        n_retries: int = 3,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI-based translator.

        This constructor resolves the configured model identifier, creates an
        OpenAI client, stores the prompt template used for translation, and
        defines the retry behavior for failed requests.

        Args:
            model: Identifier of the supported OpenAI model variant to use for
                translation.
            prompt_template: Optional structured prompt template. If not
                provided, a default user prompt requesting translation into
                English is used.
            n_retries: Maximum number of attempts for each text before falling
                back to the original input.
            *args: Additional positional arguments reserved for compatibility.
            **kwargs: Additional keyword arguments reserved for compatibility.

        Returns:
            None.

        Raises:
            ValueError: If ``model`` is not a valid member of
                ``OpenAILLMModels``.
        """
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
        """Translate a list of texts into English using OpenAI.

        This method processes each input text individually, formats the prompt
        with the given text, sends it to the configured OpenAI model using
        structured parsing, and returns the parsed ``Translation`` objects. If
        a request fails, it is retried up to the configured limit. After all
        retries are exhausted, the original text is returned as a fallback
        translation.

        Args:
            texts: A list of input strings to translate.

        Returns:
            A list of ``Translation`` objects corresponding to the input texts.
        """
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
