"""Extraction components for concept extraction pipelines.

This module defines the abstract interface for extractors used in the AATM
pipeline and provides a concrete implementation based on Google's Gemini
models. Extractors are responsible for receiving one or more raw texts and
returning structured concept annotations as `ExtractedConcept` objects grouped
per input text.

The module currently includes:

- `BaseExtractor`: abstract base class for extractor implementations.
- `GeminiExtractor`: extractor that prompts a Gemini model to return concept
  annotations in a structured JSON format validated against AATM data models.

The Gemini-based implementation optionally filters extracted concepts by a
predefined label set and enriches each extracted concept with character-level
start and end offsets computed from the source text.
"""

from typing import List, Optional
import dotenv
from abc import ABC, abstractmethod
from google import genai

from aatm.data_models import ExtractedConcept, ListOfExtractedConcepts
from aatm.pipeline import PipelineBaseClass

dotenv.load_dotenv()


class BaseExtractor(PipelineBaseClass, ABC):
    """Abstract base class for concept extractors.

    This class defines the common interface for extraction components in the AATM
    pipeline. Concrete subclasses must implement the `extract` method, which takes
    a batch of input texts and returns extracted concepts for each text.

    As a pipeline component, this class also exposes a callable interface through
    `__call__`, allowing extractor instances to be invoked directly on a list of
    strings.
    """

    @abstractmethod
    def extract(
        self, texts: List[str]
    ) -> List[List[ExtractedConcept]] | List[ListOfExtractedConcepts]:
        """Extract concepts from a batch of texts.

        Args:
            texts: A list of input texts to be processed.

        Returns:
            A list with one extraction result per input text. Each result may be
                represented either as a list of `ExtractedConcept` objects or as a
                `ListOfExtractedConcepts` instance, depending on the implementation.

        Raises:
            NotImplementedError: Raised by subclasses that do not implement this
                abstract method.
        """
        pass

    def __call__(
        self, texts: List[str]
    ) -> List[List[ExtractedConcept]] | List[ListOfExtractedConcepts]:
        """Run extraction on a batch of texts.

        This method validates that the input is a list whose first element is a string
        and then delegates execution to `extract`.

        Args:
            texts: A list of input texts.

        Returns:
            The extraction results returned by `extract`.

        Raises:
            AssertionError: If the input is not a list of strings.
        """

        assert isinstance(texts, list) and isinstance(texts[0], str), (
            "Input must be a list of strings."
        )
        return self.extract(texts)


class GeminiExtractor(BaseExtractor):
    """Extractor implementation backed by Gemini generative models.

    This extractor formats each input text into a prompt template, sends the prompt
    to a Gemini model configured to return JSON, validates the model response
    against the `ListOfExtractedConcepts` schema, and post-processes the extracted
    concepts.

    Post-processing includes:

    - computing character-level start and end offsets for each extracted concept
      based on its first occurrence in the original text;
    - discarding concepts not found in the original text;
    - optionally filtering concepts to a predefined set of allowed labels.

    This class is useful for zero-shot or few-shot structured extraction workflows
    where the model is expected to emit JSON matching the package's extraction
    schema.
    """

    def __init__(
        self,
        model_id: str,
        prompt_template: str,
        api_key: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> None:
        """Initialize a Gemini-based extractor.

        Args:
            model_id: Identifier of the Gemini model to be used for extraction.
            prompt_template: Prompt template used to instruct the model. The template
                must contain a `{text}` placeholder where the input text will be
                inserted.
            api_key: Optional Google API key used to initialize the Gemini client.
                If omitted, authentication may rely on environment configuration.
            labels: Optional list of allowed labels. When provided, extracted concepts
                whose labels are not in this list are discarded.

        Raises:
            AssertionError: If the prompt template does not contain the `{text}`
                placeholder.
        """

        self.model_id = model_id
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.prompt_template = prompt_template
        self.labels = labels

        assert "{text}" in self.prompt_template, (
            "Prompt template must contain a {text} placeholder."
        )

    def _get_start_and_end_indices(self, text: str, prompt: str) -> List[int]:
        """Compute character offsets for an extracted span in the source text.

        This method finds the first occurrence of the extracted text in the original
        input text and returns its start and end indices.

        Args:
            text: The original source text.
            prompt: The extracted text span whose position should be located in the
                source text.

        Returns:
            A two-element list containing the start and end character offsets. If the
                span is not found, the start index will be `-1`, and the end index will be
                computed relative to that value.
        """
        start_index = text.find(prompt)
        end_index = start_index + len(prompt)
        return [start_index, end_index]

    def extract(self, texts: List[str]) -> List[ListOfExtractedConcepts]:
        """Extract structured concepts from a batch of texts using Gemini.

        For each input text, this method formats the configured prompt template,
        requests structured JSON output from the Gemini model, validates the response
        as `ListOfExtractedConcepts`, and post-processes the extracted concepts by
        assigning character offsets and applying optional label filtering.

        Args:
            texts: A list of raw texts to process.

        Returns:
            A list of `ListOfExtractedConcepts` objects, with one validated extraction
                result per input text.

        Notes:
            Character offsets are computed using the first occurrence of each extracted
                concept text in the original input text. If the same extracted span appears
                multiple times in the source text, this method does not disambiguate among
                occurrences.
        """

        results = []
        for text in texts:
            prompt = self.prompt_template.format(text=text)
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": ListOfExtractedConcepts.model_json_schema(),
                },
            )
            result = ListOfExtractedConcepts.model_validate_json(response.text)

            # Add start and end indices
            for concept in result.extracted_concepts:
                start_end_indices = self._get_start_and_end_indices(text, concept.text)
                concept.start = start_end_indices[0]
                concept.end = start_end_indices[1]

            # Keep only extracted concept that are present in the original text
            result.extracted_concepts = [
                concept for concept in result.extracted_concepts if concept.start != -1
            ]

            # Keep only valid labels
            if self.labels:
                result.extracted_concepts = [
                    concept
                    for concept in result.extracted_concepts
                    if concept.label in self.labels
                ]

            results.append(result)

        return results
