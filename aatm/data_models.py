"""Data models and enums for terminology mapping workflows.

This module defines the core structured data models used throughout the
AATM package to represent source concepts, retrieved expressions,
selected mapping results, translations, and task configurations. It also
provides supporting enumerations for expression provenance and standard
vocabularies, along with utility methods for loading, validating,
serializing, and transforming these objects.

The models are primarily implemented with Pydantic and are designed to
support terminology mapping pipelines that involve retrieval, selection,
reranking, and export of standardized concepts.

Notes:
    The models in this module include validation helpers for coercing
    concept identifiers to strings, validating date fields, loading
    records from CSV files, reading task configurations from JSON or
    YAML, and serializing results back to plain dictionaries or config
    files.
"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from enum import Enum
import pandas as pd
import yaml
import itertools


def deterministic_id_from_strings(strings: list[str], digest_size: int = 16) -> str:
    """
    Generate a deterministic id from a list of strings.

    Args:
        strings (list[str]): A list of strings to generate the id from.
        digest_size (int, optional): The size of the digest in bytes.

    Returns:
        str: A deterministic id as a hexadecimal string.
    """
    joined = "||".join(strings)
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=digest_size).hexdigest()


class ExpressionOrigin(Enum):
    """Enumerate the possible origins of an expression in the mapping pipeline.

    This enum identifies whether an expression comes directly from a
    standard concept, from a synonym of a standard concept, from a
    mapped non-standard concept, or from a synonym of a mapped
    non-standard concept.
    """

    STANDARD_CONCEPT = "standard_concept"
    STANDARD_CONCEPT_SYNONYM = "standard_concept_synonym"
    NON_STANDARD_CONCEPT = "mapped_non_standard_concept"
    NON_STANDARD_CONCEPT_SYNONYM = "mapped_non_standard_concept_synonym"


class StandardVocabulary(Enum):
    """Enumerate the supported standard vocabularies.

    This enum defines the target standardized vocabularies currently
    supported by the terminology mapping workflow.
    """

    SNOMED = "SNOMED"
    RXNORM = "RxNorm"
    LOINC = "LOINC"


class ExpressionMetadata(BaseModel):
    """Represent metadata for a terminology expression.

    This model stores information about an expression, its originating
    concept, and its associated standard concept metadata. A
    deterministic `expression_id` is generated after initialization from
    the main identifying fields.

    Attributes:
        expression_id: Deterministic identifier generated from the
            expression metadata.
        expression: Original text of the expression.
        expression_concept_id: Identifier of the source concept
            associated with the expression.
        expression_origin: Origin of the expression in relation to
            standard or non-standard concepts.
        std_concept_id: Identifier of the mapped standard concept.
        std_concept_name: Name of the mapped standard concept.
        std_vocabulary_id: Standard vocabulary to which the mapped
            concept belongs.
        std_vocabulary_code: Code of the mapped concept in the standard
            vocabulary.
        std_domain_id: Domain associated with the mapped standard
            concept.
    """

    expression_id: Optional[str] = None
    expression: Optional[str]
    expression_concept_id: Optional[str]
    expression_origin: Optional[ExpressionOrigin]
    std_concept_id: Optional[str]
    std_concept_name: Optional[str]
    std_vocabulary_id: Optional[StandardVocabulary]
    std_vocabulary_code: Optional[str]
    std_domain_id: Optional[str]

    @field_validator(
        "expression_concept_id", "std_concept_id", "std_vocabulary_code", mode="before"
    )
    @classmethod
    def validate_concept_id(cls, value: Any) -> str:
        """Convert concept-related values to strings before validation.

        This validator normalizes concept identifier fields by coercing the
        incoming value to `str`. It is applied before standard Pydantic
        validation for selected identifier fields.

        Args:
            value: Raw value provided for a concept-related field.

        Returns:
            The normalized string representation of the input value.
        """
        return str(value)

    def model_post_init(self, *args: Any, **kwargs: Any) -> None:
        """Generate a deterministic expression identifier after initialization.

        This hook runs after model initialization and populates
        `expression_id` using a deterministic hash derived from the main
        expression metadata fields.

        Args:
            *args: Positional arguments passed by Pydantic's post-init
                lifecycle.
            **kwargs: Keyword arguments passed by Pydantic's post-init
                lifecycle.

        Returns:
            None
        """
        self.expression_id = deterministic_id_from_strings(
            [
                self.expression,
                self.expression_concept_id,
                self.expression_origin.value if self.expression_origin else None,
                self.std_concept_id,
                self.std_concept_name,
                self.std_vocabulary_id.value if self.std_vocabulary_id else None,
                self.std_vocabulary_code,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary with enum values serialized.

        This method returns the model data as a plain dictionary and
        replaces enum instances in `expression_origin` and
        `std_vocabulary_id` with their raw `.value` representations.

        Returns:
            A dictionary representation of the model with enum fields
            serialized as plain values.
        """
        model_dict = self.model_dump()
        model_dict["expression_origin"] = model_dict["expression_origin"].value
        model_dict["std_vocabulary_id"] = model_dict["std_vocabulary_id"].value
        return model_dict


class Translation(BaseModel):
    """Represent a translated text value.

    Attributes:
        text: Translated text string.
    """

    text: str


class SourceConcept(BaseModel):
    """Represent a source concept to be mapped.

    This model stores the original source terminology fields and related
    validity metadata. Extra fields are ignored to allow loading from
    broader tabular inputs.

    Attributes:
        source_code: Code of the source concept.
        source_concept_id: Identifier of the source concept.
        source_vocabulary_id: Vocabulary identifier of the source
            concept.
        source_code_description: Human-readable description of the
            source concept.
        valid_start_date: Start date of validity in `YYYY-MM-DD` format.
        valid_end_date: End date of validity in `YYYY-MM-DD` format.
        invalid_reason: Reason why the source concept is invalid, when
            applicable.
    """

    # ignore extra fields
    model_config = ConfigDict(extra="ignore")

    # fields
    source_code: Optional[str] = Field(None, examples=["I63"])
    source_concept_id: Optional[str] = Field(None, examples=["45543186"])
    source_vocabulary_id: Optional[str] = Field(None, examples=["ICD10"])
    source_code_description: Optional[str] = Field(
        None, examples=["Cerebral infarction"]
    )
    valid_start_date: Optional[str] = Field(None, examples=["1990-05-01"])
    valid_end_date: Optional[str] = Field(None, examples=["2099-12-31"])
    invalid_reason: Optional[str] = Field(None, examples=[None])

    @field_validator(
        "source_code", "source_concept_id", "source_vocabulary_id", mode="before"
    )
    @classmethod
    def validate_strings(cls, value: Any) -> str:
        """Convert source identifier fields to strings before validation.

        This validator coerces selected source concept fields to `str`
        before standard Pydantic validation is applied.

        Args:
            value: Raw value provided for a source identifier field.

        Returns:
            The normalized string representation of the input value.
        """
        return str(value) if value is not None else value

    @field_validator("valid_start_date", "valid_end_date", mode="before")
    @classmethod
    def validate_yyyy_mm_dd(cls, v: str) -> str:
        """Validate that a date string follows the `YYYY-MM-DD` format.

        This validator accepts empty strings unchanged and otherwise checks
        that the provided value is a string matching the expected date
        format.

        Args:
            v: Raw date value to validate.

        Returns:
            The validated date string.

        Raises:
            TypeError: If the provided value is not a string.
            ValueError: If the string is not empty and does not match the
                `YYYY-MM-DD` format.
        """
        if v is None:
            return v

        if not isinstance(v, str):
            raise TypeError("date must be a string in YYYY-MM-DD format")

        elif v == "":
            return v

        try:
            # Strict format check
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date must be in YYYY-MM-DD format")

        return v

    @classmethod
    def from_csv(cls, path: str | Path) -> List["SourceConcept"]:
        """Load source concepts from a CSV file.

        This class method reads a CSV file into a pandas DataFrame, replaces
        missing values with empty strings, and constructs one
        `SourceConcept` instance per row.

        Args:
            path: Path to the CSV file containing source concept records.

        Returns:
            A list of `SourceConcept` instances loaded from the CSV file.
        """
        if isinstance(path, str):
            path = Path(path)
        df = pd.read_csv(path).fillna("")

        return [cls(**row) for row in df.to_dict("records")]


class MappedSourceConcept(SourceConcept):
    """Represent a source concept together with its mapped target concept.

    This model extends `SourceConcept` by including the selected target
    standardized concept and related vocabulary metadata.

    Attributes:
        target_concept_id: Identifier of the mapped target concept.
        target_vocabulary_id: Standard vocabulary of the mapped target
            concept.
        target_vocabulary_code: Code of the mapped target concept in the
            target vocabulary.
        domain_id: Domain of the mapped target concept.
        confidence_score: Confidence score of the mapping.
        source_code_description_original: Source code description before translation.
    """

    target_concept_id: Optional[str]
    target_vocabulary_id: Optional[StandardVocabulary]
    target_vocabulary_code: Optional[str]
    domain_id: Optional[str]
    confidence_score: Optional[float]
    source_code_description_original: Optional[str]

    @field_validator(
        "source_code",
        "source_concept_id",
        "source_vocabulary_id",
        "target_concept_id",
        "target_vocabulary_id",
        mode="before",
    )
    @classmethod
    def validate_strings(cls, value: Any) -> str:
        """Convert selected source and target fields to strings before validation.

        This validator coerces selected identifier and vocabulary fields to
        `str` when a value is provided. `None` values are preserved.

        Args:
            value: Raw value provided for a validated field.

        Returns:
            The normalized string representation of the input value, or
            `None` when the input value is `None`.
        """
        if value is not None:
            return str(value)

    @classmethod
    def from_csv(cls, path: str | Path) -> List["MappedSourceConcept"]:
        """Load mapped source concepts from a CSV file.

        This class method reads a CSV file into a pandas DataFrame, replaces
        missing values with empty strings, and constructs one
        `MappedSourceConcept` instance per row.

        Args:
            path: Path to the CSV file containing mapped source concept
                records.

        Returns:
            A list of `MappedSourceConcept` instances loaded from the CSV
            file.
        """
        if isinstance(path, str):
            path = Path(path)
        df = pd.read_csv(path).fillna("")

        return [cls(**row) for row in df.to_dict("records")]

    @classmethod
    def from_selector_results(
        cls,
        source_concepts: List[SourceConcept],
        results: "SelectorResults",
        translated_source_code_descriptions: Optional[List[Translation]] = None,
    ) -> List["MappedSourceConcept"]:
        """Build mapped source concepts from source concepts and selector results.

        This class method combines each `SourceConcept` with its
        corresponding selected result and produces a list of
        `MappedSourceConcept` instances containing both source and mapped
        target fields.

        Args:
            source_concepts: Source concepts to be combined with mapping
                selections.
            results: Selection results containing the chosen standardized
                concept for each source concept.
            translated_source_code_descriptions: Optional list of translations for the source code descriptions.

        Returns:
            A list of `MappedSourceConcept` instances built from the source
            concepts and selector results.
        """
        mapped_source_concepts = []
        for source_concept, selected_result, translation in itertools.zip_longest(
            source_concepts, results.results, translated_source_code_descriptions
        ):
            # Add type annotations
            source_concept: SourceConcept
            selected_result: SelectedExpressionMetadata
            translation: Optional[Translation]

            # Create MappedSourceConcept
            mapped_source_concepts.append(
                cls(
                    source_code=source_concept.source_code,
                    source_concept_id=source_concept.source_concept_id,
                    source_vocabulary_id=source_concept.source_vocabulary_id,
                    source_code_description=translation.text
                    if translation
                    else source_concept.source_code_description,
                    target_concept_id=selected_result.std_concept_id,
                    target_vocabulary_id=selected_result.std_vocabulary_id.value
                    if selected_result.std_vocabulary_id
                    else None,
                    domain_id=selected_result.std_domain_id,
                    valid_start_date=source_concept.valid_start_date,
                    valid_end_date=source_concept.valid_end_date,
                    invalid_reason=source_concept.invalid_reason,
                    target_vocabulary_code=selected_result.std_vocabulary_code,
                    confidence_score=1 - selected_result.distance,
                    source_code_description_original=source_concept.source_code_description
                    if translation
                    else None,
                )
            )

        return mapped_source_concepts

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary with enum values serialized.

        This method returns the model data as a plain dictionary and
        replaces `target_vocabulary_id` with its raw `.value`
        representation when present.

        Returns:
            A dictionary representation of the model with enum fields
            serialized as plain values.
        """
        model_dict = self.model_dump()
        model_dict["target_vocabulary_id"] = (
            model_dict["target_vocabulary_id"].value
            if model_dict["target_vocabulary_id"]
            else None
        )
        return model_dict


class RetrievedExpressionMetadata(ExpressionMetadata):
    """Represent retrieved expression metadata with ranking information.

    This model extends `ExpressionMetadata` with fields produced during
    retrieval and reranking steps. Extra fields are ignored to support
    flexible loading from retrieval outputs.

    Attributes:
        distance: Retrieval distance or similarity-derived distance for
            the expression.
        rerank_score: Score assigned during reranking.
    """

    # ignore extra fields
    model_config = ConfigDict(extra="ignore")

    # fields
    distance: Optional[float] = None
    rerank_score: Optional[float] = None

    def to_prompt_object(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        """Convert retrieved metadata to a prompt-friendly dictionary.

        This method returns a reduced dictionary representation containing
        the main standardized concept fields needed for prompt construction
        in downstream components.

        Args:
            *args: Additional positional arguments accepted for interface
                compatibility.
            **kwargs: Additional keyword arguments accepted for interface
                compatibility.

        Returns:
            A dictionary containing the expression identifier, expression
            text, standard concept name, standard vocabulary identifier,
            standard vocabulary code, and standard domain identifier.
        """
        return {
            "expression_id": self.expression_id,
            "expression": self.expression,
            "standard_concept_name": self.std_concept_name,
            "standard_vocabulary_id": self.std_vocabulary_id.value,
            "standard_vocabulary_code": self.std_vocabulary_code,
            "standard_domain_id": self.std_domain_id,
        }


class SelectedExpressionMetadata(RetrievedExpressionMetadata):
    """Represent the expression selected from a retrieval result list.

    This model extends `RetrievedExpressionMetadata` by recording the
    index of the selected item in the original candidate list.

    Attributes:
        result_list_index: Index of the selected result in the candidate
            list.
    """

    result_list_index: int


class EmptySelectionMetadata(BaseModel):
    """Represent an empty selection result.

    This placeholder model is used when no expression is selected. All
    fields are explicitly set to `None` to preserve the expected schema.

    Attributes:
        expression_id: Always `None`.
        expression: Always `None`.
        expression_concept_id: Always `None`.
        expression_origin: Always `None`.
        std_concept_id: Always `None`.
        std_concept_name: Always `None`.
        std_vocabulary_id: Always `None`.
        std_vocabulary_code: Always `None`.
        std_domain_id: Always `None`.
        result_list_index: Always `None`.
        distance: Always `None`.
    """

    expression_id: None = None
    expression: None = None
    expression_concept_id: None = None
    expression_origin: None = None
    std_concept_id: None = None
    std_concept_name: None = None
    std_vocabulary_id: None = None
    std_vocabulary_code: None = None
    std_domain_id: None = None
    result_list_index: None = None
    distance: None = None


class RetrieverResults(BaseModel):
    """Represent the output of a retrieval step.

    Attributes:
        results: Nested list of retrieved expression metadata grouped by
            query.
        queries: List of query strings used in retrieval.
    """

    results: List[List[RetrievedExpressionMetadata]]
    queries: List[str]


class SelectorResults(BaseModel):
    """Represent the output of a selection step.

    Attributes:
        results: List of selected expression metadata, one per query or
            source item.
        queries: List of query strings associated with the selections.
    """

    results: List[SelectedExpressionMetadata]
    queries: List[str]


class SelectedResult(BaseModel):
    """Represent a minimal selected result reference.

    Attributes:
        expression_id: Identifier of the selected expression.
    """

    expression_id: Optional[str] = None


class TerminologyMappingTask(BaseModel):
    """Represent the configuration for a terminology mapping task.

    This model defines the inputs and execution parameters required to
    run a terminology mapping workflow. It also provides convenience
    methods for loading task definitions from JSON or YAML files and for
    saving them back to disk.

    Attributes:
        input_file: Path to the input file containing source concepts.
        output_dir: Directory where mapping outputs will be written.
        translator_id: Identifier of the translator component to use.
        retriever_id: Identifier of the retriever component to use.
        selector_id: Identifier of the selector component to use.
        reranker_id: Identifier of the reranker component to use.
        batch_size: Batch size for processing source concepts.
        rate_limit: Rate limit applied during processing.
        column_mapping: Optional mapping describing how input columns
            correspond to expected fields.
        limit_to: Optional limit on the number of source concepts to
            process.
    """

    input_file: Path
    output_dir: Optional[Path] = None
    translator_id: Optional[str] = None
    retriever_id: Optional[str] = None
    selector_id: Optional[str] = None
    reranker_id: Optional[str] = None
    batch_size: Optional[int] = None
    rate_limit: Optional[int] = None
    column_mapping: Optional[dict] = None
    limit_to: Optional[int] = None

    @field_validator("input_file", "output_dir", mode="before")
    def validate_paths(cls, value: Any) -> Path:
        """Convert path-like input values to `Path` objects before validation.

        This validator normalizes the `input_file` and `output_dir` fields by
        converting incoming string values to `Path` instances. Existing
        `Path` objects are returned unchanged.

        Args:
            value: Raw value provided for a path field.

        Returns:
            A normalized `Path` instance.
        """
        if isinstance(value, Path):
            return value

        return Path(value)

    @classmethod
    def from_json(cls, path: str | Path) -> "TerminologyMappingTask":
        """Load a terminology mapping task from a JSON configuration file.

        Args:
            path: Path to a JSON file containing the task configuration.

        Returns:
            A `TerminologyMappingTask` instance created from the JSON file.
        """
        if isinstance(path, str):
            path = Path(path)
        return cls(**json.loads(path.read_text()))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TerminologyMappingTask":
        """Load a terminology mapping task from a YAML configuration file.

        Args:
            path: Path to a YAML file containing the task configuration.

        Returns:
            A `TerminologyMappingTask` instance created from the YAML file.
        """
        if isinstance(path, str):
            path = Path(path)
        return cls(**yaml.safe_load(path.read_text()))

    @classmethod
    def from_config_file(cls, path: str | Path) -> "TerminologyMappingTask":
        """Load a terminology mapping task from a supported config file.

        This method dispatches to `from_json` or `from_yaml` based on the
        file extension.

        Args:
            path: Path to a supported configuration file.

        Returns:
            A `TerminologyMappingTask` instance created from the provided            configuration file.

        Raises:
            ValueError: If the file extension is not supported.
        """
        if isinstance(path, str):
            path = Path(path)
        if path.suffix == ".json":
            return cls.from_json(path)
        elif path.suffix == ".yaml":
            return cls.from_yaml(path)
        else:
            raise ValueError(f"Unsupported config file format: '{path.suffix}'")

    def save_to_disk(self, path: str | Path) -> None:
        """Save the task configuration to a JSON or YAML file.

        This method serializes the current model and writes it to disk based
        on the output file extension.

        Args:
            path: Destination path for the serialized configuration file.

        Returns:
            None

        Raises:
            ValueError: If the file extension is not supported.
        """
        if isinstance(path, str):
            path = Path(path)
        if path.suffix == ".json":
            path.write_text(json.dumps(self.model_dump()))
        elif path.suffix == ".yaml":
            path.write_text(yaml.safe_dump(self.model_dump()))
        else:
            raise ValueError(f"Unsupported config file format: '{path.suffix}'")


class ExtractedConcept(BaseModel):
    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    label: str


class ListOfExtractedConcepts(BaseModel):
    extracted_concepts: List[ExtractedConcept]

    def __getitem__(
        self, index: int | slice
    ) -> ExtractedConcept | List[ExtractedConcept]:
        return self.extracted_concepts[index]
