from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, field_validator
from enum import Enum
import pandas as pd


def deterministic_id_from_strings(strings: list[str], digest_size: int = 16) -> str:
    """
    Generate a deterministic id from a list of strings.

    Args:
        strings (list[str]): A list of strings to generate the id from.
        digest_size (int, optional): The size of the digest in bytes. Defaults to 8.

    Returns:
        str: A deterministic id as a hexadecimal string.
    """
    joined = "||".join(strings)
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=digest_size).hexdigest()


class ExpressionOrigin(Enum):
    STANDARD_CONCEPT = "standard_concept"
    STANDARD_CONCEPT_SYNONYM = "standard_concept_synonym"
    NON_STANDARD_CONCEPT = "mapped_non_standard_concept"
    NON_STANDARD_CONCEPT_SYNONYM = "mapped_non_standard_concept_synonym"


class StandardVocabulary(Enum):
    SNOMED = "SNOMED"
    RXNORM = "RxNorm"
    LOINC = "LOINC"


class ExpressionMetadata(BaseModel):
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
        return str(value)

    def model_post_init(self, *args, **kwargs):
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

    def to_dict(self):
        model_dict = self.model_dump()
        model_dict["expression_origin"] = model_dict["expression_origin"].value
        model_dict["std_vocabulary_id"] = model_dict["std_vocabulary_id"].value
        return model_dict


class Translation(BaseModel):
    text: str


class SourceConcept(BaseModel):
    # ignore extra fields
    model_config = ConfigDict(extra="ignore")

    # fields
    source_code: str
    source_concept_id: str
    source_vocabulary_id: str
    source_code_description: str
    valid_start_date: str
    valid_end_date: str
    invalid_reason: str

    @field_validator(
        "source_code", "source_concept_id", "source_vocabulary_id", mode="before"
    )
    @classmethod
    def validate_strings(cls, value: Any) -> str:
        return str(value)

    @field_validator("valid_start_date", "valid_end_date", mode="before")
    @classmethod
    def validate_yyyy_mm_dd(cls, v: str) -> str:
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
        if isinstance(path, str):
            path = Path(path)
        df = pd.read_csv(path).fillna("")

        return [cls(**row) for row in df.to_dict("records")]


class MappedSourceConcept(SourceConcept):
    target_concept_id: Optional[str]
    target_vocabulary_id: Optional[StandardVocabulary]
    target_vocabulary_code: Optional[str]
    domain_id: Optional[str]

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
        return str(value)

    @classmethod
    def from_csv(cls, path: str | Path) -> List["MappedSourceConcept"]:
        if isinstance(path, str):
            path = Path(path)
        df = pd.read_csv(path).fillna("")

        return [cls(**row) for row in df.to_dict("records")]

    @classmethod
    def from_selector_results(
        cls, source_concepts: List[SourceConcept], results: "SelectorResults"
    ):
        mapped_source_concepts = []
        for source_concept, selected_result in zip(source_concepts, results.results):
            mapped_source_concepts.append(
                cls(
                    source_code=source_concept.source_code,
                    source_concept_id=source_concept.source_concept_id,
                    source_vocabulary_id=source_concept.source_vocabulary_id,
                    source_code_description=source_concept.source_code_description,
                    target_concept_id=selected_result.std_concept_id,
                    target_vocabulary_id=selected_result.std_vocabulary_id.value,
                    domain_id=selected_result.std_domain_id,
                    valid_start_date=source_concept.valid_start_date,
                    valid_end_date=source_concept.valid_end_date,
                    invalid_reason=source_concept.invalid_reason,
                    target_vocabulary_code=selected_result.std_vocabulary_code,
                )
            )

        return mapped_source_concepts

    def to_dict(self):
        model_dict = self.model_dump()
        model_dict["target_vocabulary_id"] = model_dict["target_vocabulary_id"].value
        return model_dict


class RetrievedExpressionMetadata(ExpressionMetadata):
    # ignore extra fields
    model_config = ConfigDict(extra="ignore")

    # fields
    distance: Optional[float] = None


class SelectedExpressionMetadata(RetrievedExpressionMetadata):
    result_list_index: int


class EmptySelectionMetadata(RetrievedExpressionMetadata):
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


class RetrieverResults(BaseModel):
    results: List[List[RetrievedExpressionMetadata]]
    queries: List[str]


class SelectorResults(BaseModel):
    results: List[SelectedExpressionMetadata]
    queries: List[str]
