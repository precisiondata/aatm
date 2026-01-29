import hashlib
from typing import Any, Optional
from pydantic import BaseModel, field_validator
from enum import Enum


def deterministic_id_from_strings(strings: list[str], digest_size: int = 8) -> str:
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
    expression: str
    expression_concept_id: str
    expression_origin: ExpressionOrigin
    std_concept_id: str
    std_concept_name: str
    std_vocabulary_id: StandardVocabulary
    std_vocabulary_code: str

    @field_validator("expression_concept_id", "std_concept_id", mode="before")
    @classmethod
    def validate_concept_id(cls, value: Any) -> str:
        return str(value)

    def model_post_init(self, *args, **kwargs):
        self.expression_id = deterministic_id_from_strings(
            [
                self.expression,
                self.expression_concept_id,
                self.expression_origin.value,
                self.std_concept_id,
                self.std_concept_name,
                self.std_vocabulary_id.value,
                self.std_vocabulary_code,
            ]
        )

    def to_dict(self):
        model_dict = self.model_dump()
        model_dict["expression_origin"] = model_dict["expression_origin"].value
        model_dict["std_vocabulary_id"] = model_dict["std_vocabulary_id"].value
        return model_dict
