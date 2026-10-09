from pathlib import Path
from typing import List, Literal, Optional, Self
import re
import unicodedata
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml
from importlib.resources import files

from aatm.logs import get_logger

logger = get_logger(__name__, level="DEBUG")


def make_entity_key(name: str) -> str:
    normalized_name = unicodedata.normalize("NFKD", name)
    ascii_name = normalized_name.encode("ascii", "ignore").decode("ascii")

    key = re.sub(
        r"[^a-z0-9]+",
        "_",
        ascii_name.casefold(),
    ).strip("_")

    if not key:
        raise ValueError(f"Could not generate a valid key from name {name!r}")

    return key


class EntityDefinition(BaseModel):
    """
    A definition of an entity.

    Args:
        name (str): The name of the entity.
        key (str): The key of the entity. If not provided, the name will be used with all letters lowercased and spaces replaced by underscores.
        definition (str): The definition of the entity.
    """

    name: str
    key: str = Field(default_factory=lambda data: make_entity_key(data["name"]))
    definition: str
    has_parent: bool = False
    parent_key: Optional[str] = Field(
        default_factory=lambda data: data["key"] if data["has_parent"] else None
    )

    def to_prompt_str(self):
        return f"{self.name} (key: `{self.key}`, has_parent: {self.has_parent}, parent_key: `{self.parent_key}`): {self.definition}"


class EntityDefinitions(BaseModel):
    """
    A list of entity definitions.

    Args:
        entities (List[EntityDefinition]): A list of entity definitions.
    """

    entities: List[EntityDefinition]

    @property
    def keys(self) -> set[str]:
        return {entity.key for entity in self.entities if entity.key is not None}

    def to_prompt_str(self):
        return "\n".join([ent.to_prompt_str() for ent in self.entities])

    @classmethod
    def from_predefined(cls, key: str) -> Self:
        available_defs = {
            p.stem: p
            for p in Path(files("aatm").joinpath("omop/entity_definitions")).glob(
                "*.yaml"
            )
        }
        if key not in available_defs:
            raise ValueError(
                f"Unknown entity definition: {key}. Available definitions: {list(available_defs.keys())}"
            )

        path = available_defs[key]
        return cls.from_yaml(path)

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        if not path.exists():
            raise ValueError(f"File not found: {path}")

        if not path.name.endswith(".yaml") and not path.name.endswith(".yml"):
            raise ValueError(f"Unsupported file format: {path}. Expected .yaml or .yml")

        with open(path, encoding="utf-8") as f:
            return cls(entities=yaml.safe_load(f))


class ExtractedEntityCandidate(BaseModel):
    """
    An entity extracted from a text without any post-processing or validation.

    Args:
        text (str): The text of the entity.
        key (str): The key of the entity.
        parents (Optional[List[str]]): A list of spans that represent the parent of this extracted entity. Parent spans must receive a key compatible with this entity definition.
    """

    text: str = Field(min_length=1)
    key: str = Field(min_length=1, pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    parents: Optional[List[str]] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class ExtractedEntityCandidates(BaseModel):
    entities: List[ExtractedEntityCandidate]


RejectionReason = Literal[
    "duplicate",
    "invalid_key",
    "invalid_text",
    "required_parent_missing",
]


class RejectedEntityCandidate(ExtractedEntityCandidate):
    start: Optional[int] = Field(default=None, ge=0)
    end: Optional[int] = Field(default=None, ge=0)
    reasons: List[RejectionReason]

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        start, end = self.start, self.end

        if (start is None) != (end is None):
            raise ValueError("start and end must be provided together")

        if start is None and end is None:
            return self

        if start >= end:
            raise ValueError("start must be less than end")

        if end - start != len(self.text):
            raise ValueError("span length must equal the length of text")

        return self


class ExtractedEntity(ExtractedEntityCandidate):
    """
    An entity extracted from a text after post-processing and validation.

    Args:
        text (str): The text of the entity. Inherited from ExtractedEntityCandidate.
        key (str): The key of the entity. Inherited from ExtractedEntityCandidate.
        start (int): The start index of the entity in the text.
        end (int): The end index of the entity in the text.
    """

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.start >= self.end:
            raise ValueError("start must be less than end")

        if self.end - self.start != len(self.text):
            raise ValueError("span length must equal the length of text")
        return self


class EntityValidationResult(BaseModel):
    validated_entities: list[ExtractedEntity] = Field(default_factory=list)
    unmatched_entities: list[RejectedEntityCandidate] = Field(default_factory=list)
    unmatched_keys: list[str] = Field(default_factory=list)


def validate_extracted_entity(
    text: str,
    entity_definitions: EntityDefinitions,
    extracted_entities: ExtractedEntityCandidates,
) -> EntityValidationResult:
    """
    Validate a list of extracted entities against a text. Returns a list of ExtractedEntity objects, which includes spans that are exact matches in the text. Matched entities are deduplicated based on span indices, span text and key.

    Args:
        text (str): The text to validate against.
        extracted_entities (ExtractedEntityCandidates): A list of extracted entities to validate.
        entity_definitions (EntityDefinitions): An object containing entity definitions.

    Returns:
        EntityValidationResult: An object containing a list of validated entities and a list of unmatched entities.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text:
        raise ValueError("text must not be empty")

    allowed_keys = entity_definitions.keys
    results = EntityValidationResult()
    seen: set[tuple[int, int, str, str]] = set()
    seen_keys: set[str] = set()
    entity_keys_with_required_parents: set[str] = {
        e.key for e in entity_definitions.entities if e.has_parent
    }

    logger.debug(f"Validating {len(extracted_entities.entities)} entities")
    logger.debug(f"""Allowed keys: {", ".join(allowed_keys)}""")
    logger.debug(
        f"""Entity keys with required parents: {", ".join(entity_keys_with_required_parents)}"""
    )

    for ent in extracted_entities.entities:
        matches = list(re.finditer(re.escape(ent.text), text))
        rejection_reasons: List[RejectionReason] = []

        if not matches:
            rejection_reasons.append("invalid_text")

        if allowed_keys is not None and ent.key not in allowed_keys:
            rejection_reasons.append("invalid_key")

        if ent.key in entity_keys_with_required_parents and (
            ent.parents is None or len(ent.parents) == 0
        ):
            rejection_reasons.append("required_parent_missing")

        if rejection_reasons:
            results.unmatched_entities.append(
                RejectedEntityCandidate(
                    **ent.model_dump(),
                    reasons=rejection_reasons,
                )
            )
            continue

        for match in matches:
            validated_entity = ExtractedEntity(
                **ent.model_dump(),
                start=match.start(),
                end=match.end(),
            )

            identity = (
                match.start(),
                match.end(),
                ent.key,
                ent.text,
                "|".join(ent.parents) if ent.parents else None,
            )

            if identity in seen:
                results.unmatched_entities.append(
                    RejectedEntityCandidate(
                        **ent.model_dump(),
                        start=match.start(),
                        end=match.end(),
                        reasons=["duplicate"],
                    )
                )
                continue

            seen.add(identity)
            seen_keys.add(ent.key)
            results.validated_entities.append(validated_entity)
    results.unmatched_keys = list(allowed_keys - seen_keys)
    results.validated_entities = sorted(
        results.validated_entities, key=lambda x: x.start
    )
    return results
