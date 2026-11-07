from pathlib import Path
from typing import List, Literal, Optional, Self
import logging
import re
import unicodedata
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml
from importlib.resources import files

from aatm.logs import get_logger

logger = get_logger(__name__, level=logging.DEBUG)


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
        has_parent (bool): Whether extracted spans of this entity must reference a parent span.
        parent_key (Optional[str]): The key of the parent entity. Required when `has_parent` is True.
    """

    name: str
    key: str = Field(default_factory=lambda data: make_entity_key(data["name"]))
    definition: str
    has_parent: bool = False
    parent_key: Optional[str] = None

    @model_validator(mode="after")
    def validate_parent(self) -> Self:
        if self.has_parent and not self.parent_key:
            raise ValueError(
                f"Entity {self.key!r} has has_parent=True but no parent_key"
            )
        if not self.has_parent and self.parent_key:
            raise ValueError(
                f"Entity {self.key!r} defines parent_key but has_parent is False"
            )
        if self.parent_key == self.key:
            raise ValueError(f"Entity {self.key!r} cannot be its own parent")
        return self

    def to_prompt_str(self) -> str:
        return f"{self.name} (key: `{self.key}`, has_parent: {self.has_parent}, parent_key: `{self.parent_key}`): {self.definition}"


class EntityDefinitions(BaseModel):
    """
    A list of entity definitions.

    Args:
        entities (List[EntityDefinition]): A list of entity definitions.
    """

    entities: List[EntityDefinition]

    @model_validator(mode="after")
    def validate_keys(self) -> Self:
        keys = [entity.key for entity in self.entities]
        duplicates = {k for k in keys if keys.count(k) > 1}
        if duplicates:
            raise ValueError(f"Duplicate entity keys: {sorted(duplicates)}")

        for entity in self.entities:
            if entity.parent_key is not None and entity.parent_key not in keys:
                raise ValueError(
                    f"Entity {entity.key!r} references unknown parent_key "
                    f"{entity.parent_key!r}"
                )
        return self

    @property
    def keys(self) -> set[str]:
        return {entity.key for entity in self.entities}

    @property
    def parent_keys(self) -> dict[str, str]:
        """Mapping from entity key to its parent key, for entities that require a parent."""
        return {
            entity.key: entity.parent_key
            for entity in self.entities
            if entity.has_parent and entity.parent_key is not None
        }

    def to_prompt_str(self) -> str:
        return "\n".join([ent.to_prompt_str() for ent in self.entities])

    @classmethod
    def from_predefined(cls, key: str) -> Self:
        available_defs = {
            p.stem: p
            for p in Path(str(files("aatm").joinpath("omop/entity_definitions"))).glob(
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


class TextSegment(BaseModel):
    """
    A non-empty line (or part of a long line) of a source text, identified by a line id.

    Args:
        id (str): The line id shown to the model, e.g. ``"L12"``.
        start (int): Start index of the segment in the source text.
        end (int): End index of the segment in the source text.
        text (str): The segment text, equal to ``source[start:end]``.
    """

    id: str = Field(pattern=r"^L[1-9][0-9]*$")
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.end - self.start != len(self.text):
            raise ValueError("span length must equal the length of text")
        return self


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;])\s+")
_LINE_PREFIX = re.compile(r"^\s*\[L[0-9]+\]\s*")


def _split_long_line(line: str, max_chars: int) -> List[tuple[int, int]]:
    """Split a line into chunks of at most `max_chars`, cutting only at sentence boundaries.

    Sentences longer than `max_chars` are kept whole.
    """
    if len(line) <= max_chars:
        return [(0, len(line))]

    sentences: List[tuple[int, int]] = []
    pos = 0
    for m in _SENTENCE_BOUNDARY.finditer(line):
        sentences.append((pos, m.start()))
        pos = m.end()
    sentences.append((pos, len(line)))

    chunks: List[tuple[int, int]] = []
    chunk_start, chunk_end = sentences[0]
    for start, end in sentences[1:]:
        if end - chunk_start <= max_chars:
            chunk_end = end
        else:
            chunks.append((chunk_start, chunk_end))
            chunk_start, chunk_end = start, end
    chunks.append((chunk_start, chunk_end))
    return chunks


def segment_text(
    text: str, max_segment_chars: Optional[int] = None
) -> List[TextSegment]:
    """
    Split a text into numbered segments for line-grounded extraction.

    Each non-blank line becomes one segment, with surrounding whitespace trimmed. Segment ids
    are sequential (``L1``, ``L2``, ...) and every segment keeps its character offsets in `text`.

    By default lines are never split. If `max_segment_chars` is set, lines longer than it are
    split at sentence boundaries into separately numbered segments. This helps with sources
    that have long paragraphs or lost their line breaks, but the sentence splitter is naive
    (it also cuts after abbreviations such as "p.o. "), so enable it only when needed.

    Args:
        text (str): The source text.
        max_segment_chars (Optional[int]): Soft maximum segment length before sentence
            splitting. ``None`` (default) disables splitting.

    Returns:
        List[TextSegment]: The segments in document order.
    """
    if max_segment_chars is not None and max_segment_chars < 1:
        raise ValueError("max_segment_chars must be positive")

    segments: List[TextSegment] = []
    for line_match in re.finditer(r"[^\r\n]+", text):
        line = line_match.group()
        pieces = (
            [(0, len(line))]
            if max_segment_chars is None
            else _split_long_line(line, max_segment_chars)
        )
        for start, end in pieces:
            piece = line[start:end]
            stripped = piece.strip()
            if not stripped:
                continue
            abs_start = line_match.start() + start + (len(piece) - len(piece.lstrip()))
            segments.append(
                TextSegment(
                    id=f"L{len(segments) + 1}",
                    start=abs_start,
                    end=abs_start + len(stripped),
                    text=stripped,
                )
            )
    return segments


def format_segments_for_prompt(segments: List[TextSegment]) -> str:
    """Render segments as ``[L1] text`` lines, the input format the extractor model sees."""
    return "\n".join(f"[{s.id}] {s.text}" for s in segments)


class ExtractedEntityCandidate(BaseModel):
    """
    An entity extracted from a text without any post-processing or validation.

    Args:
        text (str): The exact text of the entity, copied from the line it appears in.
        key (str): The key of the entity.
        line (str): The id of the line the entity appears in, e.g. ``"L12"``.
        occurrence (Optional[int]): Which occurrence of `text` within that line is meant,
            starting at 1. ``None`` means the first one.
        parents (Optional[List[str]]): A list of spans that represent the parent of this extracted entity. Parent spans must receive a key compatible with this entity definition.
    """

    text: str = Field(min_length=1)
    key: str = Field(min_length=1, pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    line: str = Field(min_length=1)
    occurrence: Optional[int] = Field(default=None, ge=1)
    parents: Optional[List[str]] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        # Models sometimes copy the "[L12] " line prefix into the span.
        value = _LINE_PREFIX.sub("", value)
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator("line")
    @classmethod
    def normalize_line(cls, value: str) -> str:
        # Accept "L12", "[L12]", "l12" and "12".
        value = value.strip().strip("[]").strip().upper()
        if value.isdigit():
            value = f"L{value}"
        return value


class ExtractedEntityCandidates(BaseModel):
    entities: List[ExtractedEntityCandidate]


RejectionReason = Literal[
    "duplicate",
    "invalid_key",
    "invalid_line",
    "invalid_text",
    "text_not_in_line",
    "invalid_occurrence",
    "required_parent_missing",
    "invalid_parent",
]


class RejectedEntityCandidate(ExtractedEntityCandidate):
    start: Optional[int] = Field(default=None, ge=0)
    end: Optional[int] = Field(default=None, ge=0)
    reasons: List[RejectionReason]

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        start, end = self.start, self.end

        if start is None or end is None:
            if start is not None or end is not None:
                raise ValueError("start and end must be provided together")
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
        line (str): The line id the entity was grounded in. Inherited from ExtractedEntityCandidate.
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


def _ground_entity(
    ent: ExtractedEntityCandidate,
    segments_by_id: dict[str, TextSegment],
    text: str,
) -> tuple[Optional[tuple[int, int]], List[RejectionReason]]:
    """Resolve a candidate to a single (start, end) span in `text` using its line id."""
    segment = segments_by_id.get(ent.line)
    if segment is None:
        return None, ["invalid_line"]

    starts = [m.start() for m in re.finditer(re.escape(ent.text), segment.text)]
    if not starts:
        reason: RejectionReason = (
            "text_not_in_line" if ent.text in text else "invalid_text"
        )
        return None, [reason]

    occurrence = ent.occurrence or 1
    if occurrence > len(starts):
        return None, ["invalid_occurrence"]

    start = segment.start + starts[occurrence - 1]
    return (start, start + len(ent.text)), []


def validate_extracted_entity(
    text: str,
    entity_definitions: EntityDefinitions,
    extracted_entities: ExtractedEntityCandidates,
    segments: Optional[List[TextSegment]] = None,
) -> EntityValidationResult:
    """
    Validate a list of extracted entities against a text.

    Each candidate is grounded to exactly one span: its `text` must appear verbatim in the
    segment named by its `line` id (the `occurrence`-th match, or the first one). Candidates
    that cannot be grounded are rejected with `invalid_line`, `invalid_text` (not in the
    document at all), `text_not_in_line` (in the document, but not on that line) or
    `invalid_occurrence`. Validated entities are deduplicated on span, key and parents.

    Entities whose definition requires a parent must list at least one parent span, and every
    parent span must match the text of a grounded candidate extracted under the parent key
    defined in `entity_definitions`. Otherwise the entity is rejected with
    `required_parent_missing` or `invalid_parent`.

    Args:
        text (str): The text to validate against.
        entity_definitions (EntityDefinitions): An object containing entity definitions.
        extracted_entities (ExtractedEntityCandidates): A list of extracted entities to validate.
        segments (Optional[List[TextSegment]]): The segments shown to the model. Must be the
            same segmentation used to build the prompt. Defaults to `segment_text(text)`.

    Returns:
        EntityValidationResult: An object containing a list of validated entities and a list of unmatched entities.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text:
        raise ValueError("text must not be empty")

    if segments is None:
        segments = segment_text(text)
    segments_by_id = {s.id: s for s in segments}

    allowed_keys = entity_definitions.keys
    parent_keys = entity_definitions.parent_keys
    results = EntityValidationResult()
    seen: set[tuple[int, int, str, tuple[str, ...]]] = set()
    seen_keys: set[str] = set()

    logger.debug(f"Validating {len(extracted_entities.entities)} entities")
    logger.debug(f"""Allowed keys: {", ".join(sorted(allowed_keys))}""")
    logger.debug(
        f"""Entity keys with required parents: {", ".join(sorted(parent_keys))}"""
    )

    grounded = [
        (ent, *_ground_entity(ent, segments_by_id, text))
        for ent in extracted_entities.entities
    ]

    # Texts of grounded candidates with a valid key: these can act as parents.
    texts_by_key: dict[str, set[str]] = {}
    for ent, span, _ in grounded:
        if span is not None and ent.key in allowed_keys:
            texts_by_key.setdefault(ent.key, set()).add(ent.text)

    for ent, span, grounding_reasons in grounded:
        rejection_reasons: List[RejectionReason] = list(grounding_reasons)

        if ent.key not in allowed_keys:
            rejection_reasons.append("invalid_key")

        if ent.key in parent_keys:
            if not ent.parents:
                rejection_reasons.append("required_parent_missing")
            else:
                valid_parents = texts_by_key.get(parent_keys[ent.key], set())
                if any(p not in valid_parents for p in ent.parents):
                    rejection_reasons.append("invalid_parent")

        start, end = span if span is not None else (None, None)

        if rejection_reasons:
            results.unmatched_entities.append(
                RejectedEntityCandidate(
                    **ent.model_dump(),
                    start=start,
                    end=end,
                    reasons=rejection_reasons,
                )
            )
            continue

        assert start is not None and end is not None
        identity = (start, end, ent.key, tuple(ent.parents or ()))
        if identity in seen:
            results.unmatched_entities.append(
                RejectedEntityCandidate(
                    **ent.model_dump(),
                    start=start,
                    end=end,
                    reasons=["duplicate"],
                )
            )
            continue

        seen.add(identity)
        seen_keys.add(ent.key)
        results.validated_entities.append(
            ExtractedEntity(**ent.model_dump(), start=start, end=end)
        )

    results.unmatched_keys = sorted(allowed_keys - seen_keys)
    results.validated_entities = sorted(
        results.validated_entities, key=lambda x: x.start
    )
    return results
