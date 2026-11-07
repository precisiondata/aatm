from pathlib import Path
from typing import List, Literal, Optional, Self
import re
import unicodedata
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
import yaml
from importlib.resources import files

from aatm.logs import get_logger

logger = get_logger(__name__)


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


def _compute_key_depths(entities: List[EntityDefinition]) -> dict[str, int]:
    parent_of = {e.key: e.parent_key for e in entities}
    depths: dict[str, int] = {}

    for key in parent_of:
        chain: List[str] = []
        current: Optional[str] = key
        while current is not None and current not in depths:
            if current in chain:
                cycle = " -> ".join([*chain[chain.index(current) :], current])
                raise ValueError(f"Cycle in entity parent keys: {cycle}")
            chain.append(current)
            current = parent_of.get(current)

        base = -1 if current is None else depths[current]
        for offset, k in enumerate(reversed(chain), start=1):
            depths[k] = base + offset

    return depths


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

        # Raises on cycles such as a -> b -> a.
        _compute_key_depths(self.entities)
        return self

    @property
    def keys(self) -> set[str]:
        return {entity.key for entity in self.entities}

    @property
    def key_depths(self) -> dict[str, int]:
        """Depth of each key in the parent hierarchy: 0 for roots, 1 for their children, etc."""
        return _compute_key_depths(self.entities)

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
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Invalid YAML in entity definitions {path}: {e}"
                ) from e

        try:
            return cls(entities=data)
        except ValidationError as e:
            # Name the file, so developers loading several definition files know
            # which one is broken. The original error stays attached as __cause__.
            raise ValueError(f"Invalid entity definitions in {path}:\n{e}") from e


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
        id (str): A local id for this entity, unique within one extraction, e.g. ``"e1"``.
        text (str): The exact text of the entity, copied from the line it appears in.
        key (str): The key of the entity.
        line (str): The id of the line the entity appears in, e.g. ``"L12"``.
        occurrence (Optional[int]): Which occurrence of `text` within that line is meant,
            starting at 1. ``None`` means the first one.
        parents (Optional[List[str]]): Ids of the parent entities of this entity. Each parent
            must have the `parent_key` defined for this entity's key. An entity may have
            several parents, e.g. one date that applies to two conditions.
    """

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    key: str = Field(min_length=1, pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    line: str = Field(min_length=1)
    occurrence: Optional[int] = Field(default=None, ge=1)
    parents: Optional[List[str]] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("id must not be blank")
        return value

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

    @field_validator("parents")
    @classmethod
    def normalize_parents(cls, value: Optional[List[str]]) -> List[str]:
        if not value:
            return []
        return [p.strip() for p in value if p.strip()]


class ExtractedEntityCandidates(BaseModel):
    entities: List[ExtractedEntityCandidate]


RejectionReason = Literal[
    "duplicate",
    "duplicate_id",
    "invalid_key",
    "invalid_line",
    "invalid_text",
    "text_not_in_line",
    "invalid_occurrence",
    "required_parent_missing",
    "unexpected_parent",
    "invalid_parent",
    "parent_rejected",
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
        id (str): The local id of the entity. Inherited from ExtractedEntityCandidate.
        text (str): The text of the entity. Inherited from ExtractedEntityCandidate.
        key (str): The key of the entity. Inherited from ExtractedEntityCandidate.
        line (str): The line id the entity was grounded in. Inherited from ExtractedEntityCandidate.
        parents (List[str]): Ids of validated parent entities. Ids of dropped duplicates are
            replaced by the id of the entity they duplicate.
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


class EntityRecord(BaseModel):
    """
    A validated entity with its children nested under it, grouped by child key.

    Built by `EntityValidationResult.to_records`. A child with several parents appears under
    each of them.
    """

    id: str
    key: str
    text: str
    line: str
    start: int
    end: int
    children: dict[str, List["EntityRecord"]] = Field(default_factory=dict)


class EntityValidationResult(BaseModel):
    validated_entities: list[ExtractedEntity] = Field(default_factory=list)
    unmatched_entities: list[RejectedEntityCandidate] = Field(default_factory=list)
    unmatched_keys: list[str] = Field(default_factory=list)

    def get_entity(self, entity_id: str) -> Optional[ExtractedEntity]:
        """Return the validated entity with this id, or ``None``."""
        return next((e for e in self.validated_entities if e.id == entity_id), None)

    def to_records(self) -> List[EntityRecord]:
        """
        Nest validated entities under their parents.

        Returns one record per root entity (an entity without parents), in document order.
        Each record holds its children grouped by key, recursively.
        """
        children_of: dict[str, List[ExtractedEntity]] = {}
        for entity in self.validated_entities:
            for parent_id in entity.parents or []:
                children_of.setdefault(parent_id, []).append(entity)

        def build(entity: ExtractedEntity, ancestors: frozenset[str]) -> EntityRecord:
            record = EntityRecord(
                id=entity.id,
                key=entity.key,
                text=entity.text,
                line=entity.line,
                start=entity.start,
                end=entity.end,
            )
            for child in children_of.get(entity.id, []):
                if child.id in ancestors:
                    continue
                record.children.setdefault(child.key, []).append(
                    build(child, ancestors | {entity.id})
                )
            return record

        return [build(e, frozenset()) for e in self.validated_entities if not e.parents]


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
    Validate extracted entities against a text and their entity definitions.

    The result is a graph: each validated entity has one exact span in the text and refers
    to its parents by id.

    Grounding: each candidate's `text` must appear verbatim in the segment named by its
    `line` id (the `occurrence`-th match, or the first one). Otherwise it is rejected with
    `invalid_line`, `invalid_text` (not in the document at all), `text_not_in_line` (in the
    document, but not on that line) or `invalid_occurrence`.

    Ids: ids must be unique. A candidate reusing an id is rejected with `duplicate_id`, and
    the id keeps referring to the first candidate.

    Parents: an entity whose definition has a `parent_key` must list at least one parent id
    (`required_parent_missing`). Every parent id must belong to a candidate with that parent
    key (`invalid_parent`) that was itself validated (`parent_rejected`). Entities whose
    definition has no parent must not list parents (`unexpected_parent`).

    Duplicates: candidates with the same span and key as an already validated entity are
    rejected with `duplicate`; their parents are merged into the kept entity, and children
    pointing to the duplicate are relinked to the kept entity.

    Args:
        text (str): The text to validate against.
        entity_definitions (EntityDefinitions): An object containing entity definitions.
        extracted_entities (ExtractedEntityCandidates): A list of extracted entities to validate.
        segments (Optional[List[TextSegment]]): The segments shown to the model. Must be the
            same segmentation used to build the prompt. Defaults to `segment_text(text)`.

    Returns:
        EntityValidationResult: The validated entities (in document order), the rejected
            candidates (in input order) and the entity keys with no validated entity.
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
    depths = entity_definitions.key_depths

    logger.debug(f"Validating {len(extracted_entities.entities)} entities")
    logger.debug(f"""Allowed keys: {", ".join(sorted(allowed_keys))}""")
    logger.debug(
        f"""Entity keys with required parents: {", ".join(sorted(parent_keys))}"""
    )

    rejected: List[tuple[int, RejectedEntityCandidate]] = []

    def reject(
        index: int,
        ent: ExtractedEntityCandidate,
        span: Optional[tuple[int, int]],
        reasons: List[RejectionReason],
    ) -> None:
        start, end = span if span is not None else (None, None)
        rejected.append(
            (
                index,
                RejectedEntityCandidate(
                    **ent.model_dump(), start=start, end=end, reasons=reasons
                ),
            )
        )

    # The first candidate with a given id owns it.
    candidates_by_id: dict[str, ExtractedEntityCandidate] = {}
    queue: List[tuple[int, int, ExtractedEntityCandidate]] = []
    for index, ent in enumerate(extracted_entities.entities):
        if ent.id in candidates_by_id:
            reject(index, ent, None, ["duplicate_id"])
            continue
        candidates_by_id[ent.id] = ent
        queue.append((depths.get(ent.key, 0), index, ent))

    # Parents always have a smaller depth than their children, so processing by depth
    # guarantees every parent is decided before its children.
    queue.sort(key=lambda item: (item[0], item[1]))

    # Candidate id -> id of the validated entity it ended up as (itself, or the entity
    # it duplicates). Ids missing here belong to rejected candidates.
    resolved: dict[str, str] = {}
    validated_by_span: dict[tuple[int, int, str], ExtractedEntity] = {}

    for _, index, ent in queue:
        span, grounding_reasons = _ground_entity(ent, segments_by_id, text)
        reasons: List[RejectionReason] = list(grounding_reasons)

        if ent.key not in allowed_keys:
            reasons.append("invalid_key")

        parent_ids = ent.parents or []
        resolved_parents: List[str] = []
        required_parent_key = parent_keys.get(ent.key)

        if required_parent_key is None:
            if parent_ids and ent.key in allowed_keys:
                reasons.append("unexpected_parent")
        elif not parent_ids:
            reasons.append("required_parent_missing")
        else:
            for parent_id in parent_ids:
                parent = candidates_by_id.get(parent_id)
                if parent is None or parent.key != required_parent_key:
                    reasons.append("invalid_parent")
                    break
                if parent_id not in resolved:
                    reasons.append("parent_rejected")
                    break
                if resolved[parent_id] not in resolved_parents:
                    resolved_parents.append(resolved[parent_id])

        if reasons:
            reject(index, ent, span, reasons)
            continue

        assert span is not None
        identity = (span[0], span[1], ent.key)
        existing = validated_by_span.get(identity)
        if existing is not None:
            merged = list(existing.parents or [])
            merged += [p for p in resolved_parents if p not in merged]
            existing.parents = merged
            resolved[ent.id] = existing.id
            reject(index, ent, span, ["duplicate"])
            continue

        entity = ExtractedEntity(
            **{**ent.model_dump(), "parents": resolved_parents},
            start=span[0],
            end=span[1],
        )
        validated_by_span[identity] = entity
        resolved[ent.id] = entity.id

    validated = sorted(validated_by_span.values(), key=lambda e: (e.start, e.end))
    return EntityValidationResult(
        validated_entities=validated,
        unmatched_entities=[r for _, r in sorted(rejected, key=lambda item: item[0])],
        unmatched_keys=sorted(allowed_keys - {e.key for e in validated}),
    )
