"""Data models and validation for LLM-based entity extraction.

This module holds everything the extraction pipeline needs that does not depend on
a particular LLM provider:

- **Entity definitions** (`EntityDefinition`, `EntityDefinitions`): which entities
  to extract, loaded from YAML. Entities may require a parent entity through
  `parent_key` (for example, a start date that belongs to a condition), at any depth.
  Entities may also define `attributes` (`AttributeDefinition`): classifications with a
  fixed set of labels, such as the result of a test, assigned to each mention.
- **Line segmentation** (`segment_text`, `format_segments_for_prompt`): the note is
  shown to the model as numbered lines (``[L1] ...``) so that every extracted span can
  be tied to one exact position.
- **Model output** (`ExtractedEntityCandidate`, `ExtractedEntityCandidates`): what the
  model returns: a local id, the exact text, the entity key, the line id and the ids
  of its parents.
- **Validation** (`validate_extracted_entity`): grounds each candidate to a single
  character span, checks keys and parent links, merges duplicates, and returns an
  `EntityValidationResult` with validated entities, rejected candidates and their
  rejection reasons.
- **Records view** (`EntityValidationResult.to_records`): validated entities nested
  under their parents, ready to be turned into rows such as OMOP
  `condition_occurrence`.

See the "Information extraction" guide for a walkthrough.
"""

from pathlib import Path
from typing import Annotated, List, Literal, Optional, Self
import re
import unicodedata
from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
import yaml
from importlib.resources import files

from aatm.logs import get_logger
from aatm.omop.mapping import OmopMapping, validate_entity_mapping

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


_KEY_PATTERN = r"^[a-z0-9]+(?:_[a-z0-9]+)*$"

# Text the model reads. Stripped so YAML block scalars (`>` or `|`) leave no trailing newline.
_PromptText = Annotated[str, StringConstraints(strip_whitespace=True)]


class AttributeLabel(BaseModel):
    """
    One allowed value of an attribute.

    Attributes:
        value (str): The label, e.g. ``"positive"``. This is what the model returns.
        description (str): When this label applies. The model reads it.
    """

    value: str = Field(min_length=1)
    description: _PromptText

    @field_validator("value")
    @classmethod
    def strip_value(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("label value must not be blank")
        return value


class AttributeDefinition(BaseModel):
    """
    A classification attached to each mention of an entity, e.g. the result of a test.

    The model picks one label per attribute, based on what the note states about that
    mention. It is not a span: the label does not have to appear in the text.

    Attributes:
        key (str): The key of the attribute, e.g. ``"result"``. Unique within the entity.
        description (str): What the attribute describes. The model reads it.
        required (bool): If True, a mention without a valid label for this attribute is
            rejected. If False (default), the mention is kept without the attribute.
        labels (List[AttributeLabel]): The allowed labels.
    """

    key: str = Field(pattern=_KEY_PATTERN)
    description: _PromptText
    required: bool = False
    labels: List[AttributeLabel] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        values = [label.value.casefold() for label in self.labels]
        duplicates = {v for v in values if values.count(v) > 1}
        if duplicates:
            raise ValueError(
                f"Attribute {self.key!r} has duplicate labels: {sorted(duplicates)}"
            )
        return self

    def match_label(self, value: str) -> Optional[str]:
        """Return the defined label matching `value` (ignoring case and surrounding spaces), or ``None``."""
        wanted = value.strip().casefold()
        return next(
            (label.value for label in self.labels if label.value.casefold() == wanted),
            None,
        )

    def to_prompt_str(self) -> str:
        requirement = "required" if self.required else "optional"
        lines = [f"  - `{self.key}` ({requirement}): {self.description}"]
        lines += [
            f"    - `{label.value}`: {label.description}" for label in self.labels
        ]
        return "\n".join(lines)


class EntityDefinition(BaseModel):
    """
    A definition of an entity.

    Attributes:
        name (str): The name of the entity.
        key (str): The key of the entity. If not provided, the name will be used with all letters lowercased and spaces replaced by underscores.
        definition (str): The definition of the entity.
        has_parent (bool): Whether extracted spans of this entity must reference a parent span.
        parent_key (Optional[str]): The key of the parent entity. Required when `has_parent` is True.
        attributes (List[AttributeDefinition]): Classifications the model assigns to each
            mention of this entity, e.g. the result of a test.
        omop (Optional[OmopMapping]): How each mention becomes a row of an OMOP CDM
            table, see `aatm.omop.mapping`. Never sent to the model.
    """

    name: str
    key: str = Field(default_factory=lambda data: make_entity_key(data["name"]))
    definition: _PromptText
    has_parent: bool = False
    parent_key: Optional[str] = None
    attributes: List[AttributeDefinition] = Field(default_factory=list)
    omop: Optional[OmopMapping] = None

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        # Generated keys always match; this catches keys written by hand. The model must
        # return keys in this form (see ExtractedEntityCandidate.key), so a key such as
        # "BRCA-Test" would otherwise make every model response fail to parse.
        if not re.fullmatch(_KEY_PATTERN, value):
            try:
                hint = f" Use {make_entity_key(value)!r} instead."
            except ValueError:
                hint = ""
            raise ValueError(
                f"Entity key {value!r} must use only lowercase letters, digits and "
                f"single underscores between them.{hint}"
            )
        return value

    @model_validator(mode="after")
    def validate_attributes(self) -> Self:
        keys = [attribute.key for attribute in self.attributes]
        duplicates = {k for k in keys if keys.count(k) > 1}
        if duplicates:
            raise ValueError(
                f"Entity {self.key!r} has duplicate attribute keys: {sorted(duplicates)}"
            )
        return self

    def get_attribute(self, key: str) -> Optional[AttributeDefinition]:
        """Return the attribute definition with this key (ignoring case and surrounding spaces), or ``None``."""
        wanted = key.strip().casefold()
        return next((a for a in self.attributes if a.key == wanted), None)

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
        # Only what the model needs: the `omop` mapping is deliberately left out.
        line = f"{self.name} (key: `{self.key}`, has_parent: {self.has_parent}, parent_key: `{self.parent_key}`): {self.definition}"
        if not self.attributes:
            return line
        attributes = "\n".join(a.to_prompt_str() for a in self.attributes)
        return f"{line}\n  Attributes:\n{attributes}"


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

    Attributes:
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

        for entity in self.entities:
            if entity.omop is not None:
                child_keys = {
                    e.key for e in self.entities if e.parent_key == entity.key
                }
                validate_entity_mapping(entity, child_keys)
        return self

    @property
    def keys(self) -> set[str]:
        return {entity.key for entity in self.entities}

    @property
    def key_depths(self) -> dict[str, int]:
        """Depth of each key in the parent hierarchy: 0 for roots, 1 for their children, etc."""
        return _compute_key_depths(self.entities)

    def get(self, key: str) -> Optional[EntityDefinition]:
        """Return the entity definition with this key, or ``None``."""
        return next((e for e in self.entities if e.key == key), None)

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
    def from_yaml(cls, path: str | Path) -> Self:
        """Load entity definitions from a YAML file containing a list of definitions.

        Raises:
            ValueError: If the file is missing, is not a .yaml/.yml file, is not valid
                YAML, or contains invalid definitions. The message names the file.
        """
        path = Path(path)
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

    Attributes:
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


class ExtractedAttribute(BaseModel):
    """
    A label the model assigned to one attribute of an extracted entity.

    Attributes:
        key (str): The attribute key, e.g. ``"result"``.
        value (Optional[str]): The chosen label, e.g. ``"negative"``. ``None`` means the
            note does not state it, the same as leaving the attribute out.
    """

    key: str
    value: Optional[str] = None


class ExtractedEntityCandidate(BaseModel):
    """
    An entity extracted from a text without any post-processing or validation.

    Attributes:
        id (str): A local id for this entity, unique within one extraction, e.g. ``"e1"``.
        text (str): The exact text of the entity, copied from the line it appears in.
        key (str): The key of the entity.
        line (str): The id of the line the entity appears in, e.g. ``"L12"``.
        occurrence (Optional[int]): Which occurrence of `text` within that line is meant,
            starting at 1. ``None`` means the first one.
        parents (Optional[List[str]]): Ids of the parent entities of this entity. Each parent
            must have the `parent_key` defined for this entity's key. An entity may have
            several parents, e.g. one date that applies to two conditions.
        attributes (List[ExtractedAttribute]): Labels for the attributes defined for this
            entity's key. Attributes the note does not state are left out.
    """

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    key: str = Field(min_length=1, pattern=_KEY_PATTERN)
    line: str = Field(min_length=1)
    occurrence: Optional[int] = Field(default=None, ge=1)
    parents: Optional[List[str]] = Field(default_factory=list)
    attributes: List[ExtractedAttribute] = Field(default_factory=list)

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
    "required_attribute_missing",
    "invalid_required_attribute",
]


AttributeIssueReason = Literal[
    "unknown_attribute",
    "invalid_label",
    "missing_required",
    "conflicting_values",
    "conflicting_duplicate",
]


class AttributeIssue(BaseModel):
    """
    A problem with an attribute the model returned (or failed to return) for one candidate.

    Attributes:
        entity_id (str): The id of the candidate the attribute belongs to.
        entity_key (str): The key of that candidate.
        attribute_key (str): The attribute key, as returned by the model or as defined
            (for ``missing_required``).
        value (Optional[str]): The value the model returned, if any.
        reason (AttributeIssueReason): What went wrong:

            - ``unknown_attribute``: the entity has no attribute with this key.
            - ``invalid_label``: the value is not one of the attribute's labels.
            - ``missing_required``: a required attribute was not given.
            - ``conflicting_values``: the attribute was given twice for the same
              candidate with different values; the first valid one is kept.
            - ``conflicting_duplicate``: a duplicate mention gave a different value than
              the mention it duplicates; the first mention's value is kept.
        required (bool): Whether the attribute is required by its definition.
        kept_entity_id (Optional[str]): For ``conflicting_duplicate``, the id of the
            validated entity whose value was kept.
        kept_value (Optional[str]): For ``conflicting_values`` and
            ``conflicting_duplicate``, the value that was kept.
    """

    entity_id: str
    entity_key: str
    attribute_key: str
    value: Optional[str] = None
    reason: AttributeIssueReason
    required: bool = False
    kept_entity_id: Optional[str] = None
    kept_value: Optional[str] = None


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

    Attributes:
        id (str): The local id of the entity. Inherited from ExtractedEntityCandidate.
        text (str): The text of the entity. Inherited from ExtractedEntityCandidate.
        key (str): The key of the entity. Inherited from ExtractedEntityCandidate.
        line (str): The line id the entity was grounded in. Inherited from ExtractedEntityCandidate.
        parents (List[str]): Ids of validated parent entities. Ids of dropped duplicates are
            replaced by the id of the entity they duplicate.
        attributes (List[ExtractedAttribute]): Only valid attributes, each once, with the
            label spelled as defined. Invalid ones are reported as `AttributeIssue`.
        start (int): The start index of the entity in the text.
        end (int): The end index of the entity in the text.
    """

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    def get_attribute(self, key: str) -> Optional[str]:
        """Return the label of attribute `key`, or ``None`` if it was not given."""
        return next((a.value for a in self.attributes if a.key == key), None)

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
    attributes: dict[str, str] = Field(default_factory=dict)
    children: dict[str, List["EntityRecord"]] = Field(default_factory=dict)


class EntityValidationResult(BaseModel):
    """
    The outcome of validating one extraction.

    Attributes:
        validated_entities: Validated entities in document order.
        unmatched_entities: Rejected candidates with their reasons, in input order.
        unmatched_keys: Definition keys with no validated entity.
        required_attribute_issues: Problems with attributes defined as required. A
            ``missing_required`` or ``invalid_label`` issue here means the candidate was
            rejected.
        optional_attribute_issues: Problems with optional or unknown attributes. The
            candidate is kept, without the faulty attribute.
    """

    validated_entities: list[ExtractedEntity] = Field(default_factory=list)
    unmatched_entities: list[RejectedEntityCandidate] = Field(default_factory=list)
    unmatched_keys: list[str] = Field(default_factory=list)
    required_attribute_issues: list[AttributeIssue] = Field(default_factory=list)
    optional_attribute_issues: list[AttributeIssue] = Field(default_factory=list)

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
                attributes={
                    a.key: a.value for a in entity.attributes if a.value is not None
                },
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


def _check_attributes(
    ent: ExtractedEntityCandidate,
    definition: EntityDefinition,
) -> tuple[List[ExtractedAttribute], List[AttributeIssue], List[RejectionReason]]:
    """Check a candidate's attributes against its entity definition.

    Returns the valid attributes (in definition order, labels spelled as defined), the
    issues found, and the rejection reasons caused by required attributes.
    """
    chosen: dict[str, str] = {}
    issues: List[AttributeIssue] = []
    invalid_keys: set[str] = set()

    def issue(
        attribute_key: str,
        value: Optional[str],
        reason: AttributeIssueReason,
        required: bool,
        kept_value: Optional[str] = None,
    ) -> None:
        issues.append(
            AttributeIssue(
                entity_id=ent.id,
                entity_key=ent.key,
                attribute_key=attribute_key,
                value=value,
                reason=reason,
                required=required,
                kept_value=kept_value,
            )
        )

    for attribute in ent.attributes:
        # A null or blank value means "not stated", the same as leaving it out.
        if attribute.value is None or not attribute.value.strip():
            continue
        attribute_def = definition.get_attribute(attribute.key)
        if attribute_def is None:
            issue(attribute.key, attribute.value, "unknown_attribute", False)
            continue
        label = attribute_def.match_label(attribute.value)
        if label is None:
            invalid_keys.add(attribute_def.key)
            issue(
                attribute_def.key,
                attribute.value,
                "invalid_label",
                attribute_def.required,
            )
            continue
        kept = chosen.get(attribute_def.key)
        if kept is None:
            chosen[attribute_def.key] = label
        elif kept != label:
            issue(
                attribute_def.key,
                attribute.value,
                "conflicting_values",
                attribute_def.required,
                kept_value=kept,
            )

    reasons: List[RejectionReason] = []
    for attribute_def in definition.attributes:
        if not attribute_def.required or attribute_def.key in chosen:
            continue
        if attribute_def.key in invalid_keys:
            reason: RejectionReason = "invalid_required_attribute"
        else:
            reason = "required_attribute_missing"
            issue(attribute_def.key, None, "missing_required", True)
        if reason not in reasons:
            reasons.append(reason)

    valid = [
        ExtractedAttribute(key=a.key, value=chosen[a.key])
        for a in definition.attributes
        if a.key in chosen
    ]
    return valid, issues, reasons


def _merge_duplicate_attributes(
    existing: ExtractedEntity,
    duplicate_id: str,
    attributes: List[ExtractedAttribute],
    definition: EntityDefinition,
) -> List[AttributeIssue]:
    """Add a duplicate's attributes to the kept entity; the kept entity wins conflicts."""
    issues: List[AttributeIssue] = []
    for attribute in attributes:
        kept = existing.get_attribute(attribute.key)
        if kept is None:
            existing.attributes.append(attribute)
        elif kept != attribute.value:
            attribute_def = definition.get_attribute(attribute.key)
            issues.append(
                AttributeIssue(
                    entity_id=duplicate_id,
                    entity_key=existing.key,
                    attribute_key=attribute.key,
                    value=attribute.value,
                    reason="conflicting_duplicate",
                    required=attribute_def.required if attribute_def else False,
                    kept_entity_id=existing.id,
                    kept_value=kept,
                )
            )
    order = [a.key for a in definition.attributes]
    existing.attributes.sort(key=lambda a: order.index(a.key))
    return issues


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

    Attributes: checked only for candidates that pass the checks above. Labels are matched
    ignoring case and surrounding spaces. An unknown attribute, an invalid label or a
    second, different value for the same attribute is dropped and reported as an
    `AttributeIssue`; the span is kept. A required attribute without a valid label rejects
    the candidate (`required_attribute_missing` or `invalid_required_attribute`), and its
    children with it. For duplicates, attributes missing on the kept entity are copied
    from the duplicate, and different values are reported as `conflicting_duplicate`
    (the kept entity's value wins). Issues on required attributes go to
    `required_attribute_issues`, all others to `optional_attribute_issues`.

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

    definitions_by_key = {e.key: e for e in entity_definitions.entities}

    rejected: List[tuple[int, RejectedEntityCandidate]] = []
    required_issues: List[AttributeIssue] = []
    optional_issues: List[AttributeIssue] = []

    def record_attribute_issues(issues: List[AttributeIssue]) -> None:
        for issue in issues:
            (required_issues if issue.required else optional_issues).append(issue)

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
        definition = definitions_by_key[ent.key]
        attributes, attribute_issues, attribute_reasons = _check_attributes(
            ent, definition
        )

        identity = (span[0], span[1], ent.key)
        existing = validated_by_span.get(identity)
        if existing is not None:
            merged = list(existing.parents or [])
            merged += [p for p in resolved_parents if p not in merged]
            existing.parents = merged
            # The kept entity already has every required attribute, so a duplicate
            # missing one is not a problem worth reporting.
            record_attribute_issues(
                [i for i in attribute_issues if i.reason != "missing_required"]
                + _merge_duplicate_attributes(existing, ent.id, attributes, definition)
            )
            resolved[ent.id] = existing.id
            reject(index, ent, span, ["duplicate"])
            continue

        record_attribute_issues(attribute_issues)
        if attribute_reasons:
            reject(index, ent, span, attribute_reasons)
            continue

        entity = ExtractedEntity(
            **{
                **ent.model_dump(),
                "parents": resolved_parents,
                "attributes": [a.model_dump() for a in attributes],
            },
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
        required_attribute_issues=required_issues,
        optional_attribute_issues=optional_issues,
    )
