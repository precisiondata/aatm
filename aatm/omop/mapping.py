"""Declarative mapping from extracted entities to OMOP CDM table rows.

An entity definition can carry an ``omop`` block that says which table each mention
becomes a row of, and where each column's value comes from. The block is never sent to
the model: it is only read by `aatm.omop.builder.OmopTableBuilder`.

```yaml
- name: ECOG performance status
  key: ecog_performance_status
  definition: ...
  attributes: [...]
  omop:
    table: measurement
    fields:
      person_id: {from: context.person_id}
      measurement_concept_id: {concept: "SNOMED:423740007", maps_to: true}
      measurement_source_value: {from: text}
      measurement_date:              # a list: the first source with a value wins
        - {from: child.assessment_date, as: date}
        - {from: context.note_date}
      measurement_type_concept_id: {value: 32858}
      value_as_number: {from: attribute.score, as: float}
```

Each column has one source, or a list of sources tried in order:

- ``{from: text}``: the text of the mention.
- ``{from: attribute.<key>}``: the label of one of the entity's attributes. Add
  ``concepts: {<label>: "<VOCABULARY>:<code>"}`` to store a concept id instead.
- ``{from: child.<key>}``: the text of the first child mention with that key.
- ``{from: context.<name>}``: a value given with the note, see `ClinicalNote`.
- ``{value: <constant>}``: a fixed value, e.g. a type concept id.
- ``{concept: "<VOCABULARY>:<code>"}``: the concept id of a code, looked up in the
  local OMOP vocabulary database.

``maps_to: true`` replaces a concept by the standard concept it maps to ("Maps to"),
as the CDM requires for ``*_concept_id`` columns. ``as: int | float | str | date | days``
converts the value; dates are parsed from text such as "March 2024", and ``days`` reads
a duration such as "7 days", "2 weeks" or "1 month" (30 days) as a number of days.

Two options adjust any source:

- ``period_start: <column>``: the value is a number of days N, and the column gets the
  last day of the N-day period that starts on the date of ``<column>``, i.e. that date
  plus N - 1 days. ``{from: child.duration, as: days, period_start:
  drug_exposure_start_date}`` turns "for 7 days" into the end date of a 7-day course;
  ``{value: 1, period_start: ...}`` repeats the start date. ``<column>`` must be listed
  before this column.
- ``when: {attribute.<key>: [<label>, ...]}``: use the source only for mentions whose
  attribute has one of these labels; otherwise the next source is tried.

Text from the note can also be coded, with a ``from`` source and one of:

- ``map: {domain: Measurement, min_score: 0.8}``: send the text through the
  terminology mapper (translation, retrieval, reranking, selection), e.g. a lab test
  name to its standard concept. Any standard concept qualifies, whatever its
  vocabulary; ``domain`` defaults to the domain the column requires, and
  ``vocabularies: [...]`` optionally narrows the search. Results scoring below
  ``min_score`` are stored as ``0`` and reported.
- ``lookup: UCUM``: find the text as a code of that vocabulary, e.g. the unit
  ``"mg/dL"``. Exact matches win; otherwise a single standard match ignoring case.
"""

from datetime import date
from typing import Any, Literal, Optional, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aatm.omop.tables import OMOP_TABLES

SourceKind = Literal["text", "attribute", "child", "context"]


class ClinicalNote(BaseModel):
    """A clinical note and the values that come with it rather than from its text.

    Only `text` is needed for extraction. The other fields fill table columns through
    ``{from: context.<name>}`` sources, e.g. `person_id`, and `note_date` is the usual
    fallback for event dates the note does not state.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    note_id: Optional[int] = None
    person_id: Optional[int] = None
    note_date: Optional[date] = None
    visit_occurrence_id: Optional[int] = None
    visit_detail_id: Optional[int] = None
    provider_id: Optional[int] = None


CONTEXT_FIELDS = tuple(name for name in ClinicalNote.model_fields if name != "text")


class ConceptRef(BaseModel):
    """A concept identified by its vocabulary and code, e.g. ``SNOMED:423740007``."""

    model_config = ConfigDict(frozen=True)

    vocabulary_id: str
    concept_code: str

    @classmethod
    def parse(cls, value: str) -> "ConceptRef":
        vocabulary_id, sep, concept_code = value.partition(":")
        if not sep or not vocabulary_id.strip() or not concept_code.strip():
            raise ValueError(
                f"Concept {value!r} must be written as '<VOCABULARY>:<code>', "
                "e.g. 'SNOMED:423740007'"
            )
        return cls(
            vocabulary_id=vocabulary_id.strip(), concept_code=concept_code.strip()
        )

    def __str__(self) -> str:
        return f"{self.vocabulary_id}:{self.concept_code}"


class MapSpec(BaseModel):
    """How to code a text with the terminology mapper.

    Attributes:
        vocabularies: Optionally, only retrieve from these vocabularies, e.g.
            ``["LOINC"]``. By default every indexed vocabulary is searched: what a
            ``*_concept_id`` column requires is a standard concept, not a vocabulary.
        domain: Domain to retrieve from, e.g. ``"Measurement"``. Defaults to the domain
            the column requires.
        min_score: Mappings with a lower confidence score are stored as ``0`` (an
            unmapped value, as the CDM asks) and reported.
    """

    model_config = ConfigDict(extra="forbid")

    vocabularies: Optional[list[str]] = None
    domain: Optional[str] = None
    min_score: Optional[float] = Field(default=None, ge=0, le=1)


class FieldSource(BaseModel):
    """Where one column's value comes from. See the module docstring."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: Optional[str] = Field(default=None, alias="from")
    value: Optional[int | float | str] = None
    concept: Optional[ConceptRef] = None
    concepts: Optional[dict[str, ConceptRef]] = None
    maps_to: bool = False
    map: Optional[MapSpec] = None
    lookup: Optional[str] = None
    as_: Optional[Literal["int", "float", "str", "date", "days"]] = Field(
        default=None, alias="as"
    )
    period_start: Optional[str] = None
    when: Optional[dict[str, list[str]]] = None

    @field_validator("concept", mode="before")
    @classmethod
    def parse_concept(cls, value: Any) -> Any:
        return ConceptRef.parse(value) if isinstance(value, str) else value

    @field_validator("concepts", mode="before")
    @classmethod
    def parse_concepts(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {
            str(label): ConceptRef.parse(ref) if isinstance(ref, str) else ref
            for label, ref in value.items()
        }

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        given = [
            name
            for name, v in (
                ("from", self.from_),
                ("value", self.value),
                ("concept", self.concept),
            )
            if v is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "A field source needs exactly one of 'from', 'value' or 'concept', "
                f"got {given or 'none'}"
            )
        if self.from_ is not None:
            kind, _, key = self.from_.partition(".")
            if kind == "text":
                if key:
                    raise ValueError("Use 'from: text' without a key")
            elif kind in ("attribute", "child", "context"):
                if not key:
                    raise ValueError(
                        f"'from: {kind}' needs a key, e.g. 'from: {kind}.<key>'"
                    )
            else:
                raise ValueError(
                    f"Unknown source {self.from_!r}. Use 'text', 'attribute.<key>', "
                    "'child.<key>' or 'context.<name>'"
                )
        if self.concepts is not None and self.kind != "attribute":
            raise ValueError("'concepts' can only be used with 'from: attribute.<key>'")
        if self.maps_to and self.concept is None and self.concepts is None:
            raise ValueError("'maps_to' needs 'concept' or 'concepts'")
        coders = [
            name
            for name, v in (
                ("concepts", self.concepts),
                ("map", self.map),
                ("lookup", self.lookup),
            )
            if v is not None
        ]
        if len(coders) > 1:
            raise ValueError(f"Use only one of {coders}")
        if (self.map is not None or self.lookup is not None) and self.kind not in (
            "text",
            "attribute",
            "child",
        ):
            raise ValueError(
                "'map' and 'lookup' code text from the note: use them with "
                "'from: text', 'from: attribute.<key>' or 'from: child.<key>'"
            )
        if self.as_ is not None and (self.concept is not None or coders):
            raise ValueError("'as' cannot be used with concepts")
        if self.period_start is not None:
            numeric_constant = isinstance(self.value, int) and not isinstance(
                self.value, bool
            )
            if self.as_ != "days" and not numeric_constant:
                raise ValueError(
                    "'period_start' needs a number of days: 'as: days' with a 'from' "
                    "source, or an integer 'value'"
                )
        for condition, labels in (self.when or {}).items():
            kind, _, key = condition.partition(".")
            if kind != "attribute" or not key:
                raise ValueError(
                    f"'when' takes attribute labels, e.g. 'when: {{attribute.<key>: "
                    f"[<label>]}}', not {condition!r}"
                )
            if not labels:
                raise ValueError(f"'when: {condition}' lists no labels")
        return self

    @property
    def codes_text(self) -> bool:
        """Whether the value is coded from text with the mapper or a lookup."""
        return self.map is not None or self.lookup is not None

    @property
    def kind(self) -> Optional[SourceKind]:
        """The kind of a ``from`` source, e.g. ``"attribute"``."""
        if self.from_ is None:
            return None
        return self.from_.partition(".")[0]  # type: ignore[return-value]

    @property
    def key(self) -> Optional[str]:
        """The key of a ``from`` source, e.g. ``"score"`` for ``attribute.score``."""
        if self.from_ is None:
            return None
        return self.from_.partition(".")[2] or None

    def concept_refs(self) -> list[ConceptRef]:
        refs = [self.concept] if self.concept is not None else []
        return refs + list((self.concepts or {}).values())

    def when_attributes(self) -> dict[str, list[str]]:
        """The ``when`` conditions keyed by attribute key, e.g. ``{"kind": [...]}``."""
        return {
            condition.partition(".")[2]: labels
            for condition, labels in (self.when or {}).items()
        }


class OmopMapping(BaseModel):
    """The ``omop`` block of an entity definition: one table row per mention."""

    model_config = ConfigDict(extra="forbid")

    table: str
    fields: dict[str, list[FieldSource]]

    @field_validator("fields", mode="before")
    @classmethod
    def wrap_single_sources(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {k: v if isinstance(v, list) else [v] for k, v in value.items()}

    @model_validator(mode="after")
    def validate_columns(self) -> Self:
        model = OMOP_TABLES.get(self.table)
        if model is None:
            raise ValueError(
                f"Unknown OMOP table {self.table!r}. Supported: {sorted(OMOP_TABLES)}"
            )
        for column, sources in self.fields.items():
            if column not in model.model_fields:
                raise ValueError(f"Table {self.table!r} has no column {column!r}")
            if column == model.primary_key:
                raise ValueError(
                    f"{column!r} is generated by the table builder and cannot be mapped"
                )
            if not sources:
                raise ValueError(f"Column {column!r} has no sources")
        listed = list(self.fields)
        for column, sources in self.fields.items():
            for source in sources:
                start = source.period_start
                if start is None:
                    continue
                if start not in model.model_fields:
                    raise ValueError(
                        f"Column {column!r}: period_start {start!r} is not a column "
                        f"of {self.table!r}"
                    )
                if start not in listed or listed.index(start) > listed.index(column):
                    raise ValueError(
                        f"Column {column!r}: period_start {start!r} must be mapped "
                        "before it"
                    )
        return self

    def concept_refs(self) -> list[ConceptRef]:
        return [
            ref
            for sources in self.fields.values()
            for source in sources
            for ref in source.concept_refs()
        ]


def validate_entity_mapping(entity: Any, child_keys: set[str]) -> None:
    """Check that an entity's ``omop`` block only refers to things the entity has.

    Args:
        entity: An `EntityDefinition` with an ``omop`` block.
        child_keys: Keys of the entities whose parent is this entity.

    Raises:
        ValueError: If the entity is not top-level, or a source refers to an unknown
            attribute, label, child or context field.
    """
    mapping: OmopMapping = entity.omop
    if entity.has_parent:
        raise ValueError(
            f"Entity {entity.key!r} has a parent; only top-level entities can have an "
            "'omop' mapping. Map child values from the parent with 'from: child.<key>'"
        )
    for column, sources in mapping.fields.items():
        for source in sources:
            where = f"Entity {entity.key!r}, column {column!r}"
            for key, labels in source.when_attributes().items():
                attribute = entity.get_attribute(key)
                if attribute is None:
                    raise ValueError(f"{where}: 'when' uses unknown attribute {key!r}")
                for label in labels:
                    if attribute.match_label(label) is None:
                        raise ValueError(
                            f"{where}: 'when' label {label!r} is not a label of "
                            f"attribute {key!r}"
                        )
            if source.kind == "attribute":
                attribute = entity.get_attribute(source.key)
                if attribute is None:
                    raise ValueError(f"{where}: unknown attribute {source.key!r}")
                for label in source.concepts or {}:
                    if attribute.match_label(label) is None:
                        raise ValueError(
                            f"{where}: {label!r} is not a label of attribute "
                            f"{source.key!r}"
                        )
            elif source.kind == "child" and source.key not in child_keys:
                raise ValueError(
                    f"{where}: {source.key!r} is not a child entity of {entity.key!r}"
                )
            elif source.kind == "context" and source.key not in CONTEXT_FIELDS:
                raise ValueError(
                    f"{where}: unknown context field {source.key!r}. "
                    f"Available: {list(CONTEXT_FIELDS)}"
                )
