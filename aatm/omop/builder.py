"""Build OMOP CDM table rows from extracted entities.

`OmopTableBuilder` turns the validated output of an extractor into (partial) rows of
OMOP CDM tables, following the ``omop`` block of each entity definition (see
`aatm.omop.mapping`). `OmopExtractionPipeline` runs extraction and table building in
one call:

```python
from aatm.omop import ClinicalNote, OmopExtractionPipeline

pipeline = OmopExtractionPipeline("ecog_performance_status", model_id="openai/gpt-5-mini")
notes = [ClinicalNote(text="ECOG 1. Continue pembrolizumab.", person_id=42, note_date="2025-05-12")]
tables = await pipeline(notes)

tables["measurement"]  # a pandas DataFrame with one row per ECOG mention
tables.issues_dataframe()  # what could not be filled, and why
```
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, get_args

import asyncio

import pandas as pd
from dateutil import parser as date_parser
from pydantic import BaseModel, ValidationError

from aatm.data_models.extractors import (
    EntityDefinitions,
    EntityRecord,
    EntityValidationResult,
)
from aatm.logs import get_logger
from aatm.omop.coding import CodedText, TerminologyMapperCoder, TextCoder
from aatm.omop.mapping import ClinicalNote, ConceptRef, FieldSource, OmopMapping
from aatm.omop.tables import OMOP_TABLES, OmopTableRow
from aatm.omop.vocabulary import (
    DEFAULT_VOCABULARY_DB,
    Concept,
    ConceptLookup,
    SqliteConceptLookup,
)

logger = get_logger(__name__)

IssueReason = Literal[
    "missing_required",
    "unparsable_date",
    "imputed_date_parts",
    "unmapped_label",
    "invalid_value",
    "truncated",
    "multiple_children",
    "unmapped",
    "low_confidence",
    "domain_mismatch",
]

# ("map", domain, vocabularies) or ("lookup", vocabulary_id)
CodingKey = tuple


class OmopIssue(BaseModel):
    """A problem found while building one row.

    Attributes:
        table: The table of the row.
        row_id: The row's primary key, e.g. its ``measurement_id``.
        note_index: Position of the note in the input.
        note_id: The note's id, when given.
        entity_id: Id of the extracted mention the row was built from.
        column: The column concerned.
        reason: What went wrong, e.g. ``missing_required`` or ``unparsable_date``.
        detail: A human-readable explanation.
    """

    table: str
    row_id: Optional[int]
    note_index: int
    note_id: Optional[int]
    entity_id: Optional[str]
    column: Optional[str]
    reason: IssueReason
    detail: str


class RowProvenance(BaseModel):
    """Where a row came from: the note and the exact span of the mention."""

    note_index: int
    note_id: Optional[int]
    entity_id: str
    entity_key: str
    text: str
    start: int
    end: int


class ConceptMapping(BaseModel):
    """How a text from the note was coded, for reviewing mappings.

    Attributes:
        table: The table of the row.
        row_id: The row's primary key.
        column: The column the concept went to.
        text: The text that was coded, e.g. ``"Hemoglobin"``.
        method: ``"map"`` (terminology mapper) or ``"lookup"`` (code lookup).
        concept_id: The concept found, or ``None``.
        concept_name: Its name.
        vocabulary_id: Its vocabulary.
        concept_code: Its code.
        domain_id: Its domain.
        score: The mapper's confidence, for ``"map"``.
        stored: The value stored in the column: the concept id, or ``0`` when the
            mapping was rejected.
    """

    table: str
    row_id: Optional[int] = None
    column: str
    text: str
    method: Literal["map", "lookup"]
    concept_id: Optional[int]
    concept_name: Optional[str]
    vocabulary_id: Optional[str]
    concept_code: Optional[str]
    domain_id: Optional[str]
    score: Optional[float]
    stored: int


@dataclass
class OmopTables:
    """The rows built from a batch of notes, with their provenance and issues.

    ``tables["measurement"]`` returns the rows of a table as a DataFrame with the CDM
    columns, in CDM order. Missing values are ``None``/``NaN``.
    """

    rows: dict[str, list[OmopTableRow]] = field(default_factory=dict)
    provenance: dict[str, list[RowProvenance]] = field(default_factory=dict)
    issues: list[OmopIssue] = field(default_factory=list)
    mappings: list[ConceptMapping] = field(default_factory=list)
    extraction_results: list[EntityValidationResult] = field(default_factory=list)

    @property
    def tables(self) -> list[str]:
        return list(self.rows)

    def to_dataframes(self, provenance: bool = False) -> dict[str, pd.DataFrame]:
        """Return one DataFrame per table.

        Args:
            provenance: Add columns ``_note_index``, ``_note_id``, ``_entity_id``,
                ``_text``, ``_start`` and ``_end`` locating each row's mention.
        """
        frames = {}
        for table, rows in self.rows.items():
            model = OMOP_TABLES[table]
            columns = list(model.model_fields)
            df = pd.DataFrame([r.model_dump() for r in rows], columns=columns)
            # Nullable integers, so that ids are not shown as floats next to NULLs.
            for name, info in model.model_fields.items():
                if int in get_args(info.annotation):
                    df[name] = df[name].astype("Int64")
            if provenance:
                prov = pd.DataFrame(
                    [p.model_dump() for p in self.provenance[table]],
                    columns=list(RowProvenance.model_fields),
                ).drop(columns="entity_key")
                df = pd.concat([df, prov.add_prefix("_")], axis=1)
            frames[table] = df
        return frames

    def __getitem__(self, table: str) -> pd.DataFrame:
        if table not in self.rows:
            raise KeyError(f"No rows for table {table!r}. Tables: {self.tables}")
        return self.to_dataframes()[table]

    def issues_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [i.model_dump() for i in self.issues], columns=list(OmopIssue.model_fields)
        )

    def mappings_dataframe(self) -> pd.DataFrame:
        """One row per text coded with ``map`` or ``lookup``, to review the mappings."""
        return pd.DataFrame(
            [m.model_dump() for m in self.mappings],
            columns=list(ConceptMapping.model_fields),
        )


class _RowContext:
    """Collects the issues found while building one row."""

    def __init__(
        self,
        table: str,
        note_index: int,
        note: ClinicalNote,
        entity_id: str,
        coded: dict[tuple[CodingKey, str], CodedText],
    ):
        self.table = table
        self.note_index = note_index
        self.note = note
        self.entity_id = entity_id
        self.coded = coded
        self.issues: list[OmopIssue] = []
        self.mappings: list[ConceptMapping] = []

    def add(self, column: Optional[str], reason: IssueReason, detail: str) -> None:
        self.issues.append(
            OmopIssue(
                table=self.table,
                row_id=None,
                note_index=self.note_index,
                note_id=self.note.note_id,
                entity_id=self.entity_id,
                column=column,
                reason=reason,
                detail=detail,
            )
        )


def _parse_date(text: str, dayfirst: bool) -> tuple[Optional[date], bool]:
    """Parse a date written in a note.

    Returns the date and whether the day or month was missing and imputed as the
    first ("March 2024" -> 2024-03-01, True). Returns ``(None, False)`` when the text is
    not a date or has no year.
    """
    try:
        first = date_parser.parse(text, default=datetime(1900, 1, 1), dayfirst=dayfirst)
        second = date_parser.parse(
            text, default=datetime(1904, 2, 2), dayfirst=dayfirst
        )
    except (ValueError, OverflowError):
        return None, False
    if first.year != second.year:
        return None, False
    return first.date(), first.date() != second.date()


class OmopTableBuilder:
    """Turn extracted entities into OMOP CDM table rows.

    Every concept code in the definitions' ``omop`` blocks is looked up when the
    builder is created, so a typo in a code, a code without a standard mapping or a
    concept of the wrong domain fails here, before any note is processed.

    Args:
        entity_definitions: Definitions with ``omop`` blocks. Entities without one are
            ignored.
        vocabulary: The vocabulary database: a path to the SQLite database built by
            ``aatm init`` (default ``.aatm/omop.db``) or a `ConceptLookup`.
        dayfirst: Read ambiguous dates such as "05/12/2025" as day first (12 May would
            be month first).
        mapper: Codes ``map`` sources: a configured
            `aatm.terminology_mapper.TerminologyMapper`, or any `TextCoder`. Required
            when a mapping uses ``map``.

    Raises:
        ValueError: If no definition has an ``omop`` block, or a concept is unknown,
            has no single standard mapping, or belongs to the wrong domain, or a
            ``map`` source has no mapper or an unsupported vocabulary or domain.
    """

    def __init__(
        self,
        entity_definitions: EntityDefinitions,
        vocabulary: ConceptLookup | str | Path = DEFAULT_VOCABULARY_DB,
        dayfirst: bool = False,
        mapper: Any = None,
    ):
        self.entity_definitions = entity_definitions
        self.dayfirst = dayfirst
        self.mappings: dict[str, OmopMapping] = {
            e.key: e.omop for e in entity_definitions.entities if e.omop is not None
        }
        if not self.mappings:
            raise ValueError("None of the entity definitions has an 'omop' mapping")

        coded_sources = [
            (key, column, source)
            for key, m in self.mappings.items()
            for column, sources in m.fields.items()
            for source in sources
            if source.codes_text
        ]
        needs_vocabulary = bool(coded_sources) or any(
            m.concept_refs() or self._constant_concept_ids(m)
            for m in self.mappings.values()
        )
        self.vocabulary: Optional[ConceptLookup] = None
        if needs_vocabulary:
            self.vocabulary = (
                SqliteConceptLookup(vocabulary)
                if isinstance(vocabulary, (str, Path))
                else vocabulary
            )
        self.coder: Optional[TextCoder] = None
        if mapper is not None:
            self.coder = (
                mapper if hasattr(mapper, "code") else TerminologyMapperCoder(mapper)
            )
        # Configuration checks first: they are cheap and need no database.
        self._check_coded_sources(coded_sources)

        self.concepts: dict[tuple[ConceptRef, bool], Concept] = {}
        self._resolve_concepts()

    def _check_coded_sources(self, coded_sources: list) -> None:
        from aatm.data_models.data_models import StandardVocabulary

        indexed = {v.value for v in StandardVocabulary}
        for key, column, source in coded_sources:
            where = f"Entity {key!r}, column {column!r}"
            if source.map is None:
                continue
            if self.coder is None:
                raise ValueError(
                    f"{where} uses 'map', so the builder needs a terminology mapper: "
                    "OmopTableBuilder(..., mapper=TerminologyMapper(...))"
                )
            unknown = set(source.map.vocabularies or []) - indexed
            if unknown:
                raise ValueError(
                    f"{where}: the terminology mapper only indexes {sorted(indexed)}, "
                    f"not {sorted(unknown)}"
                )
            if source.map.domain is not None:
                problem = OMOP_TABLES[self.mappings[key].table].domain_problem(
                    column, source.map.domain
                )
                if problem:
                    raise ValueError(
                        f"{where}: 'map' would retrieve concepts {problem}"
                    )

    def _coding_key(self, table: str, column: str, source: FieldSource) -> CodingKey:
        if source.lookup is not None:
            return ("lookup", source.lookup)
        assert source.map is not None
        domain = source.map.domain or OMOP_TABLES[table].concept_domains.get(column)
        return ("map", domain, tuple(source.map.vocabularies or ()))

    @staticmethod
    def _constant_concept_ids(mapping: OmopMapping) -> dict[str, int]:
        """Constant concept ids to check in the vocabulary.

        ``0`` is left out: it is the CDM's value for "present but unmapped", e.g. a
        unit with no standard concept.
        """
        return {
            column: int(source.value)
            for column, sources in mapping.fields.items()
            if column.endswith("_concept_id")
            for source in sources
            if source.value is not None and int(source.value) != 0
        }

    def _resolve_concepts(self) -> None:
        if self.vocabulary is None:
            return
        refs = {r for m in self.mappings.values() for r in m.concept_refs()}
        found = self.vocabulary.by_code(refs)
        missing = sorted(str(r) for r in refs - set(found))
        if missing:
            raise ValueError(f"Concepts not found in the vocabulary: {missing}")

        needs_mapping = {
            ref
            for m in self.mappings.values()
            for sources in m.fields.values()
            for s in sources
            if s.maps_to
            for ref in s.concept_refs()
        }
        targets = self.vocabulary.maps_to(found[r].concept_id for r in needs_mapping)
        target_concepts = self.vocabulary.by_id(
            t for ids in targets.values() for t in ids
        )
        for ref, concept in found.items():
            self.concepts[(ref, False)] = concept
            if ref in needs_mapping:
                standard = [
                    target_concepts[t]
                    for t in targets.get(concept.concept_id, [])
                    if t in target_concepts
                ]
                if len(standard) != 1:
                    raise ValueError(
                        f"{concept} has {len(standard)} standard 'Maps to' concepts; "
                        "exactly one is needed for 'maps_to: true'"
                    )
                self.concepts[(ref, True)] = standard[0]

        constants = {
            (key, column): concept_id
            for key, m in self.mappings.items()
            for column, concept_id in self._constant_concept_ids(m).items()
        }
        constant_concepts = self.vocabulary.by_id(constants.values())

        for key, mapping in self.mappings.items():
            table_model = OMOP_TABLES[mapping.table]
            for column, sources in mapping.fields.items():
                for source in sources:
                    concepts = [
                        self.concepts[(r, source.maps_to)]
                        for r in source.concept_refs()
                    ]
                    if source.value is not None and (key, column) in constants:
                        concept_id = constants[(key, column)]
                        if concept_id not in constant_concepts:
                            raise ValueError(
                                f"Entity {key!r}, column {column!r}: concept_id "
                                f"{concept_id} not found in the vocabulary"
                            )
                        concepts.append(constant_concepts[concept_id])
                    for concept in concepts:
                        if column.endswith("_concept_id") and not column.endswith(
                            "source_concept_id"
                        ):
                            if not concept.is_standard:
                                raise ValueError(
                                    f"Entity {key!r}, column {column!r}: {concept} is "
                                    "not a standard concept. Add 'maps_to: true' to "
                                    "use the standard concept it maps to"
                                )
                        problem = table_model.domain_problem(column, concept.domain_id)
                        if problem:
                            raise ValueError(
                                f"Entity {key!r}, column {column!r}: {concept} is "
                                f"{problem}"
                            )

    def describe_concepts(self) -> pd.DataFrame:
        """The concepts the mappings resolve to, for checking them before a run."""
        rows = []
        for key, mapping in self.mappings.items():
            for column, sources in mapping.fields.items():
                for source in sources:
                    for label, ref in [(None, source.concept)] + list(
                        (source.concepts or {}).items()
                    ):
                        if ref is None:
                            continue
                        concept = self.concepts[(ref, source.maps_to)]
                        rows.append(
                            {
                                "entity": key,
                                "column": column,
                                "label": label,
                                "code": str(ref),
                                "concept_id": concept.concept_id,
                                "concept_name": concept.concept_name,
                                "vocabulary_id": concept.vocabulary_id,
                                "domain_id": concept.domain_id,
                            }
                        )
        return pd.DataFrame(rows)

    def build(
        self,
        notes: Sequence[ClinicalNote | str],
        results: Sequence[EntityValidationResult],
        id_start: int = 1,
    ) -> OmopTables:
        """Build table rows from extraction results.

        Args:
            notes: The notes, in the same order as `results`. Plain strings are
                accepted when no context (person, dates, visits) is available.
            results: The extractor's output for each note.
            id_start: First primary key value of each table.

        Returns:
            The rows of each mapped table, with provenance and issues.
        """
        if len(notes) != len(results):
            raise ValueError(
                f"Got {len(notes)} notes but {len(results)} extraction results"
            )
        tables = OmopTables(extraction_results=list(results))
        next_id: dict[str, int] = {}
        records = [result.to_records() for result in results]
        coded = self._code_texts(records)

        for note_index, (note, note_records) in enumerate(zip(notes, records)):
            if isinstance(note, str):
                note = ClinicalNote(text=note)
            for record in note_records:
                mapping = self.mappings.get(record.key)
                if mapping is None:
                    continue
                table = mapping.table
                row_id = next_id.get(table, id_start)
                next_id[table] = row_id + 1
                ctx = _RowContext(table, note_index, note, record.id, coded)
                row = self._build_row(mapping, record, ctx, row_id)
                for issue in ctx.issues:
                    issue.row_id = row_id
                for concept_mapping in ctx.mappings:
                    concept_mapping.row_id = row_id
                tables.mappings.extend(ctx.mappings)
                tables.rows.setdefault(table, []).append(row)
                tables.provenance.setdefault(table, []).append(
                    RowProvenance(
                        note_index=note_index,
                        note_id=note.note_id,
                        entity_id=record.id,
                        entity_key=record.key,
                        text=record.text,
                        start=record.start,
                        end=record.end,
                    )
                )
                tables.issues.extend(ctx.issues)
        return tables

    def _code_texts(
        self, records: list[list[EntityRecord]]
    ) -> dict[tuple[CodingKey, str], CodedText]:
        """Code the texts of all ``map`` and ``lookup`` sources, once per distinct text."""
        requests: dict[CodingKey, set[str]] = {}
        for note_records in records:
            for record in note_records:
                mapping = self.mappings.get(record.key)
                if mapping is None:
                    continue
                for column, sources in mapping.fields.items():
                    for source in sources:
                        if not source.codes_text:
                            continue
                        raw = self._raw(source, record)
                        if raw is not None and raw.strip():
                            key = self._coding_key(mapping.table, column, source)
                            requests.setdefault(key, set()).add(raw.strip())

        coded: dict[tuple[CodingKey, str], CodedText] = {}
        for key, texts in requests.items():
            ordered = sorted(texts)
            if key[0] == "map":
                assert self.coder is not None
                results = self.coder.code(
                    ordered, domain=key[1], vocabularies=list(key[2]) or None
                )
            else:
                results = self._lookup_codes(key[1], ordered)
            for result in results:
                coded[(key, result.text)] = result

        # The mapper does not return concept names: add them for review.
        unnamed = {
            c.concept_id for c in coded.values() if c.concept_id and not c.concept_name
        }
        if unnamed and self.vocabulary is not None:
            names = self.vocabulary.by_id(unnamed)
            for c in coded.values():
                if c.concept_id in names and not c.concept_name:
                    c.concept_name = names[c.concept_id].concept_name
        return coded

    def _lookup_codes(self, vocabulary_id: str, texts: list[str]) -> list[CodedText]:
        """Find texts as codes of a vocabulary: an exact match, else one standard match ignoring case."""
        assert self.vocabulary is not None
        found = self.vocabulary.search_codes(vocabulary_id, texts)
        results = []
        for text in texts:
            candidates = found.get(text.lower(), [])
            exact = [c for c in candidates if c.concept_code == text]
            pool = exact or candidates
            standard = [c for c in pool if c.is_standard] or pool
            chosen = standard[0] if len(standard) == 1 else None
            results.append(
                CodedText(
                    text=text,
                    concept_id=chosen.concept_id if chosen else None,
                    concept_name=chosen.concept_name if chosen else None,
                    vocabulary_id=chosen.vocabulary_id if chosen else None,
                    concept_code=chosen.concept_code if chosen else None,
                    domain_id=chosen.domain_id if chosen else None,
                    standard=chosen.is_standard if chosen else None,
                )
            )
        return results

    def _build_row(
        self,
        mapping: OmopMapping,
        record: EntityRecord,
        ctx: _RowContext,
        row_id: int,
    ) -> OmopTableRow:
        model = OMOP_TABLES[mapping.table]
        values: dict[str, Any] = {model.primary_key: row_id}
        for column, sources in mapping.fields.items():
            failed: list[OmopIssue] = []
            for source in sources:
                n_issues = len(ctx.issues)
                value = self._value(source, column, record, ctx)
                if value is not None:
                    values[column] = value
                    for issue in failed:
                        issue.detail += f"; used {self._describe(source)} instead"
                    break
                failed += ctx.issues[n_issues:]

        row = self._validate_row(model, values, ctx)
        for column in model.required_fields:
            if getattr(row, column) is None:
                ctx.add(column, "missing_required", f"{column} is required by the CDM")
        return row

    @staticmethod
    def _describe(source: FieldSource) -> str:
        if source.from_ is not None:
            return source.from_
        if source.concept is not None:
            return str(source.concept)
        return f"the constant {source.value!r}"

    def _validate_row(
        self, model: type[OmopTableRow], values: dict[str, Any], ctx: _RowContext
    ) -> OmopTableRow:
        while True:
            try:
                return model.model_validate(values)
            except ValidationError as e:
                for error in e.errors():
                    column = str(error["loc"][0])
                    value = values.pop(column, None)
                    max_length = (error.get("ctx") or {}).get("max_length")
                    if error["type"] == "string_too_long" and max_length:
                        values[column] = str(value)[:max_length]
                        ctx.add(
                            column,
                            "truncated",
                            f"{value!r} is longer than {max_length} characters",
                        )
                    else:
                        ctx.add(column, "invalid_value", f"{value!r}: {error['msg']}")

    def _value(
        self,
        source: FieldSource,
        column: str,
        record: EntityRecord,
        ctx: _RowContext,
    ) -> Any:
        if source.value is not None:
            return source.value
        if source.concept is not None:
            return self.concepts[(source.concept, source.maps_to)].concept_id

        if source.kind == "context":
            return getattr(ctx.note, source.key)  # type: ignore[arg-type]
        if source.kind == "child":
            children = record.children.get(source.key, [])  # type: ignore[arg-type]
            if len(children) > 1:
                ctx.add(
                    column,
                    "multiple_children",
                    f"{len(children)} {source.key!r} mentions; used the first, "
                    f"{children[0].text!r}",
                )

        raw = self._raw(source, record)
        if raw is None:
            return None
        if source.concepts is not None:
            return self._label_concept(source, column, raw, ctx)
        if source.codes_text:
            return self._coded_value(source, column, raw, ctx)
        return self._convert(raw, source.as_, column, ctx)

    @staticmethod
    def _raw(source: FieldSource, record: EntityRecord) -> Optional[str]:
        """The text a ``from`` source reads: the mention, a label or the first child."""
        if source.kind == "text":
            return record.text
        if source.kind == "attribute":
            return record.attributes.get(source.key)  # type: ignore[arg-type]
        if source.kind == "child":
            children = record.children.get(source.key, [])  # type: ignore[arg-type]
            return children[0].text if children else None
        return None

    def _coded_value(
        self, source: FieldSource, column: str, raw: str, ctx: _RowContext
    ) -> int:
        key = self._coding_key(ctx.table, column, source)
        method: Literal["map", "lookup"] = "map" if key[0] == "map" else "lookup"
        text = raw.strip()
        result = ctx.coded.get((key, text)) or CodedText(text=text)
        how = (
            f"the terminology mapper ({', '.join(key[2]) or 'all vocabularies'})"
            if method == "map"
            else f"a {key[1]} code lookup"
        )
        found = f"{text!r} -> {result.concept_name!r} ({result.concept_id})"
        stored = result.concept_id or 0
        if result.concept_id is None:
            ctx.add(column, "unmapped", f"No concept for {text!r} with {how}; stored 0")
        elif result.standard is False and not column.endswith("source_concept_id"):
            # *_concept_id columns hold standard concepts, whatever their vocabulary.
            stored = 0
            ctx.add(
                column,
                "unmapped",
                f"{found} is not a standard concept; stored 0",
            )
        elif (
            source.map is not None
            and source.map.min_score is not None
            and (result.score is None or result.score < source.map.min_score)
        ):
            stored = 0
            ctx.add(
                column,
                "low_confidence",
                f"{found} scored {result.score}, below min_score "
                f"{source.map.min_score}; stored 0",
            )
        else:
            problem = OMOP_TABLES[ctx.table].domain_problem(column, result.domain_id)
            if problem:
                ctx.add(column, "domain_mismatch", f"{found} is {problem}")
        ctx.mappings.append(
            ConceptMapping(
                table=ctx.table,
                column=column,
                text=text,
                method=method,
                concept_id=result.concept_id,
                concept_name=result.concept_name,
                vocabulary_id=result.vocabulary_id,
                concept_code=result.concept_code,
                domain_id=result.domain_id,
                score=result.score,
                stored=stored,
            )
        )
        return stored

    def _label_concept(
        self, source: FieldSource, column: str, label: str, ctx: _RowContext
    ) -> int:
        wanted = label.casefold()
        for mapped_label, ref in (source.concepts or {}).items():
            if mapped_label.casefold() == wanted:
                return self.concepts[(ref, source.maps_to)].concept_id
        ctx.add(
            column,
            "unmapped_label",
            f"Label {label!r} of {source.from_} has no concept; stored 0 as the CDM "
            "requires for unmapped values",
        )
        return 0

    def _convert(
        self, raw: str, as_: Optional[str], column: str, ctx: _RowContext
    ) -> Any:
        if as_ is None or as_ == "str":
            return raw
        if as_ == "date":
            parsed, imputed = _parse_date(raw, self.dayfirst)
            if parsed is None:
                ctx.add(column, "unparsable_date", f"{raw!r} is not a full date")
            elif imputed:
                ctx.add(
                    column,
                    "imputed_date_parts",
                    f"{raw!r} has no day or month; read as {parsed.isoformat()}",
                )
            return parsed
        number = raw.strip()
        if "," in number and "." not in number:
            number = number.replace(",", ".")  # decimal comma, e.g. "11,2"
        try:
            return int(number) if as_ == "int" else float(number)
        except ValueError:
            ctx.add(column, "invalid_value", f"{raw!r} is not a number")
            return None


def _load_definitions(
    definitions: EntityDefinitions
    | str
    | Path
    | Sequence[EntityDefinitions | str | Path],
) -> EntityDefinitions:
    """Load definitions given as objects, predefined names or YAML paths, and combine a list."""
    if isinstance(definitions, EntityDefinitions):
        return definitions
    if isinstance(definitions, (str, Path)):
        path = Path(definitions)
        if path.suffix in (".yaml", ".yml"):
            return EntityDefinitions.from_yaml(path)
        return EntityDefinitions.from_predefined(str(definitions))
    loaded = [
        (_source_name(d, i), _load_definitions(d)) for i, d in enumerate(definitions)
    ]
    _check_unique_keys(loaded)
    return EntityDefinitions(entities=[e for _, d in loaded for e in d.entities])


def _source_name(definitions: EntityDefinitions | str | Path, index: int) -> str:
    """How to name one item of a definitions list in error messages."""
    if isinstance(definitions, EntityDefinitions):
        return f"item {index + 1} (an EntityDefinitions object)"
    path = Path(definitions)
    if path.suffix in (".yaml", ".yml"):
        return str(path)
    return f"predefined '{definitions}'"


def _check_unique_keys(loaded: list[tuple[str, EntityDefinitions]]) -> None:
    """Raise if items of a definitions list repeat, or define the same entity key.

    The message names the files, so a conflict between a predefined file and your own
    can be found without reading every file.
    """
    names = [source for source, _ in loaded]
    repeated = sorted({n for n in names if names.count(n) > 1})

    sources_of: dict[str, list[str]] = {}
    seen: set[str] = set()
    for source, definitions in loaded:
        if source in seen:
            continue  # a repeated file is reported once below, not once per key
        seen.add(source)
        for entity in definitions.entities:
            sources_of.setdefault(entity.key, []).append(source)
    conflicts = {k: v for k, v in sorted(sources_of.items()) if len(v) > 1}
    if not repeated and not conflicts:
        return

    lines = [f"  - {n} is listed {names.count(n)} times" for n in repeated]
    lines += [
        f"  - key {key!r} is defined in {' and '.join(sources)}"
        for key, sources in conflicts.items()
    ]
    raise ValueError(
        "Entity keys must be unique across the combined definitions, since the "
        "model's output is matched to definitions by key:\n"
        + "\n".join(lines)
        + "\nList each file once, and rename keys that clash in your own files."
    )


class OmopExtractionPipeline:
    """Extract entities from notes and build OMOP CDM tables in one call.

    Args:
        entity_definitions: Definitions with ``omop`` blocks, the name of a predefined
            file (e.g. ``"ecog_performance_status"``), a path to a YAML file, or a list
            of these, combined so each note is extracted once, e.g.
            ``["smoking_status", "family_history"]``.
        model_id: Provider-prefixed model id, e.g. ``"openai/gpt-5-mini"``. Ignored
            when `extractor` is given.
        extractor: A ready-made extractor, e.g. a configured `SimpleLLMExtractor`.
        vocabulary: See `OmopTableBuilder`.
        dayfirst: See `OmopTableBuilder`.
        mapper: See `OmopTableBuilder`.
        **extractor_kwargs: Passed to `SimpleLLMExtractor`, e.g. ``max_concurrency``.
    """

    def __init__(
        self,
        entity_definitions: EntityDefinitions
        | str
        | Path
        | Sequence[EntityDefinitions | str | Path],
        model_id: Optional[str] = None,
        extractor: Any = None,
        vocabulary: ConceptLookup | str | Path = DEFAULT_VOCABULARY_DB,
        dayfirst: bool = False,
        mapper: Any = None,
        **extractor_kwargs: Any,
    ):
        entity_definitions = _load_definitions(entity_definitions)
        self.entity_definitions = entity_definitions
        self.builder = OmopTableBuilder(
            entity_definitions, vocabulary=vocabulary, dayfirst=dayfirst, mapper=mapper
        )
        if extractor is None:
            from aatm.extractors import SimpleLLMExtractor

            if model_id is None:
                raise ValueError("Either model_id or extractor must be given")
            extractor = SimpleLLMExtractor(
                model_id=model_id,
                entity_definitions=entity_definitions,
                **extractor_kwargs,
            )
        self.extractor = extractor

    async def __call__(
        self, notes: Sequence[ClinicalNote | str], id_start: int = 1
    ) -> OmopTables:
        clinical_notes = [
            ClinicalNote(text=n) if isinstance(n, str) else n for n in notes
        ]
        results = await self.extractor([n.text for n in clinical_notes])
        # Building may call the terminology mapper, which blocks: keep the loop free.
        return await asyncio.to_thread(
            self.builder.build, clinical_notes, results, id_start
        )
