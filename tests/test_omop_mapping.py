import asyncio
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from aatm.data_models.extractors import (
    EntityDefinitions,
    ExtractedEntityCandidates,
    validate_extracted_entity,
)
from aatm.omop import ClinicalNote, ConceptRef, OmopExtractionPipeline, OmopTableBuilder
from aatm.omop.builder import _parse_date
from aatm.omop.vocabulary import Concept, SqliteConceptLookup


def concept(concept_id, code, name, domain, vocabulary="SNOMED", standard="S"):
    return Concept(
        concept_id=concept_id,
        concept_name=name,
        domain_id=domain,
        vocabulary_id=vocabulary,
        concept_code=code,
        standard_concept=standard,
    )


ECOG_CONCEPTS = [
    concept(
        4308014, "423740007", "ECOG performance status", "Observation", standard=None
    ),
    concept(
        36305384, "89247-1", "ECOG Performance Status score", "Measurement", "LOINC"
    ),
    concept(4175026, "425389002", "ECOG grade 0", "Observation"),
    concept(4173614, "422512005", "ECOG grade 1", "Observation"),
    concept(4172043, "422894000", "ECOG grade 2", "Observation"),
    concept(4174241, "423053003", "ECOG grade 3", "Observation"),
    concept(4174251, "423237006", "ECOG grade 4", "Observation"),
    concept(4173456, "423409001", "ECOG grade 5", "Observation"),
    concept(32858, "OMOP4976931", "NLP", "Type Concept", "Type Concept"),
]


class FakeLookup:
    def __init__(self, concepts=ECOG_CONCEPTS, maps_to=None):
        self.concepts = {c.concept_id: c for c in concepts}
        self.maps = maps_to if maps_to is not None else {4308014: [36305384]}

    def by_code(self, refs):
        refs = set(refs)
        return {
            ref: c
            for c in self.concepts.values()
            if (
                ref := ConceptRef(
                    vocabulary_id=c.vocabulary_id, concept_code=c.concept_code
                )
            )
            in refs
        }

    def by_id(self, ids):
        return {i: self.concepts[i] for i in ids if i in self.concepts}

    def maps_to(self, ids):
        return {i: self.maps[i] for i in ids if i in self.maps}


def ecog_definitions(omop=None, **fields):
    mapping = omop or {
        "table": "measurement",
        "fields": {
            "person_id": {"from": "context.person_id"},
            "measurement_concept_id": {"concept": "SNOMED:423740007", "maps_to": True},
            "measurement_source_concept_id": {"concept": "SNOMED:423740007"},
            "measurement_source_value": {"from": "text"},
            "measurement_date": [
                {"from": "child.assessment_date", "as": "date"},
                {"from": "context.note_date"},
            ],
            "measurement_type_concept_id": {"value": 32858},
            "value_as_number": {"from": "attribute.score", "as": "float"},
            "value_as_concept_id": {
                "from": "attribute.score",
                "concepts": {"0": "SNOMED:425389002", "1": "SNOMED:422512005"},
            },
            "value_source_value": {"from": "attribute.score"},
            **fields,
        },
    }
    return EntityDefinitions(
        entities=[
            {
                "name": "ECOG",
                "key": "ecog",
                "definition": "An ECOG performance status.",
                "attributes": [
                    {
                        "key": "score",
                        "description": "The grade.",
                        "labels": [
                            {"value": str(g), "description": f"Grade {g}."}
                            for g in range(6)
                        ],
                    }
                ],
                "omop": mapping,
            },
            {
                "name": "Assessment date",
                "key": "assessment_date",
                "definition": "When it was assessed.",
                "has_parent": True,
                "parent_key": "ecog",
            },
        ]
    )


def extract(definitions, text, entities):
    candidates = ExtractedEntityCandidates.model_validate({"entities": entities})
    return validate_extracted_entity(text, definitions, candidates)


def ecog(id, text, score=None, line="L1"):
    attributes = [{"key": "score", "value": score}] if score is not None else []
    return {
        "id": id,
        "text": text,
        "key": "ecog",
        "line": line,
        "attributes": attributes,
    }


def child(id, text, parent, line="L1"):
    return {
        "id": id,
        "text": text,
        "key": "assessment_date",
        "line": line,
        "parents": [parent],
    }


# --- the mapping is not part of the prompt ------------------------------------


def test_omop_mapping_is_not_sent_to_the_model():
    with_mapping = ecog_definitions()
    without_mapping = EntityDefinitions(
        entities=[e.model_copy(update={"omop": None}) for e in with_mapping.entities]
    )
    assert with_mapping.to_prompt_str() == without_mapping.to_prompt_str()


def test_predefined_ecog_prompt_has_no_omop_details():
    defs = EntityDefinitions.from_predefined("ecog_performance_status")
    assert defs.get("ecog_performance_status").omop.table == "measurement"
    prompt = defs.to_prompt_str()
    for leaked in ("measurement", "SNOMED", "423740007", "concept", "omop"):
        assert leaked not in prompt


# --- load-time validation ------------------------------------------------------


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"not_a_column": {"from": "text"}}, "has no column 'not_a_column'"),
        ({"measurement_id": {"value": 1}}, "generated by the table builder"),
        (
            {"value_source_value": {"from": "attribute.result"}},
            "unknown attribute 'result'",
        ),
        (
            {"value_source_value": {"from": "child.dose"}},
            "'dose' is not a child entity",
        ),
        ({"provider_id": {"from": "context.doctor"}}, "unknown context field 'doctor'"),
        ({"value_source_value": {"from": "line"}}, "Unknown source 'line'"),
        ({"value_source_value": {"from": "text", "value": "x"}}, "exactly one of"),
        ({"measurement_concept_id": {"concept": "423740007"}}, "'<VOCABULARY>:<code>'"),
        (
            {
                "value_as_concept_id": {
                    "from": "attribute.score",
                    "concepts": {"9": "SNOMED:1"},
                }
            },
            "'9' is not a label of attribute 'score'",
        ),
        ({"value_source_value": {"from": "text", "maps_to": True}}, "'maps_to' needs"),
    ],
)
def test_invalid_mappings_fail_when_loaded(fields, message):
    with pytest.raises(ValueError, match=message):
        ecog_definitions(**fields)


def test_unknown_table_fails_when_loaded():
    with pytest.raises(ValueError, match="Unknown OMOP table 'measurements'"):
        ecog_definitions(omop={"table": "measurements", "fields": {}})


def test_mapping_on_a_child_entity_fails_when_loaded():
    defs = ecog_definitions().model_dump(by_alias=True, exclude_none=True)
    defs["entities"][1]["omop"] = {"table": "measurement", "fields": {}}
    with pytest.raises(ValueError, match="only top-level entities"):
        EntityDefinitions.model_validate(defs)


# --- concept resolution ---------------------------------------------------------


def test_concepts_are_resolved_and_mapped_to_standard():
    builder = OmopTableBuilder(ecog_definitions(), vocabulary=FakeLookup())
    described = builder.describe_concepts().set_index("column")
    assert described.loc["measurement_concept_id", "concept_id"] == 36305384
    assert described.loc["measurement_source_concept_id", "concept_id"] == 4308014


def test_unknown_concept_code_fails_when_builder_is_created():
    defs = ecog_definitions(measurement_source_concept_id={"concept": "SNOMED:999"})
    with pytest.raises(
        ValueError, match=r"not found in the vocabulary: \['SNOMED:999'\]"
    ):
        OmopTableBuilder(defs, vocabulary=FakeLookup())


def test_non_standard_concept_needs_maps_to():
    defs = ecog_definitions(measurement_concept_id={"concept": "SNOMED:423740007"})
    with pytest.raises(ValueError, match="not a standard concept. Add 'maps_to: true'"):
        OmopTableBuilder(defs, vocabulary=FakeLookup())


def test_concept_of_wrong_domain_fails():
    defs = ecog_definitions(measurement_concept_id={"concept": "SNOMED:425389002"})
    with pytest.raises(ValueError, match="not in the 'Measurement' domain"):
        OmopTableBuilder(defs, vocabulary=FakeLookup())


def test_constant_concept_id_is_checked():
    defs = ecog_definitions(measurement_type_concept_id={"value": 4175026})
    with pytest.raises(ValueError, match="not in the 'Type Concept' domain"):
        OmopTableBuilder(defs, vocabulary=FakeLookup())


def test_missing_standard_mapping_fails():
    with pytest.raises(ValueError, match="has 0 standard 'Maps to' concepts"):
        OmopTableBuilder(ecog_definitions(), vocabulary=FakeLookup(maps_to={}))


# --- building rows --------------------------------------------------------------


@pytest.fixture
def builder():
    return OmopTableBuilder(ecog_definitions(), vocabulary=FakeLookup())


NOTE = "Assessment 05/12/2025\nOncology follow-up. ECOG 1, previously ECOG 0-1."


def test_build_measurement_rows(builder):
    defs = builder.entity_definitions
    result = extract(
        defs,
        NOTE,
        [
            ecog("e1", "ECOG 1", "1", line="L2"),
            child("e2", "05/12/2025", "e1"),
            ecog("e3", "ECOG 0-1", line="L2"),
        ],
    )
    note = ClinicalNote(text=NOTE, note_id=7, person_id=42, note_date=date(2025, 6, 1))
    tables = builder.build([note], [result], id_start=100)

    df = tables.to_dataframes(provenance=True)["measurement"]
    first, second = df.to_dict("records")
    assert first["measurement_id"] == 100
    assert first["person_id"] == 42
    assert first["measurement_concept_id"] == 36305384
    assert first["measurement_source_concept_id"] == 4308014
    assert first["measurement_source_value"] == "ECOG 1"
    assert first["measurement_date"] == date(2025, 5, 12)
    assert first["measurement_type_concept_id"] == 32858
    assert first["value_as_number"] == 1.0
    assert first["value_as_concept_id"] == 4173614
    assert first["value_source_value"] == "1"
    assert (first["_note_id"], first["_entity_id"], first["_text"]) == (
        7,
        "e1",
        "ECOG 1",
    )
    assert NOTE[first["_start"] : first["_end"]] == "ECOG 1"

    # A range has no single grade: the row is kept without a value, dated by the note.
    assert second["measurement_id"] == 101
    assert second["measurement_date"] == date(2025, 6, 1)
    assert second["measurement_source_value"] == "ECOG 0-1"
    assert pd.isna(second["value_as_number"])
    assert pd.isna(second["value_as_concept_id"])
    assert tables.issues == []


def test_unparsable_date_falls_back_to_the_note_date(builder):
    text = "ECOG on admission was 1."
    result = extract(
        builder.entity_definitions,
        text,
        [ecog("e1", "ECOG on admission was 1", "1"), child("e2", "on admission", "e1")],
    )
    note = ClinicalNote(text=text, person_id=1, note_date=date(2025, 6, 1))
    tables = builder.build([note], [result])

    assert tables.rows["measurement"][0].measurement_date == date(2025, 6, 1)
    (issue,) = tables.issues
    assert issue.reason == "unparsable_date"
    assert issue.column == "measurement_date"
    assert issue.row_id == 1
    assert "used context.note_date instead" in issue.detail


def test_partial_date_is_imputed_and_reported(builder):
    text = "In March 2024 she was ECOG 0."
    result = extract(
        builder.entity_definitions,
        text,
        [ecog("e1", "ECOG 0", "0"), child("e2", "March 2024", "e1")],
    )
    tables = builder.build([ClinicalNote(text=text, person_id=1)], [result])
    assert tables.rows["measurement"][0].measurement_date == date(2024, 3, 1)
    assert [i.reason for i in tables.issues] == ["imputed_date_parts"]


def test_label_without_concept_is_stored_as_zero(builder):
    text = "ECOG 2."
    result = extract(builder.entity_definitions, text, [ecog("e1", "ECOG 2", "2")])
    tables = builder.build(
        [ClinicalNote(text=text, person_id=1, note_date="2025-01-01")], [result]
    )
    row = tables.rows["measurement"][0]
    assert row.value_as_concept_id == 0
    assert row.value_as_number == 2.0
    assert [i.reason for i in tables.issues] == ["unmapped_label"]


def test_missing_context_is_reported_as_missing_required(builder):
    text = "ECOG 1."
    result = extract(builder.entity_definitions, text, [ecog("e1", "ECOG 1", "1")])
    tables = builder.build([text], [result])
    issues = {(i.column, i.reason) for i in tables.issues}
    assert issues == {
        ("person_id", "missing_required"),
        ("measurement_date", "missing_required"),
    }


def test_long_source_value_is_truncated(builder):
    text = (
        "ECOG performance status, as assessed by the treating oncologist today, is 1."
    )
    span = text[:-1]
    result = extract(builder.entity_definitions, text, [ecog("e1", span, "1")])
    tables = builder.build(
        [ClinicalNote(text=text, person_id=1, note_date="2025-01-01")], [result]
    )
    assert tables.rows["measurement"][0].measurement_source_value == span[:50]
    assert [i.reason for i in tables.issues] == ["truncated"]


def test_several_children_use_the_first(builder):
    text = "ECOG 1 on 01/02/2025 and 03/04/2025."
    result = extract(
        builder.entity_definitions,
        text,
        [
            ecog("e1", "ECOG 1", "1"),
            child("e2", "01/02/2025", "e1"),
            child("e3", "03/04/2025", "e1"),
        ],
    )
    tables = builder.build([ClinicalNote(text=text, person_id=1)], [result])
    assert tables.rows["measurement"][0].measurement_date == date(2025, 1, 2)
    assert [i.reason for i in tables.issues] == ["multiple_children"]


def test_dayfirst_dates():
    builder = OmopTableBuilder(
        ecog_definitions(), vocabulary=FakeLookup(), dayfirst=True
    )
    text = "ECOG 1 on 05/12/2025."
    result = extract(
        builder.entity_definitions,
        text,
        [ecog("e1", "ECOG 1", "1"), child("e2", "05/12/2025", "e1")],
    )
    tables = builder.build([ClinicalNote(text=text, person_id=1)], [result])
    assert tables.rows["measurement"][0].measurement_date == date(2025, 12, 5)


def test_notes_and_results_must_match(builder):
    with pytest.raises(ValueError, match="Got 2 notes but 1 extraction results"):
        builder.build(["a", "b"], [extract(builder.entity_definitions, "a", [])])


def test_getitem_and_empty_tables(builder):
    tables = builder.build(["x"], [extract(builder.entity_definitions, "x", [])])
    assert tables.tables == []
    with pytest.raises(KeyError, match="No rows for table 'measurement'"):
        tables["measurement"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2025-05-12", (date(2025, 5, 12), False)),
        ("05/12/2025", (date(2025, 5, 12), False)),
        ("March 2024", (date(2024, 3, 1), True)),
        ("2024", (date(2024, 1, 1), True)),
        ("on admission", (None, False)),
        ("3", (None, False)),
        ("May 12", (None, False)),
    ],
)
def test_parse_date(text, expected):
    assert _parse_date(text, dayfirst=False) == expected


# --- pipeline -------------------------------------------------------------------


def test_pipeline_extracts_and_builds_tables():
    defs = ecog_definitions()

    class FakeExtractor:
        async def __call__(self, texts):
            return [extract(defs, t, [ecog("e1", "ECOG 1", "1")]) for t in texts]

    pipeline = OmopExtractionPipeline(
        defs, extractor=FakeExtractor(), vocabulary=FakeLookup()
    )
    tables = asyncio.run(
        pipeline(
            [
                ClinicalNote(text="ECOG 1.", person_id=3, note_date="2025-01-01"),
                "ECOG 1 again.",
            ]
        )
    )
    df = tables["measurement"]
    assert df["measurement_id"].tolist() == [1, 2]
    assert df["person_id"].tolist()[0] == 3
    assert len(tables.extraction_results) == 2


def test_pipeline_needs_a_model_or_extractor():
    with pytest.raises(ValueError, match="Either model_id or extractor"):
        OmopExtractionPipeline(ecog_definitions(), vocabulary=FakeLookup())


# --- the real vocabulary database -----------------------------------------------

LOCAL_DB = Path(".aatm/omop.db")


@pytest.mark.skipif(
    not LOCAL_DB.exists(), reason="local OMOP vocabulary database not built"
)
def test_predefined_ecog_concepts_resolve_in_local_vocabulary():
    defs = EntityDefinitions.from_predefined("ecog_performance_status")
    builder = OmopTableBuilder(defs, vocabulary=SqliteConceptLookup(LOCAL_DB))
    described = builder.describe_concepts().set_index("column")
    assert described.loc["measurement_concept_id", "concept_name"] == (
        "ECOG Performance Status score"
    )
