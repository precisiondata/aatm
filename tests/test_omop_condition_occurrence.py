"""The OMOP condition_occurrence table and the predefined condition_occurrence file."""

import asyncio
from datetime import date
from pathlib import Path

import pytest
from test_omop_mapping import FakeLookup, concept
from test_omop_observation import extract, mention, note

from aatm.data_models.extractors import EntityDefinitions
from aatm.omop import (
    CodedText,
    ConditionOccurrence,
    OmopExtractionPipeline,
    OmopTableBuilder,
    SqliteConceptLookup,
)

CONDITION_CONCEPTS = [
    concept(32893, "OMOP4976963", "Confirmed diagnosis", "Condition Status", "Condition Status"),
    concept(32899, "OMOP4976969", "Preliminary diagnosis", "Condition Status", "Condition Status"),
    concept(32858, "OMOP4976931", "NLP", "Type Concept", "Type Concept"),
    concept(201826, "44054006", "Type 2 diabetes mellitus", "Condition"),
    concept(255848, "233604007", "Pneumonia", "Condition"),
]  # fmt: skip

KNOWN = {
    "type 2 diabetes": (201826, "Condition"),
    "pneumonia": (255848, "Condition"),
    "chest x-ray": (4163872, "Procedure"),
}


class Lookup(FakeLookup):
    def __init__(self):
        super().__init__(concepts=CONDITION_CONCEPTS, maps_to={})


class ConditionCoder:
    """Codes a few condition names, ignoring case; records every call."""

    def __init__(self):
        self.calls: list[tuple] = []

    def code(self, texts, domain=None, vocabularies=None):
        self.calls.append((list(texts), domain, vocabularies))
        return [
            CodedText(
                text=t,
                concept_id=KNOWN[t.lower()][0],
                domain_id=KNOWN[t.lower()][1],
                score=0.9,
            )
            if t.lower() in KNOWN
            else CodedText(text=t)
            for t in texts
        ]


def builder(coder=None):
    defs = EntityDefinitions.from_predefined("condition_occurrence")
    return OmopTableBuilder(defs, vocabulary=Lookup(), mapper=coder or ConditionCoder())


def build(text, entities, coder=None, **note_kwargs):
    b = builder(coder)
    result = extract(b.entity_definitions, text, entities)
    return b.build([note(text, **note_kwargs)], [result])


# --- the table model -----------------------------------------------------------------


def test_condition_occurrence_domain_rules():
    check = ConditionOccurrence.domain_problem
    assert check("condition_concept_id", "Condition") is None
    assert "not 'Condition'" in check("condition_concept_id", "Observation")
    assert check("condition_status_concept_id", "Condition Status") is None
    assert "not 'Type Concept'" in check("condition_type_concept_id", "Condition")


def test_condition_occurrence_required_fields_follow_the_cdm():
    assert ConditionOccurrence.required_fields == (
        "condition_occurrence_id",
        "person_id",
        "condition_concept_id",
        "condition_start_date",
        "condition_type_concept_id",
    )


def test_stop_reason_is_limited_to_20_characters():
    with pytest.raises(ValueError, match="at most 20 characters"):
        ConditionOccurrence(stop_reason="x" * 21)


def test_map_into_another_domain_fails():
    defs = EntityDefinitions(
        entities=[
            {
                "name": "Finding",
                "definition": "A finding.",
                "omop": {
                    "table": "condition_occurrence",
                    "fields": {
                        "condition_concept_id": {
                            "from": "text",
                            "map": {"domain": "Observation"},
                        }
                    },
                },
            }
        ]
    )
    with pytest.raises(ValueError, match="'Observation' domain, not 'Condition'"):
        OmopTableBuilder(defs, vocabulary=Lookup(), mapper=ConditionCoder())


# --- the predefined definition -------------------------------------------------------

NOTE = (
    "Type 2 diabetes since 2019.\n"
    "Probable pneumonia.\n"
    "Pneumonia from March 2024 to April 2024."
)


def test_condition_rows():
    coder = ConditionCoder()
    tables = build(
        NOTE,
        [
            mention("e1", "condition_name", "Type 2 diabetes", status="confirmed"),
            mention("e2", "verbatim_start_date", "2019", parents=["e1"]),
            mention(
                "e3", "condition_name", "pneumonia", line="L2", status="preliminary"
            ),
            mention("e4", "condition_name", "Pneumonia", line="L3", status="confirmed"),
            mention("e5", "verbatim_start_date", "March 2024", "L3", parents=["e4"]),
            mention("e6", "verbatim_end_date", "April 2024", "L3", parents=["e4"]),
        ],
        coder=coder,
        visit_occurrence_id=9,
    )
    diabetes, suspected, episode = tables.rows["condition_occurrence"]

    assert diabetes.condition_occurrence_id == 1
    assert diabetes.person_id == 5
    assert diabetes.condition_concept_id == 201826
    assert diabetes.condition_source_value == "Type 2 diabetes"
    assert diabetes.condition_start_date == date(2019, 1, 1)
    assert diabetes.condition_end_date is None
    assert diabetes.condition_type_concept_id == 32858
    assert diabetes.condition_status_concept_id == 32893
    assert diabetes.condition_status_source_value == "confirmed"
    assert diabetes.visit_occurrence_id == 9

    # No date in the note: the note date, when the condition is known to be present.
    assert suspected.condition_start_date == date(2025, 6, 1)
    assert suspected.condition_status_concept_id == 32899

    assert episode.condition_start_date == date(2024, 3, 1)
    assert episode.condition_end_date == date(2024, 4, 1)
    assert episode.condition_status_concept_id == 32893

    # Coded once per distinct text, from the Condition domain, any vocabulary.
    assert coder.calls == [
        (["Pneumonia", "Type 2 diabetes", "pneumonia"], "Condition", None)
    ]
    assert episode.condition_concept_id == 255848
    assert {(i.row_id, i.column, i.reason) for i in tables.issues} == {
        (1, "condition_start_date", "imputed_date_parts"),
        (3, "condition_start_date", "imputed_date_parts"),
        (3, "condition_end_date", "imputed_date_parts"),
    }


def test_only_an_end_date_is_used_as_the_start():
    text = "pneumonia until 2024-04-10."
    tables = build(
        text,
        [
            mention("e1", "condition_name", "pneumonia", status="confirmed"),
            mention("e2", "verbatim_end_date", "2024-04-10", parents=["e1"]),
        ],
    )
    (row,) = tables.rows["condition_occurrence"]
    # Not the note date (2025-06-01), which would come after the end.
    assert row.condition_start_date == date(2024, 4, 10)
    assert row.condition_end_date == date(2024, 4, 10)
    assert tables.issues == []


def test_relative_start_date_falls_back_to_the_note_date():
    text = "pneumonia since yesterday"
    tables = build(
        text,
        [
            mention("e1", "condition_name", "pneumonia", status="confirmed"),
            mention("e2", "verbatim_start_date", "yesterday", parents=["e1"]),
        ],
    )
    (row,) = tables.rows["condition_occurrence"]
    assert row.condition_start_date == date(2025, 6, 1)
    (issue,) = tables.issues
    assert issue.reason == "unparsable_date"
    assert "used context.note_date instead" in issue.detail


def test_end_before_start_is_dropped_and_reported():
    text = "pneumonia from 2024-05-01 to 2024-04-01"
    tables = build(
        text,
        [
            mention("e1", "condition_name", "pneumonia", status="confirmed"),
            mention("e2", "verbatim_start_date", "2024-05-01", parents=["e1"]),
            mention("e3", "verbatim_end_date", "2024-04-01", parents=["e1"]),
        ],
    )
    (row,) = tables.rows["condition_occurrence"]
    assert row.condition_start_date == date(2024, 5, 1)
    assert row.condition_end_date is None
    assert [(i.column, i.reason) for i in tables.issues] == [
        ("condition_end_date", "invalid_value")
    ]
    assert "before condition_start_date 2024-05-01" in tables.issues[0].detail


def test_concept_outside_the_condition_domain_is_reported():
    tables = build(
        "chest x-ray",
        [mention("e1", "condition_name", "chest x-ray", status="confirmed")],
    )
    (row,) = tables.rows["condition_occurrence"]
    assert row.condition_concept_id == 4163872
    assert [(i.column, i.reason) for i in tables.issues] == [
        ("condition_concept_id", "domain_mismatch")
    ]


@pytest.mark.parametrize("attributes", [{}, {"status": "resolved"}])
def test_condition_without_a_valid_status_is_rejected(attributes):
    # Past and resolved conditions belong in OBSERVATION, not condition_occurrence:
    # there is no "resolved" status.
    text = "pneumonia"
    b = builder()
    result = extract(
        b.entity_definitions,
        text,
        [mention("e1", "condition_name", "pneumonia", **attributes)],
    )
    assert result.validated_entities == []
    assert b.build([note(text)], [result]).rows == {}


def test_condition_occurrence_needs_a_mapper():
    with pytest.raises(ValueError, match="needs a terminology mapper"):
        OmopTableBuilder(
            EntityDefinitions.from_predefined("condition_occurrence"),
            vocabulary=Lookup(),
        )


def test_pipeline_dataframe():
    class Extractor:
        async def __call__(self, texts):
            defs = pipeline.entity_definitions
            return [
                extract(
                    defs,
                    t,
                    [mention("e1", "condition_name", "pneumonia", status="confirmed")],
                )
                for t in texts
            ]

    pipeline = OmopExtractionPipeline(
        "condition_occurrence",
        extractor=Extractor(),
        vocabulary=Lookup(),
        mapper=ConditionCoder(),
    )
    tables = asyncio.run(pipeline([note("pneumonia")]))
    df = tables["condition_occurrence"]
    assert list(df.columns) == list(ConditionOccurrence.model_fields)
    assert df["condition_concept_id"].tolist() == [255848]
    assert str(df["condition_concept_id"].dtype) == "Int64"


def test_predefined_prompt_has_no_omop_details():
    prompt = EntityDefinitions.from_predefined("condition_occurrence").to_prompt_str()
    for leaked in ("OMOP4976", "Condition Status", "omop", "concept_id", "32858"):
        assert leaked not in prompt


@pytest.mark.skipif(
    not Path(".aatm/omop.db").exists(),
    reason="local OMOP vocabulary database not built",
)
def test_predefined_condition_concepts_resolve_in_local_vocabulary():
    b = OmopTableBuilder(
        EntityDefinitions.from_predefined("condition_occurrence"),
        vocabulary=SqliteConceptLookup(".aatm/omop.db"),
        mapper=ConditionCoder(),
    )
    assert set(b.describe_concepts().concept_name) == {
        "Confirmed diagnosis",
        "Preliminary diagnosis",
    }
