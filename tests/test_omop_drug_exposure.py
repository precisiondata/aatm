"""The OMOP drug_exposure table, the predefined drug_exposure file, and the
`as: days`, `period_start` and `when` options it relies on."""

import asyncio
from datetime import date
from pathlib import Path

import pytest
from test_omop_mapping import FakeLookup, concept
from test_omop_observation import extract, mention, note

from aatm.data_models.extractors import EntityDefinitions
from aatm.omop import (
    CodedText,
    DrugExposure,
    FieldSource,
    OmopExtractionPipeline,
    OmopMapping,
    OmopTableBuilder,
    SqliteConceptLookup,
)
from aatm.omop.builder import _parse_days

ROUTES = {
    "26643006": (4132161, "Oral"),
    "37839007": (4292110, "Sublingual"),
    "47625008": (4171047, "Intravenous"),
    "78421000": (4302612, "Intramuscular"),
    "34206005": (4142048, "Subcutaneous"),
    "6064005": (4263689, "Topical"),
    "45890007": (4262099, "Transdermal"),
    "447694001": (40486069, "Respiratory tract"),
    "46713006": (4262914, "Nasal"),
    "54485002": (4184451, "Ophthalmic"),
    "10547007": (4023156, "Otic"),
    "37161004": (4290759, "Rectal"),
    "16857009": (4057765, "Vaginal"),
}
DRUG_CONCEPTS = [
    concept(32858, "OMOP4976931", "NLP", "Type Concept", "Type Concept"),
    *(concept(cid, code, name, "Route") for code, (cid, name) in ROUTES.items()),
]
KNOWN = {
    "amoxicillin 500 mg": (19073183, "Drug"),
    "lisinopril 10 mg": (1308251, "Drug"),
    "ceftriaxone 1 g": (1777806, "Drug"),
    "aspirin": (1112807, "Drug"),
}


class Lookup(FakeLookup):
    def __init__(self):
        super().__init__(concepts=DRUG_CONCEPTS, maps_to={})


class DrugCoder:
    """Codes a few drugs, ignoring case; records every call."""

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
    defs = EntityDefinitions.from_predefined("drug_exposure")
    return OmopTableBuilder(defs, vocabulary=Lookup(), coder=coder or DrugCoder())


def build(text, entities, coder=None, **note_kwargs):
    b = builder(coder)
    result = extract(b.entity_definitions, text, entities)
    kwargs = {"visit_occurrence_id": 7, "provider_id": 3, **note_kwargs}
    return b.build([note(text, **kwargs)], [result])


def drug(id, text, kind, line="L1", **attributes):
    return mention(id, "drug_name", text, line=line, kind=kind, **attributes)


def child(id, key, text, parent, line="L1"):
    return mention(id, key, text, line=line, parents=[parent])


# --- durations -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, days",
    [
        ("7 days", 7),
        ("x7d", 7),
        ("2 weeks", 14),
        ("a 10-day course", 10),
        ("1 month", 30),
        ("14 dias", 14),
        ("duas semanas", 14),
        ("seven days", 7),
        ("once a day for 7 days", 7),  # digits win over "a day"
        ("daily", None),
        ("500 mg", None),
        ("10 days then 5 days", None),  # two durations: ambiguous
    ],
)
def test_parse_days(text, days):
    assert _parse_days(text) == days


# --- period_start and when -----------------------------------------------------------


def test_period_start_needs_a_number_of_days():
    with pytest.raises(ValueError, match="'period_start' needs a number of days"):
        FieldSource.model_validate({"from": "child.x", "period_start": "a"})
    with pytest.raises(ValueError, match="'period_start' needs a number of days"):
        FieldSource.model_validate({"value": "30", "period_start": "a"})


def test_period_start_must_be_mapped_before():
    with pytest.raises(ValueError, match="must be mapped before it"):
        OmopMapping.model_validate(
            {
                "table": "drug_exposure",
                "fields": {
                    "drug_exposure_end_date": {
                        "value": 1,
                        "period_start": "drug_exposure_start_date",
                    },
                    "drug_exposure_start_date": {"from": "context.note_date"},
                },
            }
        )


def test_when_takes_attribute_labels():
    with pytest.raises(ValueError, match="'when' takes attribute labels"):
        FieldSource.model_validate({"value": 1, "when": {"child.x": ["a"]}})


def test_when_labels_are_checked_against_the_attribute():
    defs = EntityDefinitions.from_predefined("drug_exposure")
    entity = defs.entities[0].model_dump(by_alias=True, exclude_none=True)
    entity["omop"]["fields"]["provider_id"] = {
        "from": "context.provider_id",
        "when": {"attribute.kind": ["ordered"]},
    }
    with pytest.raises(ValueError, match="'when' label 'ordered' is not a label"):
        EntityDefinitions(entities=[entity, *defs.entities[1:]])


# --- the table model -----------------------------------------------------------------


def test_drug_exposure_domain_rules():
    check = DrugExposure.domain_problem
    assert check("drug_concept_id", "Drug") is None
    assert "not 'Drug'" in check("drug_concept_id", "Condition")
    assert check("route_concept_id", "Route") is None
    assert "not 'Type Concept'" in check("drug_type_concept_id", "Drug")


def test_drug_exposure_required_fields_follow_the_cdm():
    assert DrugExposure.required_fields == (
        "drug_exposure_id",
        "person_id",
        "drug_concept_id",
        "drug_exposure_start_date",
        "drug_exposure_end_date",
        "drug_type_concept_id",
    )


def test_negative_days_supply_is_rejected():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        DrugExposure(days_supply=-5)


# --- the predefined definition -------------------------------------------------------

NOTE = (
    "Start amoxicillin 500 mg PO TID for 7 days.\n"
    "Home meds: lisinopril 10 mg daily since 2020.\n"
    "Ceftriaxone 1 g IV given in the ED.\n"
    "Rx aspirin 81 mg daily."
)


def test_drug_rows():
    coder = DrugCoder()
    tables = build(
        NOTE,
        [
            drug("e1", "amoxicillin 500 mg", "prescribed", route="oral"),
            child("e2", "drug_duration", "7 days", "e1"),
            child("e3", "sig", "PO TID for 7 days", "e1"),
            drug("e4", "lisinopril 10 mg", "taking", line="L2"),
            child("e5", "verbatim_start_date", "2020", "e4", line="L2"),
            drug(
                "e6", "Ceftriaxone 1 g", "administered", line="L3", route="intravenous"
            ),
            drug("e7", "aspirin", "prescribed", line="L4"),
        ],
        coder=coder,
    )
    amoxicillin, lisinopril, ceftriaxone, aspirin = tables.rows["drug_exposure"]

    assert amoxicillin.drug_exposure_id == 1
    assert amoxicillin.person_id == 5
    assert amoxicillin.drug_concept_id == 19073183
    assert amoxicillin.drug_source_value == "amoxicillin 500 mg"
    assert amoxicillin.drug_type_concept_id == 32858
    assert amoxicillin.route_concept_id == 4132161
    assert amoxicillin.route_source_value == "oral"
    assert amoxicillin.sig == "PO TID for 7 days"
    # THEMIS: start + duration - 1.
    assert amoxicillin.drug_exposure_start_date == date(2025, 6, 1)
    assert amoxicillin.drug_exposure_end_date == date(2025, 6, 7)
    assert amoxicillin.verbatim_end_date is None
    assert amoxicillin.days_supply is None  # never computed (CDM)
    # Prescribed in this visit: the visit and the prescriber.
    assert amoxicillin.visit_occurrence_id == 7
    assert amoxicillin.provider_id == 3

    # A medication-list entry: known to be taken from 2020 to the note date, and not
    # initiated in this visit.
    assert lisinopril.drug_exposure_start_date == date(2020, 1, 1)
    assert lisinopril.drug_exposure_end_date == date(2025, 6, 1)
    assert lisinopril.visit_occurrence_id is None
    assert lisinopril.provider_id is None
    assert lisinopril.route_concept_id is None

    # An administration: end date = start date.
    assert ceftriaxone.drug_exposure_end_date == date(2025, 6, 1)
    assert ceftriaxone.route_concept_id == 4171047
    assert ceftriaxone.visit_occurrence_id == 7

    # A written prescription without a duration: start + 29 days (THEMIS).
    assert aspirin.drug_exposure_end_date == date(2025, 6, 30)

    assert coder.calls == [
        (
            ["Ceftriaxone 1 g", "amoxicillin 500 mg", "aspirin", "lisinopril 10 mg"],
            "Drug",
            None,
        )
    ]
    assert [(i.row_id, i.column, i.reason) for i in tables.issues] == [
        (2, "drug_exposure_start_date", "imputed_date_parts")
    ]


def test_stated_end_date_is_used_and_kept_verbatim():
    text = "amoxicillin 500 mg until 2025-06-10"
    tables = build(
        text,
        [
            drug("e1", "amoxicillin 500 mg", "prescribed"),
            child("e2", "verbatim_end_date", "2025-06-10", "e1"),
        ],
    )
    (row,) = tables.rows["drug_exposure"]
    assert row.drug_exposure_end_date == date(2025, 6, 10)
    assert row.verbatim_end_date == date(2025, 6, 10)
    assert tables.issues == []


def test_only_a_stop_date_is_used_for_both_dates():
    # Stopped before the note, with no start date: not a start after the stop.
    text = "lisinopril 10 mg stopped 2025-05-01"
    tables = build(
        text,
        [
            drug("e1", "lisinopril 10 mg", "taking"),
            child("e2", "verbatim_end_date", "2025-05-01", "e1"),
        ],
    )
    (row,) = tables.rows["drug_exposure"]
    assert row.drug_exposure_start_date == date(2025, 5, 1)
    assert row.drug_exposure_end_date == date(2025, 5, 1)
    assert row.verbatim_end_date == date(2025, 5, 1)
    (issue,) = tables.issues
    assert (issue.column, issue.reason) == ("drug_exposure_start_date", "single_date")


def test_stated_end_before_stated_start_falls_back_to_the_next_rule():
    text = "amoxicillin 500 mg from 2025-05-10 to 2025-05-01"
    tables = build(
        text,
        [
            drug("e1", "amoxicillin 500 mg", "prescribed"),
            child("e2", "verbatim_start_date", "2025-05-10", "e1"),
            child("e3", "verbatim_end_date", "2025-05-01", "e1"),
        ],
    )
    (row,) = tables.rows["drug_exposure"]
    assert row.drug_exposure_start_date == date(2025, 5, 10)
    assert row.drug_exposure_end_date == date(2025, 6, 8)  # start + 29
    (issue,) = tables.issues
    assert (issue.column, issue.reason) == ("drug_exposure_end_date", "invalid_value")
    assert issue.detail == (
        "2025-05-01 is before drug_exposure_start_date 2025-05-10; "
        "used the constant 30 days from drug_exposure_start_date instead"
    )


def test_unreadable_duration_falls_back_and_is_reported():
    text = "amoxicillin 500 mg for a while"
    tables = build(
        text,
        [
            drug("e1", "amoxicillin 500 mg", "prescribed"),
            child("e2", "drug_duration", "a while", "e1"),
        ],
    )
    (row,) = tables.rows["drug_exposure"]
    assert row.drug_exposure_end_date == date(2025, 6, 30)
    (issue,) = tables.issues
    assert issue.reason == "invalid_value"
    assert "'a while' is not a duration" in issue.detail


def test_drug_without_kind_is_rejected():
    text = "aspirin"
    b = builder()
    result = extract(
        b.entity_definitions, text, [mention("e1", "drug_name", "aspirin")]
    )
    assert result.validated_entities == []


def test_drug_exposure_needs_a_mapper():
    with pytest.raises(ValueError, match="needs a coder"):
        OmopTableBuilder(
            EntityDefinitions.from_predefined("drug_exposure"), vocabulary=Lookup()
        )


def test_pipeline_dataframe():
    class Extractor:
        async def __call__(self, texts):
            defs = pipeline.entity_definitions
            return [
                extract(defs, t, [drug("e1", "aspirin", "administered")]) for t in texts
            ]

    pipeline = OmopExtractionPipeline(
        "drug_exposure",
        extractor=Extractor(),
        vocabulary=Lookup(),
        coder=DrugCoder(),
    )
    tables = asyncio.run(pipeline([note("aspirin")]))
    df = tables["drug_exposure"]
    assert list(df.columns) == list(DrugExposure.model_fields)
    assert df["drug_concept_id"].tolist() == [1112807]


def test_predefined_prompt_has_no_omop_details():
    prompt = EntityDefinitions.from_predefined("drug_exposure").to_prompt_str()
    for leaked in ("SNOMED", "omop", "concept_id", "32858", "period_start", "THEMIS"):
        assert leaked not in prompt


@pytest.mark.skipif(
    not Path(".aatm/omop.db").exists(),
    reason="local OMOP vocabulary database not built",
)
def test_predefined_drug_concepts_resolve_in_local_vocabulary():
    b = OmopTableBuilder(
        EntityDefinitions.from_predefined("drug_exposure"),
        vocabulary=SqliteConceptLookup(".aatm/omop.db"),
        coder=DrugCoder(),
    )
    concepts = b.describe_concepts()
    assert set(concepts.domain_id) == {"Route"}
    assert len(concepts) == len(ROUTES)
