"""The OMOP device_exposure table and the predefined device_exposure file."""

import asyncio
from datetime import date
from pathlib import Path

import pytest
from test_omop_mapping import FakeLookup, concept
from test_omop_observation import extract, mention, note

from aatm.data_models.extractors import EntityDefinitions
from aatm.omop import (
    CodedText,
    DeviceExposure,
    OmopExtractionPipeline,
    OmopTableBuilder,
    SqliteConceptLookup,
)

DEVICE_CONCEPTS = [
    concept(32858, "OMOP4976931", "NLP", "Type Concept", "Type Concept"),
]
KNOWN = {
    "drug-eluting stents": (4131681, "Device"),
    "pacemaker": (4030875, "Device"),
    "foley catheter": (40625746, "Device"),
    "crutches": (4251933, "Device"),
    "nicotine patch": (19005965, "Drug"),
}


class Lookup(FakeLookup):
    def __init__(self):
        super().__init__(concepts=DEVICE_CONCEPTS, maps_to={})


class DeviceCoder:
    """Codes a few devices, ignoring case; records every call."""

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
    defs = EntityDefinitions.from_predefined("device_exposure")
    return OmopTableBuilder(defs, vocabulary=Lookup(), mapper=coder or DeviceCoder())


def build(text, entities, coder=None):
    b = builder(coder)
    result = extract(b.entity_definitions, text, entities)
    return b.build(
        [note(text, visit_occurrence_id=7, provider_id=3, visit_detail_id=8)],
        [result],
    )


def device(id, text, kind, line="L1"):
    return mention(id, "device_name", text, line=line, kind=kind)


def child(id, key, text, parent, line="L1"):
    return mention(id, key, text, line=line, parents=[parent])


# --- the table model -----------------------------------------------------------------


def test_device_exposure_domain_rules():
    check = DeviceExposure.domain_problem
    assert check("device_concept_id", "Device") is None
    assert "not 'Device'" in check("device_concept_id", "Drug")
    assert check("unit_concept_id", "Unit") is None
    assert "not 'Type Concept'" in check("device_type_concept_id", "Device")


def test_device_exposure_required_fields_follow_the_cdm():
    # Unlike drug_exposure, the end date is optional.
    assert DeviceExposure.required_fields == (
        "device_exposure_id",
        "person_id",
        "device_concept_id",
        "device_exposure_start_date",
        "device_type_concept_id",
    )


def test_udi_columns_hold_255_characters():
    assert DeviceExposure(unique_device_id="x" * 255).unique_device_id == "x" * 255
    with pytest.raises(ValueError, match="at most 255 characters"):
        DeviceExposure(production_id="x" * 256)


# --- the predefined definition -------------------------------------------------------

NOTE = (
    "s/p PCI with 2 drug-eluting stents in 2021. Pacemaker since 2019.\n"
    "Foley catheter inserted today. Walks with crutches."
)


def test_device_rows():
    coder = DeviceCoder()
    tables = build(
        NOTE,
        [
            device("e1", "drug-eluting stents", "existing"),
            child("e2", "device_quantity", "2", "e1"),
            child("e3", "verbatim_start_date", "2021", "e1"),
            device("e4", "Pacemaker", "existing"),
            child("e5", "verbatim_start_date", "2019", "e4"),
            device("e6", "Foley catheter", "placed", line="L2"),
            device("e7", "crutches", "existing", line="L2"),
        ],
        coder=coder,
    )
    stents, pacemaker, foley, crutches = tables.rows["device_exposure"]

    assert stents.device_exposure_id == 1
    assert stents.person_id == 5
    assert stents.device_concept_id == 4131681
    assert stents.device_source_value == "drug-eluting stents"
    assert stents.device_exposure_start_date == date(2021, 1, 1)
    assert stents.device_exposure_end_date is None  # never inferred (CDM)
    assert stents.quantity == 2
    assert stents.device_type_concept_id == 32858
    # Existing devices were not initiated in this visit.
    assert stents.visit_occurrence_id is None
    assert stents.provider_id is None

    assert pacemaker.device_concept_id == 4030875
    assert pacemaker.quantity == 1  # the CDM's value when no quantity is given

    # Placed in this encounter: the note date, visit and provider.
    assert foley.device_exposure_start_date == date(2025, 6, 1)
    assert foley.visit_occurrence_id == 7
    assert foley.visit_detail_id == 8
    assert foley.provider_id == 3

    assert crutches.device_exposure_start_date == date(2025, 6, 1)
    assert crutches.visit_occurrence_id is None

    assert coder.calls == [
        (
            ["Foley catheter", "Pacemaker", "crutches", "drug-eluting stents"],
            "Device",
            None,
        )
    ]
    assert [(i.row_id, i.reason) for i in tables.issues] == [
        (1, "imputed_date_parts"),
        (2, "imputed_date_parts"),
    ]


def test_removal_date_alone_is_used_for_both_dates():
    text = "Foley catheter removed on 2025-05-20"
    tables = build(
        text,
        [
            device("e1", "Foley catheter", "existing"),
            child("e2", "verbatim_end_date", "2025-05-20", "e1"),
        ],
    )
    (row,) = tables.rows["device_exposure"]
    assert row.device_exposure_start_date == date(2025, 5, 20)
    assert row.device_exposure_end_date == date(2025, 5, 20)
    assert [(i.column, i.reason) for i in tables.issues] == [
        ("device_exposure_start_date", "single_date")
    ]


def test_quantity_without_digits_falls_back_to_1():
    text = "two crutches"
    tables = build(
        text,
        [
            device("e1", "crutches", "existing"),
            child("e2", "device_quantity", "two", "e1"),
        ],
    )
    (row,) = tables.rows["device_exposure"]
    assert row.quantity == 1
    (issue,) = tables.issues
    assert issue.reason == "invalid_value"
    assert "used the constant 1 instead" in issue.detail


def test_concept_outside_the_device_domain_is_reported():
    tables = build("nicotine patch", [device("e1", "nicotine patch", "existing")])
    (row,) = tables.rows["device_exposure"]
    assert row.device_concept_id == 19005965
    assert [(i.column, i.reason) for i in tables.issues] == [
        ("device_concept_id", "domain_mismatch")
    ]


def test_device_without_kind_is_rejected():
    b = builder()
    result = extract(
        b.entity_definitions, "crutches", [mention("e1", "device_name", "crutches")]
    )
    assert result.validated_entities == []


def test_device_exposure_needs_a_mapper():
    with pytest.raises(ValueError, match="needs a terminology mapper"):
        OmopTableBuilder(
            EntityDefinitions.from_predefined("device_exposure"), vocabulary=Lookup()
        )


def test_pipeline_dataframe():
    class Extractor:
        async def __call__(self, texts):
            defs = pipeline.entity_definitions
            return [
                extract(defs, t, [device("e1", "crutches", "existing")]) for t in texts
            ]

    pipeline = OmopExtractionPipeline(
        "device_exposure",
        extractor=Extractor(),
        vocabulary=Lookup(),
        mapper=DeviceCoder(),
    )
    tables = asyncio.run(pipeline([note("crutches")]))
    df = tables["device_exposure"]
    assert list(df.columns) == list(DeviceExposure.model_fields)
    assert df["device_concept_id"].tolist() == [4251933]
    assert df["quantity"].tolist() == [1]


def test_predefined_prompt_has_no_omop_details():
    prompt = EntityDefinitions.from_predefined("device_exposure").to_prompt_str()
    for leaked in ("SNOMED", "omop", "concept_id", "32858", "CDM"):
        assert leaked not in prompt


@pytest.mark.skipif(
    not Path(".aatm/omop.db").exists(),
    reason="local OMOP vocabulary database not built",
)
def test_predefined_device_definition_resolves_in_local_vocabulary():
    # The only fixed concept is the NLP type concept, a constant checked on load.
    OmopTableBuilder(
        EntityDefinitions.from_predefined("device_exposure"),
        vocabulary=SqliteConceptLookup(".aatm/omop.db"),
        mapper=DeviceCoder(),
    )
