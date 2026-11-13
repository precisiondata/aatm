"""The OMOP observation table: smoking status, family and medical history."""

import asyncio
from datetime import date
from pathlib import Path

import pytest
from test_omop_mapping import FakeLookup, concept

from aatm.data_models.extractors import (
    EntityDefinitions,
    ExtractedEntityCandidates,
    validate_extracted_entity,
)
from aatm.omop import (
    ClinicalNote,
    CodedText,
    Observation,
    OmopExtractionPipeline,
    OmopTableBuilder,
    SqliteConceptLookup,
)

OBSERVATION_CONCEPTS = [
    concept(43054909, "72166-2", "Tobacco smoking status", "Observation", "LOINC"),
    concept(45879404, "LA18978-9", "Never smoker", "Meas Value", "LOINC"),
    concept(45883458, "LA15920-4", "Former smoker", "Meas Value", "LOINC"),
    concept(45881517, "LA18976-3", "Current every day smoker", "Meas Value", "LOINC"),
    concept(45884037, "LA18977-1", "Current some day smoker", "Meas Value", "LOINC"),
    concept(
        45881518, "LA18979-7", "Smoker, current status unknown", "Meas Value", "LOINC"
    ),
    concept(
        3003421,
        "8664-5",
        "Cigarettes smoked total (pack per year)",
        "Observation",
        "LOINC",
    ),
    concept(4167217, "416471007", "Family history of clinical finding", "Observation"),
    concept(4248584, "72705000", "Mother", "Relationship"),
    concept(4283070, "66839005", "Father", "Relationship"),
    concept(201826, "44054006", "Type 2 diabetes mellitus", "Condition"),
    concept(32858, "OMOP4976931", "NLP", "Type Concept", "Type Concept"),
    concept(
        3002660, "11348-0", "History of Past illness Narrative", "Observation", "LOINC"
    ),
]


class Lookup(FakeLookup):
    def __init__(self):
        super().__init__(concepts=OBSERVATION_CONCEPTS, maps_to={})


class ConditionCoder:
    """Maps 'type 2 diabetes' to its SNOMED condition; records every call."""

    def __init__(self):
        self.calls: list[tuple] = []

    def code(self, texts, domain=None, vocabularies=None):
        self.calls.append((list(texts), domain, vocabularies))
        known = {"type 2 diabetes": (201826, "Condition")}
        return [
            CodedText(text=t, concept_id=known[t][0], domain_id=known[t][1], score=0.8)
            if t in known
            else CodedText(text=t)
            for t in texts
        ]


def extract(definitions, text, entities):
    candidates = ExtractedEntityCandidates.model_validate({"entities": entities})
    return validate_extracted_entity(text, definitions, candidates)


def note(text, **kwargs):
    return ClinicalNote(text=text, person_id=5, note_date=date(2025, 6, 1), **kwargs)


def mention(id, key, text, line="L1", parents=None, **attributes):
    return {
        "id": id,
        "text": text,
        "key": key,
        "line": line,
        "parents": parents or [],
        "attributes": [{"key": k, "value": v} for k, v in attributes.items()],
    }


def smoking_builder():
    defs = EntityDefinitions.from_predefined("smoking_status")
    return OmopTableBuilder(defs, vocabulary=Lookup())


def family_builder(coder=None):
    defs = EntityDefinitions.from_predefined("family_history")
    lookup = Lookup()
    # Relatives not used by these tests still need to resolve.
    for i, code in enumerate(
        [
            "40683002", "27733009", "70924004", "375005", "66089001", "65616008",
            "67822003", "113157001", "34871008", "38312007", "25211005", "38048003",
            "55538000", "125679009",
        ]
    ):  # fmt: skip
        lookup.concepts[9000 + i] = concept(
            9000 + i, code, f"Relative {i}", "Relationship"
        )
    return OmopTableBuilder(defs, vocabulary=lookup, coder=coder or ConditionCoder())


# --- the table model -----------------------------------------------------------------


def test_observation_domain_rules():
    assert Observation.domain_problem("observation_concept_id", "Observation") is None
    assert Observation.domain_problem("observation_concept_id", "Relationship") is None
    assert "goes to another table" in Observation.domain_problem(
        "observation_concept_id", "Condition"
    )
    # value_as_concept_id holds the condition for family history: any domain.
    assert Observation.domain_problem("value_as_concept_id", "Condition") is None
    assert Observation.domain_problem("unit_concept_id", "Unit") is None
    assert "not 'Unit'" in Observation.domain_problem("unit_concept_id", "Meas Value")


def test_value_as_string_is_limited_to_60_characters():
    with pytest.raises(ValueError, match="at most 60 characters"):
        Observation(value_as_string="x" * 61)


def condition_as_observation():
    return EntityDefinitions(
        entities=[
            {
                "name": "Finding",
                "definition": "A finding.",
                "omop": {
                    "table": "observation",
                    "fields": {
                        "observation_concept_id": {"concept": "SNOMED:44054006"},
                        "unit_concept_id": {"value": 0},
                    },
                },
            }
        ]
    )


def test_condition_concept_cannot_be_an_observation():
    with pytest.raises(
        ValueError, match="'Condition' domain, which goes to another table"
    ):
        OmopTableBuilder(condition_as_observation(), vocabulary=Lookup())


def test_map_into_an_excluded_domain_fails():
    defs = EntityDefinitions(
        entities=[
            {
                "name": "Finding",
                "definition": "A finding.",
                "omop": {
                    "table": "observation",
                    "fields": {
                        "observation_concept_id": {
                            "from": "text",
                            "map": {"domain": "Drug"},
                        }
                    },
                },
            }
        ]
    )
    with pytest.raises(ValueError, match="'map' would retrieve concepts in the 'Drug'"):
        OmopTableBuilder(defs, vocabulary=Lookup(), coder=ConditionCoder())


def test_constant_zero_is_not_looked_up():
    defs = condition_as_observation()
    defs.entities[0].omop.fields.pop("observation_concept_id")
    builder = OmopTableBuilder(defs, vocabulary=Lookup())
    tables = builder.build(
        [note("x")], [extract(defs, "x", [mention("e1", "finding", "x")])]
    )
    assert tables.rows["observation"][0].unit_concept_id == 0


# --- smoking status ------------------------------------------------------------------

SMOKING_NOTE = "Former smoker, 30 pack-years, quit 2015.\nNon-smoker per husband."


def test_smoking_status_and_pack_years_rows():
    builder = smoking_builder()
    result = extract(
        builder.entity_definitions,
        SMOKING_NOTE,
        [
            mention("e1", "smoking_status", "Former smoker", status="former"),
            mention("e2", "pack_years", "30 pack-years"),
            mention("e3", "pack_years_value", "30", parents=["e2"]),
            mention(
                "e4", "smoking_status", "Non-smoker", line="L2", status="not_current"
            ),
        ],
    )
    tables = builder.build([note(SMOKING_NOTE, visit_occurrence_id=9)], [result])
    status, pack_years, not_current = tables.rows["observation"]

    assert status.observation_concept_id == 43054909
    assert status.value_as_concept_id == 45883458
    assert status.value_source_value == "former"
    assert status.observation_source_value == "Former smoker"
    assert status.observation_date == date(2025, 6, 1)
    assert status.observation_type_concept_id == 32858
    assert status.visit_occurrence_id == 9

    assert pack_years.observation_concept_id == 3003421
    assert pack_years.value_as_number == 30.0
    assert pack_years.unit_concept_id == 0
    assert pack_years.unit_source_value == "pack-years"

    # "Non-smoker" has no LOINC answer: 0, the CDM's "present but unmapped".
    assert not_current.value_as_concept_id == 0
    assert [(i.row_id, i.reason) for i in tables.issues] == [(3, "unmapped_label")]


def test_smoking_status_without_label_is_rejected():
    builder = smoking_builder()
    result = extract(
        builder.entity_definitions,
        "Smoking: see above.",
        [mention("e1", "smoking_status", "Smoking")],
    )
    assert result.validated_entities == []
    assert builder.build([note("Smoking: see above.")], [result]).rows == {}


# --- family history ------------------------------------------------------------------

FAMILY_NOTE = "Father with type 2 diabetes. Mother with lupus."


def test_family_history_rows():
    coder = ConditionCoder()
    builder = family_builder(coder)
    result = extract(
        builder.entity_definitions,
        FAMILY_NOTE,
        [
            mention("e1", "family_history", "type 2 diabetes", relative="father"),
            mention("e2", "family_history", "lupus", relative="mother"),
        ],
    )
    tables = builder.build([note(FAMILY_NOTE)], [result])
    diabetes, lupus = tables.rows["observation"]

    assert diabetes.observation_concept_id == 4167217
    assert diabetes.value_as_concept_id == 201826
    assert diabetes.value_source_value == "type 2 diabetes"
    assert diabetes.qualifier_concept_id == 4283070
    assert diabetes.qualifier_source_value == "father"
    assert lupus.value_as_concept_id == 0
    assert lupus.qualifier_concept_id == 4248584

    # Any standard Condition concept qualifies, whatever its vocabulary; each text once.
    assert coder.calls == [(["lupus", "type 2 diabetes"], "Condition", None)]
    assert [(i.row_id, i.column, i.reason) for i in tables.issues] == [
        (2, "value_as_concept_id", "unmapped")
    ]


def test_family_history_needs_a_mapper():
    with pytest.raises(ValueError, match="needs a coder"):
        OmopTableBuilder(
            EntityDefinitions.from_predefined("family_history"), vocabulary=Lookup()
        )


# --- medical history -----------------------------------------------------------------

HISTORY_NOTE = "History of type 2 diabetes in remission. H/o lupus."


def history_builder(coder=None):
    defs = EntityDefinitions.from_predefined("medical_history")
    return OmopTableBuilder(defs, vocabulary=Lookup(), coder=coder or ConditionCoder())


def test_medical_history_rows():
    coder = ConditionCoder()
    builder = history_builder(coder)
    result = extract(
        builder.entity_definitions,
        HISTORY_NOTE,
        [
            mention("e1", "medical_history", "type 2 diabetes"),
            mention("e2", "medical_history", "lupus"),
        ],
    )
    tables = builder.build([note(HISTORY_NOTE, visit_occurrence_id=3)], [result])
    diabetes, lupus = tables.rows["observation"]

    assert diabetes.observation_concept_id == 3002660
    assert diabetes.value_as_concept_id == 201826
    assert diabetes.value_source_value == "type 2 diabetes"
    assert diabetes.observation_source_value == "type 2 diabetes"
    assert diabetes.observation_date == date(2025, 6, 1)
    assert diabetes.observation_type_concept_id == 32858
    assert diabetes.visit_occurrence_id == 3
    assert lupus.value_as_concept_id == 0

    assert coder.calls == [(["lupus", "type 2 diabetes"], "Condition", None)]
    assert [(i.row_id, i.column, i.reason) for i in tables.issues] == [
        (2, "value_as_concept_id", "unmapped")
    ]


def test_medical_history_needs_a_mapper():
    with pytest.raises(ValueError, match="needs a coder"):
        OmopTableBuilder(
            EntityDefinitions.from_predefined("medical_history"), vocabulary=Lookup()
        )


def test_medical_history_and_conditions_share_one_extraction():
    # Past conditions go to observation, current ones to condition_occurrence.
    class Extractor:
        async def __call__(self, texts):
            defs = pipeline.entity_definitions
            return [
                extract(
                    defs,
                    t,
                    [
                        mention("e1", "medical_history", "lupus"),
                        mention(
                            "e2",
                            "condition_name",
                            "type 2 diabetes",
                            status="confirmed",
                        ),
                    ],
                )
                for t in texts
            ]

    lookup = Lookup()
    for c in (
        concept(32893, "OMOP4976963", "Confirmed", "Condition Status", "Condition Status"),
        concept(32899, "OMOP4976969", "Preliminary", "Condition Status", "Condition Status"),
    ):  # fmt: skip
        lookup.concepts[c.concept_id] = c
    pipeline = OmopExtractionPipeline(
        ["condition_occurrence", "medical_history"],
        extractor=Extractor(),
        vocabulary=lookup,
        coder=ConditionCoder(),
    )
    tables = asyncio.run(pipeline([note("H/o lupus. Has type 2 diabetes.")]))
    assert tables["observation"]["value_as_concept_id"].tolist() == [0]
    assert tables["observation"]["value_source_value"].tolist() == ["lupus"]
    assert tables["condition_occurrence"]["condition_concept_id"].tolist() == [201826]


# --- several definitions in one pipeline -------------------------------------------


def test_pipeline_combines_definitions():
    class Extractor:
        def __init__(self):
            self.calls = 0

        async def __call__(self, texts):
            self.calls += 1
            defs = pipeline.entity_definitions
            return [
                extract(
                    defs,
                    t,
                    [
                        mention("e1", "smoking_status", "Never smoker", status="never"),
                        mention(
                            "e2", "family_history", "type 2 diabetes", relative="father"
                        ),
                    ],
                )
                for t in texts
            ]

    builder = family_builder()
    extractor = Extractor()
    pipeline = OmopExtractionPipeline(
        ["smoking_status", "family_history"],
        extractor=extractor,
        vocabulary=builder.vocabulary,
        coder=ConditionCoder(),
    )
    assert {
        "smoking_status",
        "pack_years",
        "family_history",
    } <= pipeline.entity_definitions.keys

    text = "Never smoker. Father with type 2 diabetes."
    tables = asyncio.run(pipeline([note(text)]))
    assert extractor.calls == 1
    df = tables["observation"]
    assert df["observation_concept_id"].tolist() == [43054909, 4167217]


def test_a_file_listed_twice_is_reported_once():
    with pytest.raises(ValueError) as error:
        OmopExtractionPipeline(
            ["smoking_status", "family_history", "smoking_status"],
            extractor=object(),
            vocabulary=Lookup(),
        )
    message = str(error.value)
    assert "predefined 'smoking_status' is listed 2 times" in message
    assert "key " not in message  # not one line per key of the repeated file
    assert "input_value" not in message


def test_a_key_clash_names_both_files(tmp_path):
    own = tmp_path / "alcohol_use.yaml"
    own.write_text(
        "- name: Alcohol use\n  definition: Alcohol use.\n"
        "- name: Smoking status\n  definition: Whether the patient smokes.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as error:
        OmopExtractionPipeline(
            ["smoking_status", own], extractor=object(), vocabulary=Lookup()
        )
    assert (
        f"key 'smoking_status' is defined in predefined 'smoking_status' and {own}"
        in str(error.value)
    )


def test_a_key_clash_with_a_definitions_object_names_its_position():
    own = EntityDefinitions(
        entities=[{"name": "Family history", "definition": "Family history."}]
    )
    with pytest.raises(
        ValueError, match="defined in predefined 'family_history' and item 2"
    ):
        OmopExtractionPipeline(
            ["family_history", own], extractor=object(), vocabulary=Lookup()
        )


# --- predefined files ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["smoking_status", "family_history", "medical_history"]
)
def test_predefined_prompts_have_no_omop_details(name):
    prompt = EntityDefinitions.from_predefined(name).to_prompt_str()
    for leaked in (
        "LOINC",
        "SNOMED",
        "416471007",
        "11348-0",
        "LA18978-9",
        "omop",
        "concept_id",
    ):
        assert leaked not in prompt


@pytest.mark.skipif(
    not Path(".aatm/omop.db").exists(),
    reason="local OMOP vocabulary database not built",
)
def test_predefined_observation_concepts_resolve_in_local_vocabulary():
    lookup = SqliteConceptLookup(".aatm/omop.db")
    smoking = OmopTableBuilder(
        EntityDefinitions.from_predefined("smoking_status"), vocabulary=lookup
    )
    family = OmopTableBuilder(
        EntityDefinitions.from_predefined("family_history"),
        vocabulary=lookup,
        coder=ConditionCoder(),
    )
    history = OmopTableBuilder(
        EntityDefinitions.from_predefined("medical_history"),
        vocabulary=lookup,
        coder=ConditionCoder(),
    )
    names = (
        set(smoking.describe_concepts().concept_name)
        | set(family.describe_concepts().concept_name)
        | set(history.describe_concepts().concept_name)
    )
    assert {
        "Tobacco smoking status",
        "Never smoker",
        "Family history of clinical finding",
        "Mother",
        "History of Past illness Narrative",
    } <= names
