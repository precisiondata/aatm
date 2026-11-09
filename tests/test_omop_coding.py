"""Coding extracted text: `map` (terminology mapper) and `lookup` (code lookup) sources."""

from datetime import date

import pytest

from aatm.data_models.data_models import RetrievedExpressionMetadata, RetrieverResults
from aatm.data_models.extractors import (
    EntityDefinitions,
    ExtractedEntityCandidates,
    validate_extracted_entity,
)
from aatm.omop import ClinicalNote, CodedText, OmopTableBuilder, TerminologyMapperCoder
from aatm.omop.coding import retriever_filter
from aatm.omop.vocabulary import Concept
from aatm.retrievers import BaseRetriever
from aatm.terminology_mapper import TerminologyMapper
from test_omop_mapping import FakeLookup, concept

UNIT_CONCEPTS = [
    concept(8840, "mg/dL", "milligram per deciliter", "Unit", "UCUM"),
    concept(8713, "g/dL", "gram per deciliter", "Unit", "UCUM"),
    concept(8554, "%", "percent", "Unit", "UCUM"),
    # Two standard units that differ only by case: "mm" vs "Mm".
    concept(8588, "mm", "millimeter", "Unit", "UCUM"),
    concept(9999, "Mm", "megameter", "Unit", "UCUM"),
    concept(32858, "OMOP4976931", "NLP", "Type Concept", "Type Concept"),
    concept(4171756, "276139006", "<", "Meas Value Operator"),
    concept(
        3000963, "718-7", "Hemoglobin [Mass/volume] in Blood", "Measurement", "LOINC"
    ),
]


class UnitLookup(FakeLookup):
    def __init__(self):
        super().__init__(concepts=UNIT_CONCEPTS, maps_to={})

    def search_codes(self, vocabulary_id, codes):
        wanted = {c.strip().lower() for c in codes}
        found: dict[str, list[Concept]] = {}
        for c in self.concepts.values():
            if c.vocabulary_id == vocabulary_id and c.concept_code.lower() in wanted:
                found.setdefault(c.concept_code.lower(), []).append(c)
        return found


class FakeCoder:
    """Maps known texts; records every call."""

    def __init__(self, known=None):
        self.known = known or {
            "Hb": CodedText(
                text="Hb",
                concept_id=3000963,
                vocabulary_id="LOINC",
                concept_code="718-7",
                domain_id="Measurement",
                score=0.9,
            ),
            "glucose": CodedText(
                text="glucose",
                concept_id=3004501,
                concept_name="Glucose [Mass/volume] in Serum or Plasma",
                vocabulary_id="LOINC",
                domain_id="Measurement",
                score=0.4,
            ),
            "BMI": CodedText(
                text="BMI",
                concept_id=4245997,
                concept_name="Body mass index",
                vocabulary_id="SNOMED",
                domain_id="Observation",
                score=0.95,
            ),
        }
        self.calls: list[tuple] = []

    def code(self, texts, domain=None, vocabularies=None):
        self.calls.append((list(texts), domain, vocabularies))
        return [self.known.get(t, CodedText(text=t)) for t in texts]


def lab_definitions(**map_options):
    return EntityDefinitions(
        entities=[
            {
                "name": "Lab test",
                "key": "lab_test",
                "definition": "A lab test.",
                "attributes": [
                    {
                        "key": "operator",
                        "description": "Comparison sign.",
                        "labels": [{"value": "<", "description": "Less than."}],
                    }
                ],
                "omop": {
                    "table": "measurement",
                    "fields": {
                        "person_id": {"from": "context.person_id"},
                        "measurement_concept_id": {
                            "from": "text",
                            "map": {"vocabularies": ["LOINC"], **map_options},
                        },
                        "measurement_source_value": {"from": "text"},
                        "measurement_date": {"from": "context.note_date"},
                        "measurement_type_concept_id": {"value": 32858},
                        "operator_concept_id": {
                            "from": "attribute.operator",
                            "concepts": {"<": "SNOMED:276139006"},
                        },
                        "value_as_number": {"from": "child.lab_value", "as": "float"},
                        "unit_concept_id": {"from": "child.lab_unit", "lookup": "UCUM"},
                        "unit_source_value": {"from": "child.lab_unit"},
                    },
                },
            },
            {
                "name": "Lab value",
                "definition": "The value.",
                "has_parent": True,
                "parent_key": "lab_test",
            },
            {
                "name": "Lab unit",
                "definition": "The unit.",
                "has_parent": True,
                "parent_key": "lab_test",
            },
        ]
    )


def lab(id, text, line="L1", operator=None):
    attributes = [{"key": "operator", "value": operator}] if operator else []
    return {
        "id": id,
        "text": text,
        "key": "lab_test",
        "line": line,
        "attributes": attributes,
    }


def part(id, key, text, parent, line="L1"):
    return {"id": id, "text": text, "key": key, "line": line, "parents": [parent]}


def run(builder, text, entities, **note):
    candidates = ExtractedEntityCandidates.model_validate({"entities": entities})
    result = validate_extracted_entity(text, builder.entity_definitions, candidates)
    note = ClinicalNote(text=text, person_id=1, note_date=date(2025, 6, 1), **note)
    return builder.build([note], [result])


def make_builder(coder=None, **map_options):
    return OmopTableBuilder(
        lab_definitions(**map_options),
        vocabulary=UnitLookup(),
        mapper=coder or FakeCoder(),
    )


# --- mapping spec -----------------------------------------------------------------


@pytest.mark.parametrize(
    "source, message",
    [
        ({"value": 1, "map": {}}, "code text from the note"),
        ({"from": "context.person_id", "lookup": "UCUM"}, "code text from the note"),
        ({"from": "text", "map": {}, "lookup": "UCUM"}, "Use only one of"),
        ({"from": "text", "map": {"min_score": 2}}, "less than or equal to 1"),
        ({"from": "text", "lookup": "UCUM", "as": "int"}, "'as' cannot be used"),
        ({"from": "text", "map": {"vocab": ["LOINC"]}}, "Extra inputs"),
    ],
)
def test_invalid_coded_sources(source, message):
    from aatm.omop import FieldSource

    with pytest.raises(ValueError, match=message):
        FieldSource.model_validate(source)


def test_map_needs_a_mapper():
    with pytest.raises(ValueError, match="needs a terminology mapper"):
        OmopTableBuilder(lab_definitions(), vocabulary=UnitLookup())


def test_map_vocabulary_must_be_indexed():
    defs = lab_definitions()
    defs.get("lab_test").omop.fields["measurement_concept_id"][0].map.vocabularies = [
        "ICD10"
    ]
    with pytest.raises(ValueError, match="only indexes .* not \\['ICD10'\\]"):
        OmopTableBuilder(defs, vocabulary=UnitLookup(), mapper=FakeCoder())


def test_map_domain_must_match_the_column():
    with pytest.raises(ValueError, match="must hold a 'Measurement' concept"):
        make_builder(domain="Condition")


def test_retriever_filter():
    assert retriever_filter(None, None) is None
    assert retriever_filter("Measurement", None) == {"std_domain_id": "Measurement"}
    assert retriever_filter("Measurement", ["LOINC"]) == {
        "$and": [
            {"std_domain_id": "Measurement"},
            {"std_vocabulary_id": {"$in": ["LOINC"]}},
        ]
    }


# --- building rows ------------------------------------------------------------------

NOTE = "Hb 11,2 g/dL, glucose 99 MG/DL\nHb 10.9 g/dL, sodium <130 mm"


def test_lab_rows_are_mapped_and_looked_up():
    coder = FakeCoder()
    builder = make_builder(coder)
    tables = run(
        builder,
        NOTE,
        [
            lab("e1", "Hb"),
            part("e2", "lab_value", "11,2", "e1"),
            part("e3", "lab_unit", "g/dL", "e1"),
            lab("e4", "glucose"),
            part("e5", "lab_value", "99", "e4"),
            part("e6", "lab_unit", "MG/DL", "e4"),
            lab("e7", "Hb", line="L2"),
            part("e8", "lab_value", "10.9", "e7", line="L2"),
        ],
    )
    hb, glucose, hb_again = tables.rows["measurement"]

    assert hb.measurement_concept_id == 3000963
    assert hb.value_as_number == 11.2  # decimal comma
    assert hb.unit_concept_id == 8713  # exact UCUM code
    assert glucose.unit_concept_id == 8840  # "MG/DL": the only match ignoring case
    assert glucose.unit_source_value == "MG/DL"
    assert hb_again.measurement_concept_id == 3000963
    assert hb_again.unit_concept_id is None  # no unit mention, no lookup

    # Each distinct text is mapped once, with the column's domain and the vocabularies.
    assert coder.calls == [(["Hb", "glucose"], "Measurement", ["LOINC"])]
    assert tables.issues == []

    mappings = tables.mappings_dataframe()
    first = mappings.iloc[0]
    assert (first.row_id, first.column, first.method) == (
        1,
        "measurement_concept_id",
        "map",
    )
    # The mapper gives no name: it is added from the vocabulary.
    assert first.concept_name == "Hemoglobin [Mass/volume] in Blood"
    assert set(mappings.method) == {"map", "lookup"}


def test_unmapped_text_and_ambiguous_unit_store_zero():
    builder = make_builder()
    tables = run(
        builder,
        "sodium <130 mm",
        [lab("e1", "sodium", operator="<"), part("e2", "lab_unit", "mm", "e1")],
    )
    (row,) = tables.rows["measurement"]
    assert row.measurement_concept_id == 0
    assert row.operator_concept_id == 4171756
    assert row.unit_concept_id == 8588  # exact match wins over "Mm"
    issues = [(i.column, i.reason) for i in tables.issues]
    assert issues == [("measurement_concept_id", "unmapped")]


def test_case_insensitive_unit_must_be_unique():
    builder = make_builder()
    tables = run(
        builder, "Hb 3 MM", [lab("e1", "Hb"), part("e2", "lab_unit", "MM", "e1")]
    )
    assert tables.rows["measurement"][0].unit_concept_id == 0
    (issue,) = tables.issues
    assert issue.reason == "unmapped"
    assert "UCUM code lookup" in issue.detail


def test_low_confidence_mapping_is_rejected():
    builder = make_builder(min_score=0.5)
    tables = run(builder, "glucose 99", [lab("e1", "glucose")])
    assert tables.rows["measurement"][0].measurement_concept_id == 0
    (issue,) = tables.issues
    assert issue.reason == "low_confidence"
    assert "scored 0.4, below min_score 0.5" in issue.detail
    mapping = tables.mappings[0]
    assert (mapping.concept_id, mapping.stored) == (3004501, 0)


def test_concept_of_another_domain_is_reported():
    builder = make_builder()
    tables = run(builder, "BMI 31", [lab("e1", "BMI")])
    assert tables.rows["measurement"][0].measurement_concept_id == 4245997
    (issue,) = tables.issues
    assert issue.reason == "domain_mismatch"
    assert "'Observation' domain, not 'Measurement'" in issue.detail


def test_no_mapper_call_without_coded_text():
    coder = FakeCoder()
    builder = make_builder(coder)
    builder.build(
        ["nothing"],
        [
            validate_extracted_entity(
                "nothing",
                builder.entity_definitions,
                ExtractedEntityCandidates(entities=[]),
            )
        ],
    )
    assert coder.calls == []


# --- the existing TerminologyMapper -------------------------------------------------


class RecordingRetriever(BaseRetriever):
    """Returns one LOINC candidate per query and records the filter it was given."""

    def __init__(self):
        self.wheres: list = []

    def retrieve(self, queries, where=None, *args, **kwargs):
        self.wheres.append(where)
        return RetrieverResults(
            queries=queries,
            results=[
                [
                    RetrievedExpressionMetadata(
                        expression=q,
                        expression_concept_id="1",
                        expression_origin="standard_concept",
                        std_concept_id="3000963",
                        std_concept_name="Hemoglobin [Mass/volume] in Blood",
                        std_vocabulary_id="LOINC",
                        std_vocabulary_code="718-7",
                        std_domain_id="Measurement",
                        distance=0.25,
                    )
                ]
                for q in queries
            ],
        )


def test_terminology_mapper_passes_the_filter_to_its_retriever(tmp_path):
    retriever = RecordingRetriever()
    mapper = TerminologyMapper(retriever=retriever, output_dir=tmp_path)
    coder = TerminologyMapperCoder(mapper)

    (coded,) = coder.code(["Hb"], domain="Measurement", vocabularies=["LOINC"])

    assert retriever.wheres == [retriever_filter("Measurement", ["LOINC"])]
    assert coded == CodedText(
        text="Hb",
        concept_id=3000963,
        vocabulary_id="LOINC",
        concept_code="718-7",
        domain_id="Measurement",
        score=0.75,
    )
    assert not any(tmp_path.iterdir())  # nothing written to disk


def test_terminology_mapper_without_filter_is_unchanged(tmp_path):
    retriever = RecordingRetriever()
    mapper = TerminologyMapper(retriever=retriever, output_dir=tmp_path)
    mapper.map(expressions=["Hb"], save_to_disk=False)
    assert retriever.wheres == [None]


def test_builder_wraps_a_terminology_mapper(tmp_path):
    mapper = TerminologyMapper(retriever=RecordingRetriever(), output_dir=tmp_path)
    builder = OmopTableBuilder(
        lab_definitions(), vocabulary=UnitLookup(), mapper=mapper
    )
    assert isinstance(builder.coder, TerminologyMapperCoder)
    tables = run(builder, "Hb 11", [lab("e1", "Hb")])
    assert tables.rows["measurement"][0].measurement_concept_id == 3000963


# --- predefined file ------------------------------------------------------------------


def test_predefined_lab_results_prompt_has_no_omop_details():
    defs = EntityDefinitions.from_predefined("lab_results")
    assert defs.get("lab_test").omop.table == "measurement"
    prompt = defs.to_prompt_str()
    for leaked in ("LOINC", "UCUM", "SNOMED", "276139006", "omop", "concept_id"):
        assert leaked not in prompt


@pytest.mark.skipif(
    not __import__("pathlib").Path(".aatm/omop.db").exists(),
    reason="local OMOP vocabulary database not built",
)
def test_predefined_lab_results_resolve_in_local_vocabulary():
    from aatm.omop import SqliteConceptLookup

    builder = OmopTableBuilder(
        EntityDefinitions.from_predefined("lab_results"),
        vocabulary=SqliteConceptLookup(".aatm/omop.db"),
        mapper=FakeCoder(),
    )
    operators = builder.describe_concepts().set_index("label")["concept_name"]
    assert operators.to_dict() == {"<": "<", "<=": "<=", ">": ">", ">=": ">="}
    (unit,) = builder._lookup_codes("UCUM", ["mg/dl"])
    assert (unit.concept_id, unit.concept_code) == (8840, "mg/dL")
