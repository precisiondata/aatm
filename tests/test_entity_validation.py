import pytest
from pydantic import ValidationError

from aatm.data_models.extractors import (
    EntityDefinition,
    EntityDefinitions,
    ExtractedEntityCandidate,
    ExtractedEntityCandidates,
    RejectedEntityCandidate,
    make_entity_key,
    validate_extracted_entity,
)

TEXT = "Patient has Hypertension since 2019. Hypertension is controlled."


@pytest.fixture
def definitions() -> EntityDefinitions:
    return EntityDefinitions(
        entities=[
            EntityDefinition(name="Condition name", definition="A condition."),
            EntityDefinition(
                name="Start date",
                key="verbatim_start_date",
                definition="Start date of a condition.",
                has_parent=True,
                parent_key="condition_name",
            ),
        ]
    )


def candidates(*items: tuple) -> ExtractedEntityCandidates:
    return ExtractedEntityCandidates(
        entities=[
            ExtractedEntityCandidate(
                text=i[0], key=i[1], parents=i[2] if len(i) > 2 else []
            )
            for i in items
        ]
    )


def test_make_entity_key_normalizes_accents_and_punctuation():
    assert make_entity_key("Condição — Início!") == "condicao_inicio"
    with pytest.raises(ValueError):
        make_entity_key("!!!")


def test_definition_requires_parent_key_when_has_parent():
    with pytest.raises(ValidationError):
        EntityDefinition(name="Date", definition="d", has_parent=True)


def test_definitions_reject_unknown_parent_and_duplicate_keys():
    with pytest.raises(ValidationError):
        EntityDefinitions(
            entities=[
                EntityDefinition(
                    name="Date", definition="d", has_parent=True, parent_key="nope"
                )
            ]
        )
    with pytest.raises(ValidationError):
        EntityDefinitions(
            entities=[
                EntityDefinition(name="A", definition="d"),
                EntityDefinition(name="A", definition="d"),
            ]
        )


@pytest.mark.parametrize("key", ["condition_occurrence", "drug_exposure"])
def test_predefined_definitions_load(key):
    defs = EntityDefinitions.from_predefined(key)
    assert defs.entities
    for child, parent in defs.parent_keys.items():
        assert parent in defs.keys


def test_valid_entities_get_spans_for_every_occurrence(definitions):
    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(
            ("Hypertension", "condition_name"),
            ("2019", "verbatim_start_date", ["Hypertension"]),
        ),
    )
    spans = [(e.text, e.start, e.end) for e in result.validated_entities]
    assert spans == [
        ("Hypertension", 12, 24),
        ("2019", 31, 35),
        ("Hypertension", 37, 49),
    ]
    for e in result.validated_entities:
        assert TEXT[e.start : e.end] == e.text
    assert result.unmatched_entities == []
    assert result.unmatched_keys == []


@pytest.mark.parametrize(
    "candidate, reasons",
    [
        (("Diabetes", "condition_name"), ["invalid_text"]),
        (("Hypertension", "drug_name"), ["invalid_key"]),
        (("2019", "verbatim_start_date"), ["required_parent_missing"]),
        # parent text not in the note
        (("2019", "verbatim_start_date", ["Diabetes"]), ["invalid_parent"]),
        # parent text in the note but not extracted as a condition
        (("2019", "verbatim_start_date", ["controlled"]), ["invalid_parent"]),
    ],
)
def test_rejection_reasons(definitions, candidate, reasons):
    result = validate_extracted_entity(
        TEXT, definitions, candidates(("Hypertension", "condition_name"), candidate)
    )
    rejected = [r for r in result.unmatched_entities if r.text == candidate[0]]
    assert len(rejected) == 1
    assert rejected[0].reasons == reasons


def test_duplicates_are_rejected(definitions):
    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(
            ("Hypertension", "condition_name"), ("Hypertension", "condition_name")
        ),
    )
    assert len(result.validated_entities) == 2
    assert [r.reasons for r in result.unmatched_entities] == [
        ["duplicate"],
        ["duplicate"],
    ]
    assert result.unmatched_keys == ["verbatim_start_date"]


def test_rejected_candidate_span_validation():
    RejectedEntityCandidate(text="abc", key="k", reasons=["duplicate"])
    RejectedEntityCandidate(text="abc", key="k", start=0, end=3, reasons=["duplicate"])
    with pytest.raises(ValidationError):
        RejectedEntityCandidate(text="abc", key="k", start=0, reasons=["duplicate"])
    with pytest.raises(ValidationError):
        RejectedEntityCandidate(
            text="abc", key="k", start=0, end=2, reasons=["duplicate"]
        )


def test_empty_text_raises(definitions):
    with pytest.raises(ValueError):
        validate_extracted_entity("", definitions, candidates())
