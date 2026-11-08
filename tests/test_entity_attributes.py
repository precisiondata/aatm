import pytest
from pydantic import ValidationError

from aatm.data_models.extractors import (
    AttributeDefinition,
    AttributeLabel,
    EntityDefinition,
    EntityDefinitions,
    ExtractedAttribute,
    ExtractedEntityCandidate,
    ExtractedEntityCandidates,
    validate_extracted_entity,
)

TEXT = (
    "BRCA1/2 panel 2024: no pathogenic variants detected.\n"
    "Repeat BRCA testing 2025 showed a VUS in BRCA2.\n"
)
# L1 "BRCA1/2 panel 2024: ...", L2 "Repeat BRCA testing 2025 ..."

RESULT_LABELS = [
    AttributeLabel(value="positive", description="Pathogenic variant reported."),
    AttributeLabel(value="negative", description="No pathogenic variant found."),
    AttributeLabel(value="uncertain", description="Only a VUS was reported."),
]


def make_definitions(result_required: bool = False) -> EntityDefinitions:
    return EntityDefinitions(
        entities=[
            EntityDefinition(
                name="BRCA test",
                definition="A BRCA1/2 genetic test.",
                attributes=[
                    AttributeDefinition(
                        key="result",
                        description="The test result.",
                        required=result_required,
                        labels=RESULT_LABELS,
                    ),
                    AttributeDefinition(
                        key="origin",
                        description="Germline or somatic testing.",
                        labels=[
                            AttributeLabel(value="germline", description="Germline."),
                            AttributeLabel(value="somatic", description="Tumour."),
                        ],
                    ),
                ],
            ),
            EntityDefinition(
                name="Test date",
                definition="Date of a test.",
                has_parent=True,
                parent_key="brca_test",
            ),
        ]
    )


def cand(id, text, key, line, attributes=None, parents=None):
    return ExtractedEntityCandidate(
        id=id,
        text=text,
        key=key,
        line=line,
        parents=parents or [],
        attributes=[ExtractedAttribute(key=k, value=v) for k, v in attributes or []],
    )


def run(*items, required=False):
    return validate_extracted_entity(
        TEXT,
        make_definitions(result_required=required),
        ExtractedEntityCandidates(entities=list(items)),
    )


def issues(result_issues):
    return [(i.entity_id, i.attribute_key, i.value, i.reason) for i in result_issues]


# --- definitions -------------------------------------------------------------


def test_attribute_definition_rejects_duplicate_labels_and_no_labels():
    with pytest.raises(ValidationError):
        AttributeDefinition(
            key="result",
            description="d",
            labels=[
                AttributeLabel(value="Positive", description="d"),
                AttributeLabel(value="positive", description="d"),
            ],
        )
    with pytest.raises(ValidationError):
        AttributeDefinition(key="result", description="d", labels=[])


def test_entity_definition_rejects_duplicate_attribute_keys():
    attribute = AttributeDefinition(
        key="result", description="d", labels=[RESULT_LABELS[0]]
    )
    with pytest.raises(ValidationError):
        EntityDefinition(name="X", definition="d", attributes=[attribute, attribute])


def test_attributes_load_from_yaml(tmp_path):
    path = tmp_path / "brca.yaml"
    path.write_text(
        "- name: BRCA test\n"
        "  definition: A BRCA test.\n"
        "  attributes:\n"
        "    - key: result\n"
        "      description: The test result.\n"
        "      required: true\n"
        "      labels:\n"
        "        - value: positive\n"
        "          description: Pathogenic variant.\n"
        "        - value: negative\n"
        "          description: No pathogenic variant.\n",
        encoding="utf-8",
    )
    (entity,) = EntityDefinitions.from_yaml(path).entities
    (attribute,) = entity.attributes
    assert attribute.required is True
    assert [label.value for label in attribute.labels] == ["positive", "negative"]


def test_prompt_lists_attributes_and_labels():
    prompt = make_definitions(result_required=True).to_prompt_str()
    assert "`result` (required): The test result." in prompt
    assert "`uncertain`: Only a VUS was reported." in prompt
    assert "`origin` (optional)" in prompt


# --- validation --------------------------------------------------------------


def test_valid_attributes_are_kept_and_normalized():
    result = run(
        cand(
            "e1",
            "BRCA1/2 panel",
            "brca_test",
            "L1",
            [("Origin", "germline"), ("result", " Negative ")],
        )
    )
    (entity,) = result.validated_entities
    # Definition order, label spelled as defined.
    assert [(a.key, a.value) for a in entity.attributes] == [
        ("result", "negative"),
        ("origin", "germline"),
    ]
    assert entity.get_attribute("result") == "negative"
    assert result.required_attribute_issues == []
    assert result.optional_attribute_issues == []


def test_missing_or_null_optional_attribute_is_not_an_issue():
    result = run(cand("e1", "BRCA1/2 panel", "brca_test", "L1", [("result", None)]))
    (entity,) = result.validated_entities
    assert entity.attributes == []
    assert result.optional_attribute_issues == []


def test_invalid_optional_attributes_keep_the_span():
    result = run(
        cand(
            "e1",
            "BRCA1/2 panel",
            "brca_test",
            "L1",
            [("result", "indeterminate"), ("lab", "Myriad")],
        )
    )
    (entity,) = result.validated_entities
    assert entity.attributes == []
    assert issues(result.optional_attribute_issues) == [
        ("e1", "result", "indeterminate", "invalid_label"),
        ("e1", "lab", "Myriad", "unknown_attribute"),
    ]
    assert result.unmatched_entities == []


def test_conflicting_values_keep_the_first():
    result = run(
        cand(
            "e1",
            "BRCA1/2 panel",
            "brca_test",
            "L1",
            [("result", "negative"), ("result", "positive"), ("result", "negative")],
        )
    )
    assert result.validated_entities[0].get_attribute("result") == "negative"
    (issue,) = result.optional_attribute_issues
    assert (issue.reason, issue.value, issue.kept_value) == (
        "conflicting_values",
        "positive",
        "negative",
    )


def test_missing_required_attribute_rejects_span_and_children():
    result = run(
        cand("e1", "BRCA1/2 panel", "brca_test", "L1", [("origin", "germline")]),
        cand("e2", "2024", "test_date", "L1", parents=["e1"]),
        required=True,
    )
    assert result.validated_entities == []
    assert {r.id: r.reasons for r in result.unmatched_entities} == {
        "e1": ["required_attribute_missing"],
        "e2": ["parent_rejected"],
    }
    assert issues(result.required_attribute_issues) == [
        ("e1", "result", None, "missing_required")
    ]
    assert result.optional_attribute_issues == []


def test_invalid_required_attribute_rejects_span():
    result = run(
        cand("e1", "BRCA1/2 panel", "brca_test", "L1", [("result", "unclear")]),
        required=True,
    )
    (rejected,) = result.unmatched_entities
    assert rejected.reasons == ["invalid_required_attribute"]
    assert issues(result.required_attribute_issues) == [
        ("e1", "result", "unclear", "invalid_label")
    ]


def test_attributes_are_not_checked_for_otherwise_rejected_candidates():
    result = run(
        cand("e1", "BRCA3 panel", "brca_test", "L1", [("result", "unclear")]),
        required=True,
    )
    assert result.unmatched_entities[0].reasons == ["invalid_text"]
    assert result.required_attribute_issues == []


def test_duplicate_keeps_first_value_and_flags_conflict():
    result = run(
        cand("e1", "BRCA1/2 panel", "brca_test", "L1", [("result", "negative")]),
        cand(
            "e2",
            "BRCA1/2 panel",
            "brca_test",
            "L1",
            [("result", "positive"), ("origin", "germline")],
        ),
    )
    (entity,) = result.validated_entities
    # The missing attribute is filled in from the duplicate; the conflict is not.
    assert entity.get_attribute("result") == "negative"
    assert entity.get_attribute("origin") == "germline"
    (issue,) = result.optional_attribute_issues
    assert issue.model_dump() == {
        "entity_id": "e2",
        "entity_key": "brca_test",
        "attribute_key": "result",
        "value": "positive",
        "reason": "conflicting_duplicate",
        "required": False,
        "kept_entity_id": "e1",
        "kept_value": "negative",
    }
    assert result.unmatched_entities[0].reasons == ["duplicate"]


def test_duplicate_missing_required_attribute_is_not_reported():
    result = run(
        cand("e1", "BRCA1/2 panel", "brca_test", "L1", [("result", "negative")]),
        cand("e2", "BRCA1/2 panel", "brca_test", "L1"),
        required=True,
    )
    assert result.validated_entities[0].get_attribute("result") == "negative"
    assert result.unmatched_entities[0].reasons == ["duplicate"]
    assert result.required_attribute_issues == []


def test_required_conflict_goes_to_required_issues():
    result = run(
        cand("e1", "BRCA1/2 panel", "brca_test", "L1", [("result", "negative")]),
        cand("e2", "BRCA1/2 panel", "brca_test", "L1", [("result", "uncertain")]),
        required=True,
    )
    assert issues(result.required_attribute_issues) == [
        ("e2", "result", "uncertain", "conflicting_duplicate")
    ]


def test_records_include_attributes():
    result = run(
        cand("e1", "BRCA1/2 panel", "brca_test", "L1", [("result", "negative")]),
        cand("e2", "2024", "test_date", "L1", parents=["e1"]),
        cand("e3", "BRCA testing", "brca_test", "L2", [("result", "uncertain")]),
    )
    records = result.to_records()
    assert [(r.text, r.attributes) for r in records] == [
        ("BRCA1/2 panel", {"result": "negative"}),
        ("BRCA testing", {"result": "uncertain"}),
    ]
    assert records[0].children["test_date"][0].attributes == {}


def test_candidate_schema_is_accepted_by_openai_strict_mode():
    pydantic_helpers = pytest.importorskip("openai.lib._pydantic")
    pydantic_helpers.to_strict_json_schema(ExtractedEntityCandidates)
