import pytest
from pydantic import ValidationError

from aatm.data_models.extractors import (
    EntityDefinition,
    EntityDefinitions,
    ExtractedEntityCandidate,
    ExtractedEntityCandidates,
    RejectedEntityCandidate,
    format_segments_for_prompt,
    make_entity_key,
    segment_text,
    validate_extracted_entity,
)

TEXT = (
    "Problems:\n"
    "* Hypertension since 2019\n"
    "\n"
    "* Asthma\r\n"
    "Plan: Hypertension is controlled; recheck Hypertension in 2027.\n"
)
# L1 "Problems:", L2 "* Hypertension since 2019", L3 "* Asthma", L4 "Plan: ..."


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


def cand(
    id, text, key, line, parents=None, occurrence=None
) -> ExtractedEntityCandidate:
    return ExtractedEntityCandidate(
        id=id,
        text=text,
        key=key,
        line=line,
        parents=parents or [],
        occurrence=occurrence,
    )


def candidates(*items: ExtractedEntityCandidate) -> ExtractedEntityCandidates:
    return ExtractedEntityCandidates(entities=list(items))


def reasons_by_id(result) -> dict[str, list[str]]:
    return {r.id: r.reasons for r in result.unmatched_entities}


# --- definitions -------------------------------------------------------------


def test_make_entity_key_normalizes_accents_and_punctuation():
    assert make_entity_key("Condição — Início!") == "condicao_inicio"
    with pytest.raises(ValueError):
        make_entity_key("!!!")


@pytest.mark.parametrize("key", ["BRCA-Test", "Brca_test", "brca__test", "_brca", "!!"])
def test_definition_rejects_keys_the_model_cannot_return(key):
    with pytest.raises(ValidationError, match="must use only lowercase letters"):
        EntityDefinition(name="BRCA test", key=key, definition="d")


def test_definition_key_error_suggests_a_valid_key():
    with pytest.raises(ValidationError, match="Use 'brca_test' instead"):
        EntityDefinition(name="BRCA test", key="BRCA-Test", definition="d")


def test_definition_accepts_valid_and_generated_keys():
    assert EntityDefinition(name="x", key="brca_test2", definition="d").key == (
        "brca_test2"
    )
    assert EntityDefinition(name="BRCA-Test", definition="d").key == "brca_test"


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


def test_definitions_reject_parent_cycles():
    with pytest.raises(ValidationError, match="Cycle"):
        EntityDefinitions(
            entities=[
                EntityDefinition(
                    name="A", definition="d", has_parent=True, parent_key="b"
                ),
                EntityDefinition(
                    name="B", definition="d", has_parent=True, parent_key="a"
                ),
            ]
        )


def test_key_depths():
    defs = EntityDefinitions(
        entities=[
            EntityDefinition(name="C", definition="d", has_parent=True, parent_key="b"),
            EntityDefinition(name="A", definition="d"),
            EntityDefinition(name="B", definition="d", has_parent=True, parent_key="a"),
        ]
    )
    assert defs.key_depths == {"a": 0, "b": 1, "c": 2}


@pytest.mark.parametrize(
    "key",
    [
        "condition_occurrence",
        "drug_exposure",
        "ecog_performance_status",
        "lab_results",
        "smoking_status",
        "family_history",
    ],
)
def test_predefined_definitions_load(key):
    defs = EntityDefinitions.from_predefined(key)
    assert defs.entities
    for parent in defs.parent_keys.values():
        assert parent in defs.keys


# --- segmentation ------------------------------------------------------------


def test_segment_text_skips_blank_lines_and_keeps_offsets():
    segments = segment_text(TEXT)
    assert [s.id for s in segments] == ["L1", "L2", "L3", "L4"]
    assert [s.text for s in segments][:3] == [
        "Problems:",
        "* Hypertension since 2019",
        "* Asthma",
    ]
    for s in segments:
        assert TEXT[s.start : s.end] == s.text


def test_segment_text_trims_whitespace():
    text = "   indented line   \n\tTabbed"
    segments = segment_text(text)
    assert [s.text for s in segments] == ["indented line", "Tabbed"]
    for s in segments:
        assert text[s.start : s.end] == s.text


def test_segment_text_does_not_split_by_default():
    line = "Sentence one. " * 100
    segments = segment_text(line)
    assert len(segments) == 1
    assert segments[0].text == line.strip()


def test_segment_text_rejects_non_positive_limit():
    with pytest.raises(ValueError):
        segment_text("a", max_segment_chars=0)


def test_segment_text_splits_long_lines_at_sentences():
    line = "First sentence here. Second sentence here. Third one."
    segments = segment_text(line, max_segment_chars=25)
    assert [s.text for s in segments] == [
        "First sentence here.",
        "Second sentence here.",
        "Third one.",
    ]
    for s in segments:
        assert line[s.start : s.end] == s.text


def test_segment_text_packs_short_sentences_and_keeps_long_ones_whole():
    line = "A. B. C. " + "x" * 50 + "."
    segments = segment_text(line, max_segment_chars=10)
    assert [s.text for s in segments] == ["A. B. C.", "x" * 50 + "."]


def test_format_segments_for_prompt():
    assert format_segments_for_prompt(segment_text("a\n\nb")) == "[L1] a\n[L2] b"


# --- candidate normalization -------------------------------------------------


@pytest.mark.parametrize("line", ["L4", "[L4]", "l4", "4", " L4 "])
def test_candidate_line_is_normalized(line):
    assert cand("e1", "x", "k", line).line == "L4"


def test_candidate_text_line_prefix_is_stripped():
    assert cand("e1", "[L2] Hypertension", "k", "L2").text == "Hypertension"


def test_candidate_id_and_parents_are_normalized():
    c = ExtractedEntityCandidate(
        id=" e1 ", text="x", key="k", line="L1", parents=[" e2", "", "  "]
    )
    assert c.id == "e1"
    assert c.parents == ["e2"]
    assert (
        ExtractedEntityCandidate(
            id="e1", text="x", key="k", line="L1", parents=None
        ).parents
        == []
    )


# --- grounding ---------------------------------------------------------------


def test_each_candidate_is_grounded_to_its_own_line(definitions):
    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(
            cand("e1", "Hypertension", "condition_name", "L2"),
            cand("e2", "2019", "verbatim_start_date", "L2", parents=["e1"]),
        ),
    )
    spans = [(e.id, e.text, e.start) for e in result.validated_entities]
    assert spans == [
        ("e1", "Hypertension", TEXT.index("Hypertension")),
        ("e2", "2019", TEXT.index("2019")),
    ]
    for e in result.validated_entities:
        assert TEXT[e.start : e.end] == e.text
    assert result.unmatched_entities == []
    assert result.unmatched_keys == []


def test_occurrence_selects_mention_within_line(definitions):
    l4 = TEXT.index("Plan:")
    first = TEXT.index("Hypertension", l4)
    second = TEXT.index("Hypertension", first + 1)

    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(
            cand("e1", "Hypertension", "condition_name", "L4"),
            cand("e2", "Hypertension", "condition_name", "L4", occurrence=2),
        ),
    )
    assert [e.start for e in result.validated_entities] == [first, second]


def test_explicit_segments_are_used(definitions):
    segments = segment_text(TEXT, max_segment_chars=30)
    # With a 30-char limit, line 4 is split at the semicolon into L4 and L5.
    assert segments[4].text == "recheck Hypertension in 2027."
    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(cand("e1", "Hypertension", "condition_name", "L5")),
        segments=segments,
    )
    assert result.validated_entities[0].start == segments[4].start + len("recheck ")


# --- rejection reasons -------------------------------------------------------


@pytest.mark.parametrize(
    "candidate, reasons",
    [
        (cand("x", "Diabetes", "condition_name", "L2"), ["invalid_text"]),
        (cand("x", "Asthma", "condition_name", "L2"), ["text_not_in_line"]),
        (cand("x", "Asthma", "condition_name", "L99"), ["invalid_line"]),
        (
            cand("x", "Hypertension", "condition_name", "L2", occurrence=2),
            ["invalid_occurrence"],
        ),
        (cand("x", "Asthma", "drug_name", "L3"), ["invalid_key"]),
        (cand("x", "2019", "verbatim_start_date", "L2"), ["required_parent_missing"]),
        (
            cand("x", "2019", "verbatim_start_date", "L2", parents=["e99"]),
            ["invalid_parent"],
        ),
        (
            cand("x", "Asthma", "condition_name", "L3", parents=["e1"]),
            ["unexpected_parent"],
        ),
    ],
)
def test_rejection_reasons(definitions, candidate, reasons):
    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(cand("e1", "Hypertension", "condition_name", "L2"), candidate),
    )
    assert reasons_by_id(result) == {"x": reasons}


def test_parent_with_wrong_key_is_invalid(definitions):
    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(
            cand("e1", "Hypertension", "condition_name", "L2"),
            cand("e2", "2019", "verbatim_start_date", "L2", parents=["e1"]),
            # a date cannot be the parent of a date
            cand("e3", "2027", "verbatim_start_date", "L4", parents=["e2"]),
        ),
    )
    assert reasons_by_id(result) == {"e3": ["invalid_parent"]}


def test_child_of_rejected_parent_is_rejected(definitions):
    # The parent is listed after the child, which must not matter.
    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(
            cand("e2", "2019", "verbatim_start_date", "L2", parents=["e1"]),
            cand("e1", "Hypertension", "condition_name", "L3"),
        ),
    )
    assert reasons_by_id(result) == {
        "e2": ["parent_rejected"],
        "e1": ["text_not_in_line"],
    }
    # rejected candidates are reported in input order
    assert [r.id for r in result.unmatched_entities] == ["e2", "e1"]


def test_duplicate_ids_keep_first_candidate(definitions):
    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(
            cand("e1", "Hypertension", "condition_name", "L2"),
            cand("e1", "Asthma", "condition_name", "L3"),
            cand("e2", "2019", "verbatim_start_date", "L2", parents=["e1"]),
        ),
    )
    assert [e.text for e in result.validated_entities] == ["Hypertension", "2019"]
    assert [r.reasons for r in result.unmatched_entities] == [["duplicate_id"]]
    assert result.unmatched_entities[0].text == "Asthma"


def test_rejection_keeps_span_when_grounded(definitions):
    result = validate_extracted_entity(
        TEXT, definitions, candidates(cand("e1", "Asthma", "drug_name", "L3"))
    )
    rejected = result.unmatched_entities[0]
    assert TEXT[rejected.start : rejected.end] == "Asthma"


# --- duplicates and graph ----------------------------------------------------


def test_duplicate_mentions_are_merged_and_children_relinked(definitions):
    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(
            cand("e1", "Hypertension", "condition_name", "L2"),
            cand("e2", "Hypertension", "condition_name", "[L2]"),
            cand("e3", "2019", "verbatim_start_date", "L2", parents=["e2"]),
        ),
    )
    assert [e.id for e in result.validated_entities] == ["e1", "e3"]
    assert result.get_entity("e3").parents == ["e1"]
    assert reasons_by_id(result) == {"e2": ["duplicate"]}
    assert result.unmatched_keys == []


def test_duplicate_children_merge_parents(definitions):
    text = "Hypertension and asthma since 2015"
    result = validate_extracted_entity(
        text,
        definitions,
        candidates(
            cand("e1", "Hypertension", "condition_name", "L1"),
            cand("e2", "asthma", "condition_name", "L1"),
            cand("e3", "2015", "verbatim_start_date", "L1", parents=["e1"]),
            cand("e4", "2015", "verbatim_start_date", "L1", parents=["e2"]),
        ),
    )
    assert result.get_entity("e3").parents == ["e1", "e2"]
    assert reasons_by_id(result) == {"e4": ["duplicate"]}


def test_child_links_to_specific_parent_mention(definitions):
    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(
            cand("e1", "Hypertension", "condition_name", "L2"),
            cand("e2", "Hypertension", "condition_name", "L4"),
            cand("e3", "2019", "verbatim_start_date", "L2", parents=["e1"]),
            cand("e4", "2027", "verbatim_start_date", "L4", parents=["e2"]),
        ),
    )
    records = result.to_records()
    assert [(r.id, r.line) for r in records] == [("e1", "L2"), ("e2", "L4")]
    assert [c.text for c in records[0].children["verbatim_start_date"]] == ["2019"]
    assert [c.text for c in records[1].children["verbatim_start_date"]] == ["2027"]


def test_to_records_puts_shared_child_under_each_parent(definitions):
    text = "Hypertension and asthma since 2015"
    result = validate_extracted_entity(
        text,
        definitions,
        candidates(
            cand("e1", "Hypertension", "condition_name", "L1"),
            cand("e2", "asthma", "condition_name", "L1"),
            cand("e3", "2015", "verbatim_start_date", "L1", parents=["e1", "e2"]),
        ),
    )
    records = result.to_records()
    assert [r.text for r in records] == ["Hypertension", "asthma"]
    for r in records:
        assert [c.id for c in r.children["verbatim_start_date"]] == ["e3"]


def test_to_records_supports_deeper_hierarchies():
    defs = EntityDefinitions(
        entities=[
            EntityDefinition(name="Drug", definition="d"),
            EntityDefinition(
                name="Dose", definition="d", has_parent=True, parent_key="drug"
            ),
            EntityDefinition(
                name="Unit", definition="d", has_parent=True, parent_key="dose"
            ),
        ]
    )
    text = "Furosemide 40 mg daily"
    result = validate_extracted_entity(
        text,
        defs,
        candidates(
            cand("e3", "mg", "unit", "L1", parents=["e2"]),
            cand("e2", "40", "dose", "L1", parents=["e1"]),
            cand("e1", "Furosemide", "drug", "L1"),
        ),
    )
    (drug,) = result.to_records()
    (dose,) = drug.children["dose"]
    (unit,) = dose.children["unit"]
    assert (drug.text, dose.text, unit.text) == ("Furosemide", "40", "mg")


# --- misc --------------------------------------------------------------------


def test_rejected_candidate_span_validation():
    base = dict(id="e1", text="abc", key="k", line="L1", reasons=["duplicate"])
    RejectedEntityCandidate(**base)
    RejectedEntityCandidate(**base, start=0, end=3)
    with pytest.raises(ValidationError):
        RejectedEntityCandidate(**base, start=0)
    with pytest.raises(ValidationError):
        RejectedEntityCandidate(**base, start=0, end=2)


def test_empty_text_raises(definitions):
    with pytest.raises(ValueError):
        validate_extracted_entity("", definitions, candidates())


# --- loading from files ------------------------------------------------------


def test_from_yaml_error_names_the_file(tmp_path):
    path = tmp_path / "looping.yaml"
    path.write_text(
        "- name: A\n  definition: d\n  has_parent: true\n  parent_key: b\n"
        "- name: B\n  definition: d\n  has_parent: true\n  parent_key: a\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="looping.yaml") as exc_info:
        EntityDefinitions.from_yaml(path)
    assert "Cycle in entity parent keys: a -> b -> a" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_from_yaml_invalid_yaml_names_the_file(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("- name: A\n  definition: [unclosed\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match="Invalid YAML in entity definitions .*broken.yaml"
    ):
        EntityDefinitions.from_yaml(path)
