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


def cand(text, key, line, parents=None, occurrence=None) -> ExtractedEntityCandidate:
    return ExtractedEntityCandidate(
        text=text, key=key, line=line, parents=parents or [], occurrence=occurrence
    )


def candidates(*items: ExtractedEntityCandidate) -> ExtractedEntityCandidates:
    return ExtractedEntityCandidates(entities=list(items))


# --- definitions -------------------------------------------------------------


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
    assert cand("x", "k", line).line == "L4"


def test_candidate_text_line_prefix_is_stripped():
    assert cand("[L2] Hypertension", "k", "L2").text == "Hypertension"


# --- validation --------------------------------------------------------------


def test_each_candidate_is_grounded_to_its_own_line(definitions):
    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(
            cand("Hypertension", "condition_name", "L2"),
            cand("2019", "verbatim_start_date", "L2", parents=["Hypertension"]),
        ),
    )
    spans = [(e.text, e.line, e.start) for e in result.validated_entities]
    assert spans == [
        ("Hypertension", "L2", TEXT.index("Hypertension")),
        ("2019", "L2", TEXT.index("2019")),
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
            cand("Hypertension", "condition_name", "L4"),
            cand("Hypertension", "condition_name", "L4", occurrence=2),
        ),
    )
    assert [e.start for e in result.validated_entities] == [first, second]


@pytest.mark.parametrize(
    "candidate, reasons",
    [
        (cand("Diabetes", "condition_name", "L2"), ["invalid_text"]),
        (cand("Asthma", "condition_name", "L2"), ["text_not_in_line"]),
        (cand("Asthma", "condition_name", "L99"), ["invalid_line"]),
        (
            cand("Hypertension", "condition_name", "L2", occurrence=2),
            ["invalid_occurrence"],
        ),
        (cand("Asthma", "drug_name", "L3"), ["invalid_key"]),
        (cand("2019", "verbatim_start_date", "L2"), ["required_parent_missing"]),
        # parent text not extracted at all
        (
            cand("2019", "verbatim_start_date", "L2", parents=["Diabetes"]),
            ["invalid_parent"],
        ),
        # parent text in the note but not extracted as a condition
        (
            cand("2019", "verbatim_start_date", "L2", parents=["Problems:"]),
            ["invalid_parent"],
        ),
    ],
)
def test_rejection_reasons(definitions, candidate, reasons):
    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(cand("Hypertension", "condition_name", "L2"), candidate),
    )
    assert [r.reasons for r in result.unmatched_entities] == [reasons]


def test_parent_must_be_grounded(definitions):
    # "Hypertension" extracted with the wrong line cannot act as a parent.
    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(
            cand("Hypertension", "condition_name", "L3"),
            cand("2019", "verbatim_start_date", "L2", parents=["Hypertension"]),
        ),
    )
    assert [r.reasons for r in result.unmatched_entities] == [
        ["text_not_in_line"],
        ["invalid_parent"],
    ]


def test_rejection_keeps_span_when_grounded(definitions):
    result = validate_extracted_entity(
        TEXT, definitions, candidates(cand("Asthma", "drug_name", "L3"))
    )
    rejected = result.unmatched_entities[0]
    assert TEXT[rejected.start : rejected.end] == "Asthma"


def test_duplicates_are_rejected(definitions):
    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(
            cand("Hypertension", "condition_name", "L2"),
            cand("Hypertension", "condition_name", "[L2]"),
        ),
    )
    assert len(result.validated_entities) == 1
    assert [r.reasons for r in result.unmatched_entities] == [["duplicate"]]
    assert result.unmatched_keys == ["verbatim_start_date"]


def test_explicit_segments_are_used(definitions):
    segments = segment_text(TEXT, max_segment_chars=30)
    # With a 30-char limit, line 4 is split at the semicolon into L4 and L5.
    assert segments[4].text == "recheck Hypertension in 2027."
    result = validate_extracted_entity(
        TEXT,
        definitions,
        candidates(cand("Hypertension", "condition_name", "L5")),
        segments=segments,
    )
    assert result.validated_entities[0].start == segments[4].start + len("recheck ")


def test_rejected_candidate_span_validation():
    RejectedEntityCandidate(text="abc", key="k", line="L1", reasons=["duplicate"])
    RejectedEntityCandidate(
        text="abc", key="k", line="L1", start=0, end=3, reasons=["duplicate"]
    )
    with pytest.raises(ValidationError):
        RejectedEntityCandidate(
            text="abc", key="k", line="L1", start=0, reasons=["duplicate"]
        )
    with pytest.raises(ValidationError):
        RejectedEntityCandidate(
            text="abc", key="k", line="L1", start=0, end=2, reasons=["duplicate"]
        )


def test_empty_text_raises(definitions):
    with pytest.raises(ValueError):
        validate_extracted_entity("", definitions, candidates())
