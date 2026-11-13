import asyncio
import json

import pandas as pd
import pytest

from aatm.omop import OmopExtractionPipeline, Pilot
from aatm.omop.pilot import NOTES_FILE, SAMPLE_FILE, _diff_codes, _diff_mentions
from aatm.omop.mapping import ClinicalNote

from test_omop_mapping import FakeLookup, ecog, ecog_definitions, extract


class WordingExtractor:
    """Finds "ECOG <grade>"; also "PS <grade>" once the definition mentions PS."""

    def __init__(self, definitions, fail=()):
        self.definitions = definitions
        self.fail = set(fail)
        self.calls = 0
        wording = definitions.get("ecog").definition
        self.prefixes = ("ECOG ", "PS ") if "PS" in wording else ("ECOG ",)

    async def __call__(self, texts):
        self.calls += len(texts)
        results = []
        for text in texts:
            if text in self.fail:
                raise RuntimeError("API down")
            entities = []
            for i, line in enumerate(text.split("\n"), start=1):
                for word in line.split(". "):
                    word = word.strip(". ")
                    if word.startswith(self.prefixes):
                        entities.append(
                            ecog(f"e{len(entities) + 1}", word, word[-1], f"L{i}")
                        )
            results.append(extract(self.definitions, text, entities))
        return results


def definitions(wording="An ECOG performance status.", **fields):
    defs = ecog_definitions(**fields)
    defs.entities[0].definition = wording
    return defs


def make_pipeline(defs=None, fail=()):
    defs = defs or definitions()
    extractor = WordingExtractor(defs, fail)
    pipeline = OmopExtractionPipeline(
        defs, extractor=extractor, vocabulary=FakeLookup()
    )
    return pipeline, extractor


def notes(n):
    return [
        {
            "note_id": i,
            "text": f"Visit {i}\nECOG 1. PS 2" if i % 2 else f"Visit {i}\nECOG 0",
            "person_id": 100 + i,
            "note_date": "2025-01-01",
        }
        for i in range(1, n + 1)
    ]


def run(pilot, pipeline, **kwargs):
    return asyncio.run(pilot.run(pipeline, **kwargs))


# --- the sample -----------------------------------------------------------------------


def test_the_sample_is_picked_once_and_kept(tmp_path):
    pilot = Pilot(tmp_path, source=notes(30), sample=5, seed=1)
    ids = pilot.sample.note_ids

    assert len(ids) == 5 and ids == sorted(ids)
    assert pilot.sample.source_notes == 30
    stored = json.loads((tmp_path / SAMPLE_FILE).read_text())
    assert stored["note_ids"] == ids
    assert len((tmp_path / NOTES_FILE).read_text().splitlines()) == 5
    # The same seed picks the same notes; the stored sample needs no source.
    assert (
        Pilot(tmp_path / "other", source=notes(30), sample=5, seed=1).sample.note_ids
        == ids
    )
    assert Pilot(tmp_path).sample.note_ids == ids
    assert [n.note_id for n in Pilot(tmp_path).notes()] == ids


def test_a_sample_of_another_size_needs_another_folder(tmp_path):
    Pilot(tmp_path, source=notes(10), sample=3).sample
    with pytest.raises(ValueError, match="sample of 3 notes"):
        Pilot(tmp_path, source=notes(10), sample=4)
    with pytest.raises(ValueError, match="seed=2"):
        Pilot(tmp_path, seed=2)


def test_a_small_source_gives_all_its_notes(tmp_path):
    assert Pilot(tmp_path, source=notes(3), sample=50).sample.note_ids == [1, 2, 3]


def test_a_pilot_without_sample_needs_a_source(tmp_path):
    with pytest.raises(ValueError, match="source"):
        Pilot(tmp_path).notes()


# --- versions -------------------------------------------------------------------------


def test_a_new_wording_makes_a_new_version(tmp_path):
    pilot = Pilot(tmp_path, source=notes(6), sample=4)
    first = run(pilot, make_pipeline()[0])
    pipeline, extractor = make_pipeline(
        definitions("An ECOG or PS performance status.")
    )
    second = run(pilot, pipeline)

    assert (first.name[:4], second.name[:4]) == ("v001", "v002")
    assert (first.outcome, second.outcome) == ("new", "new")
    assert extractor.calls == 4
    assert second.previous == first.name and second.action == "run"
    assert second.changes == ["definitions: 1 line added, 1 removed"]
    assert "+  definition: An ECOG or PS performance status." in second.definitions_diff
    assert [v.name for v in pilot.log()] == [first.name, second.name]
    assert "ECOG or PS" in second.definitions()
    assert second.summary.built == 4
    assert len(second.output().table("measurement")) > len(
        first.output().table("measurement")
    )


def test_nothing_changed_reuses_the_version(tmp_path):
    pilot = Pilot(tmp_path, source=notes(4), sample=4)
    first = run(pilot, make_pipeline()[0])
    pipeline, extractor = make_pipeline()
    again = run(pilot, pipeline)

    assert again.outcome == "reused" and again.name == first.name
    assert extractor.calls == 0
    assert len(pilot.versions()) == 1


def test_going_back_to_an_earlier_version_reuses_it(tmp_path):
    pilot = Pilot(tmp_path, source=notes(4), sample=4)
    first = run(pilot, make_pipeline()[0])
    run(pilot, make_pipeline(definitions("Another wording."))[0])
    assert run(pilot, make_pipeline()[0]).name == first.name


def test_a_new_mapping_rebuilds_with_no_extraction_call(tmp_path):
    pilot = Pilot(tmp_path, source=notes(4), sample=4)
    first = run(pilot, make_pipeline()[0])
    changed = definitions(measurement_type_concept_id={"value": 0})
    pipeline, extractor = make_pipeline(changed)
    second = run(pilot, pipeline)

    assert extractor.calls == 0
    assert second.outcome == "rebuilt"
    assert (second.action, second.based_on) == ("rebuild", first.name)
    assert second.summary.kind == "rebuild" and second.summary.built == 4
    measurement = second.output().table("measurement")
    assert set(measurement["measurement_type_concept_id"]) == {0}
    # The first version keeps its own tables.
    assert set(first.output().table("measurement")["measurement_type_concept_id"]) == {
        32858
    }
    assert second.changes == ["definitions: 1 line added, 1 removed"]
    # And the rebuilt version is reused in turn.
    assert run(pilot, make_pipeline(changed)[0]).outcome == "reused"


def test_a_version_that_stopped_is_finished(tmp_path):
    pilot = Pilot(tmp_path, source=notes(4), sample=4)
    failing = pilot.notes()[0].text
    first = run(pilot, make_pipeline(fail={failing})[0])
    assert first.summary.failed == 1

    pipeline, extractor = make_pipeline()
    again = run(pilot, pipeline)
    assert again.outcome == "resumed" and again.name == first.name
    assert extractor.calls == 1


# --- comparing ------------------------------------------------------------------------


def test_compare_shows_rows_issues_and_mentions(tmp_path):
    pilot = Pilot(tmp_path, source=notes(4), sample=4)
    run(pilot, make_pipeline()[0])
    run(pilot, make_pipeline(definitions("An ECOG or PS performance status."))[0])

    comparison = pilot.compare("v001", "v002")

    assert comparison.notes == {"a": 4, "b": 4, "both": 4}
    rows = comparison.rows.loc["measurement"]
    assert (rows["change"], rows.iloc[1] - rows.iloc[0]) == (2, 2)
    mentions = comparison.mentions
    assert list(mentions["change"]) == ["added", "added"]
    assert list(mentions["note_id"]) == [1, 3]
    assert set(mentions["text_b"]) == {"PS 2"}
    assert set(mentions["line"]) == {"ECOG 1. PS 2"}
    assert set(mentions["line_number"]) == {2}
    assert set(mentions["attributes_b"]) == {"score=2"}
    assert list(comparison.usage.columns) == [
        pilot.version(1).name,
        pilot.version(2).name,
    ]
    assert comparison.usage.loc["notes"].tolist() == [4, 4]

    text = str(comparison)
    assert "definitions: 1 line added, 1 removed" in text
    assert "mentions: 2 added, 0 removed, 0 changed" in text
    assert "+ ecog 'PS 2' [16:20] {score=2}  (line 2: 'ECOG 1. PS 2')" in text
    # By default: the latest version and the one before.
    assert pilot.compare().mentions.equals(mentions)


def test_compare_needs_two_versions(tmp_path):
    pilot = Pilot(tmp_path, source=notes(2), sample=2)
    run(pilot, make_pipeline()[0])
    with pytest.raises(ValueError, match="two versions"):
        pilot.compare()
    with pytest.raises(KeyError, match="v009"):
        pilot.compare("v001", "v009")


def test_mentions_with_another_span_or_label_are_changed():
    note = ClinicalNote(note_id=1, text="first line\nECOG 1 today")

    def mention(start, end, text, attributes="score=1"):
        return {
            "entity": "ecog",
            "start": start,
            "end": end,
            "text": text,
            "attributes": attributes,
        }

    rows = _diff_mentions(
        note,
        [mention(11, 17, "ECOG 1"), mention(0, 5, "first")],
        [mention(11, 23, "ECOG 1 today"), mention(0, 5, "first", "score=0")],
    )
    assert [(r["change"], r["text_a"], r["text_b"]) for r in rows] == [
        ("changed", "first", "first"),
        ("changed", "ECOG 1", "ECOG 1 today"),
    ]
    assert rows[1]["line"] == "ECOG 1 today" and rows[1]["line_number"] == 2


def test_codes_that_changed():
    def codes(*rows):
        return pd.DataFrame(
            rows,
            columns=["table", "column", "text", "concept_id", "concept_name", "stored"],
        )

    old = codes(
        ("condition_occurrence", "condition_concept_id", "HTN", 1, "Hypertension", 1),
        ("condition_occurrence", "condition_concept_id", "DM", 2, "Diabetes", 2),
        ("condition_occurrence", "condition_concept_id", "CKD", 3, "CKD", 3),
    )
    new = codes(
        ("condition_occurrence", "condition_concept_id", "HTN", 1, "Hypertension", 1),
        ("condition_occurrence", "condition_concept_id", "DM", 4, "Type 2 DM", 4),
        ("condition_occurrence", "condition_concept_id", "COPD", 5, "COPD", 5),
    )
    diff = _diff_codes(old, new)
    assert list(zip(diff["text"], diff["change"])) == [
        ("DM", "changed"),
        ("COPD", "added"),
        ("CKD", "removed"),
    ]
    assert diff.loc[0, ["concept_id_a", "concept_id_b"]].tolist() == [2, 4]


# --- coding ---------------------------------------------------------------------------


class LabExtractor:
    """Finds the lab tests of `test_omop_cache.hb_entities` in every note."""

    def __init__(self, definitions):
        self.definitions = definitions
        self.calls = 0

    async def __call__(self, texts):
        from test_omop_cache import hb_entities

        self.calls += len(texts)
        return [extract(self.definitions, text, hb_entities()) for text in texts]


def lab_pipeline(coder):
    from test_omop_coding import UnitLookup, lab_definitions

    defs = lab_definitions()
    extractor = LabExtractor(defs)
    pipeline = OmopExtractionPipeline(
        defs, extractor=extractor, vocabulary=UnitLookup(), coder=coder
    )
    return pipeline, extractor


def lab_notes(n):
    from test_omop_coding import NOTE

    return [
        {"note_id": i, "text": NOTE, "person_id": i, "note_date": "2025-06-01"}
        for i in range(1, n + 1)
    ]


def test_another_coder_rebuilds_and_the_cache_is_shared(tmp_path):
    from test_omop_cache import IdentifiedCoder
    from aatm.omop import CodedText

    pilot = Pilot(tmp_path, source=lab_notes(3), sample=3)
    first_coder = IdentifiedCoder(model="m1")
    first = run(pilot, lab_pipeline(first_coder)[0])
    assert len(first_coder.calls) == 1
    assert (tmp_path / "coding_cache.db").exists()
    # The pipeline gets its coder back, uncached.
    assert lab_pipeline(first_coder)[0].builder.cache is None

    # The same coder again, in a new pipeline: nothing to do.
    same = IdentifiedCoder(model="m1")
    assert run(pilot, lab_pipeline(same)[0]).outcome == "reused"

    known = dict(IdentifiedCoder().known)
    known["glucose"] = CodedText(text="glucose", concept_id=3000963, score=0.8)
    other = IdentifiedCoder(known, model="m2")
    pipeline, extractor = lab_pipeline(other)
    second = run(pilot, pipeline)

    assert extractor.calls == 0 and second.outcome == "rebuilt"
    assert second.changes == ["coder settings (IdentifiedCoder)"]
    codes = pilot.compare().mappings
    assert list(codes["text"]) == ["glucose"]
    assert codes.loc[0, ["concept_id_a", "concept_id_b"]].tolist() == [3004501, 3000963]
    assert "measurement.measurement_concept_id 'glucose'" in str(pilot.compare())

    # Back to the first coder: its version is reused, its codes come from the cache.
    assert run(pilot, lab_pipeline(IdentifiedCoder(model="m1"))[0]).name == first.name


# --- the command line -----------------------------------------------------------------


def test_cli_log_and_compare(tmp_path):
    from typer.testing import CliRunner

    from aatm.main import app

    pilot = Pilot(tmp_path, source=notes(4), sample=4)
    run(pilot, make_pipeline()[0])
    run(pilot, make_pipeline(definitions("An ECOG or PS performance status."))[0])
    runner = CliRunner()

    result = runner.invoke(app, ["omop", "pilot", "log", "-p", str(tmp_path), "--diff"])
    assert result.exit_code == 0, result.output
    assert "first version" in result.output
    assert "definitions: 1 line added, 1 removed" in result.output
    assert "+  definition: An ECOG or PS performance status." in result.output

    result = runner.invoke(
        app, ["omop", "pilot", "compare", "v001", "v002", "-p", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "mentions: 2 added, 0 removed, 0 changed" in result.output

    result = runner.invoke(
        app, ["omop", "pilot", "compare", "v001", "v7", "-p", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "No version 'v7'" in result.output

    result = runner.invoke(app, ["omop", "pilot", "log", "-p", str(tmp_path / "none")])
    assert result.exit_code == 1
