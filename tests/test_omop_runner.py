import asyncio
import sqlite3
from datetime import date

import pandas as pd
import pytest

from aatm.omop import ClinicalNote, OmopExtractionPipeline, OmopRun, notes_from_sql
from aatm.omop.builder import MAX_NOTE_ID, OmopTableBuilder, note_row_id
from aatm.omop.runner import LEDGER_FILE, RunLedger, _part_name, _schema
from aatm.omop.tables import OMOP_TABLES

from test_omop_mapping import FakeLookup, ecog, ecog_definitions, extract


class FakeExtractor:
    """Finds every "ECOG <grade>" in a note; fails for texts in `fail`."""

    def __init__(self, definitions, fail=()):
        self.definitions = definitions
        self.fail = set(fail)
        self.calls = 0

    async def __call__(self, texts):
        self.calls += len(texts)
        results = []
        for text in texts:
            if text in self.fail:
                raise RuntimeError(f"API down for {text!r}")
            entities = []
            for i, line in enumerate(text.split("\n"), start=1):
                for word in line.split(". "):
                    word = word.strip(". ")
                    if word.startswith("ECOG "):
                        entities.append(
                            ecog(f"e{len(entities) + 1}", word, word[-1], f"L{i}")
                        )
            results.append(extract(self.definitions, text, entities))
        return results


def make_pipeline(definitions=None, fail=()):
    definitions = definitions or ecog_definitions()
    extractor = FakeExtractor(definitions, fail)
    pipeline = OmopExtractionPipeline(
        definitions, extractor=extractor, vocabulary=FakeLookup()
    )
    return pipeline, extractor


def notes(n, start=1):
    return [
        {
            "note_id": i,
            "text": f"Visit {i}. ECOG 1. ECOG 0" if i % 2 else f"Visit {i}. ECOG 1",
            "person_id": 100 + i,
            "note_date": "2025-01-01",
        }
        for i in range(start, start + n)
    ]


def run(pipeline, source, out, **kwargs):
    return asyncio.run(pipeline.run(source, out, **kwargs))


# --- row ids ------------------------------------------------------------------------


def test_note_row_ids():
    assert note_row_id(1, 1) == 65537
    assert note_row_id(0, 3) == 3
    assert note_row_id(MAX_NOTE_ID, 65535) == 2**63 - 1
    with pytest.raises(ValueError, match="note_id"):
        note_row_id(None, 1)
    with pytest.raises(ValueError, match="outside"):
        note_row_id(MAX_NOTE_ID + 1, 1)
    with pytest.raises(ValueError, match="more than 65535"):
        note_row_id(1, 65536)


def test_build_with_note_row_ids():
    defs = ecog_definitions()
    builder = OmopTableBuilder(defs, vocabulary=FakeLookup())
    text = "ECOG 1. ECOG 0"
    result = extract(defs, text, [ecog("e1", "ECOG 1", "1"), ecog("e2", "ECOG 0")])
    note = ClinicalNote(text=text, note_id=7, person_id=1, note_date=date(2025, 1, 1))

    tables = builder.build([note], [result], row_ids="note")

    ids = tables["measurement"]["measurement_id"].tolist()
    assert ids == [7 * 65536 + 1, 7 * 65536 + 2]
    with pytest.raises(ValueError, match="note_id"):
        builder.build([text], [result], row_ids="note")


def test_every_table_has_a_parquet_schema():
    for model in OMOP_TABLES.values():
        assert len(_schema(model)) == len(model.model_fields)


# --- running and reading ------------------------------------------------------------


def test_run_writes_tables_incrementally(tmp_path):
    pipeline, extractor = make_pipeline()

    summary = run(pipeline, notes(5), tmp_path, chunk_size=2)

    assert (summary.notes, summary.built, summary.extracted) == (5, 5, 5)
    assert summary.rows == {"measurement": 8}
    assert extractor.calls == 5
    # One part file per chunk.
    assert len(list((tmp_path / "measurement").glob("part-*.parquet"))) == 3

    output = OmopRun(tmp_path)
    assert output.tables == ["measurement"]
    df = output.table("measurement")
    assert len(df) == 8
    assert df["measurement_id"].dtype == "Int64"
    assert sorted(df["measurement_id"])[:3] == [65537, 65538, 2 * 65536 + 1]
    assert set(df["person_id"]) == {101, 102, 103, 104, 105}
    assert set(df["measurement_date"]) == {date(2025, 1, 1)}

    provenance = output.provenance()
    row = provenance[provenance["row_id"] == 65538].iloc[0]
    assert (row["table"], row["note_id"], row["entity_id"], row["text"]) == (
        "measurement",
        1,
        "e2",
        "ECOG 0",
    )
    assert output.status() == {"built": 5}
    assert output.failures().empty
    assert "5 notes" in str(summary)


def test_rerun_skips_built_notes(tmp_path):
    pipeline, extractor = make_pipeline()
    run(pipeline, notes(4), tmp_path, chunk_size=3)

    summary = run(pipeline, notes(6), tmp_path, chunk_size=3)

    assert (summary.skipped, summary.extracted, summary.built) == (4, 2, 2)
    assert extractor.calls == 6
    df = OmopRun(tmp_path).table("measurement")
    assert df["measurement_id"].is_unique
    assert len(df) == 9


def test_stopped_run_resumes_without_duplicates(tmp_path):
    pipeline, extractor = make_pipeline()

    def stops_after_three():
        yield from notes(3)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(pipeline, stops_after_three(), tmp_path, chunk_size=2)
    # The first chunk was written; the second never completed.
    assert OmopRun(tmp_path).status() == {"built": 2}

    summary = run(pipeline, notes(5), tmp_path, chunk_size=2)

    assert (summary.skipped, summary.built) == (2, 3)
    assert extractor.calls == 5
    df = OmopRun(tmp_path).table("measurement")
    assert len(df) == 8 and df["measurement_id"].is_unique


def test_extracted_notes_are_not_extracted_again(tmp_path):
    """A stop between extraction and writing keeps the extraction results."""
    pipeline, extractor = make_pipeline()
    original = pipeline.builder.build

    def crash(*args, **kwargs):
        raise KeyboardInterrupt  # not caught: the run stops

    pipeline.builder.build = crash
    with pytest.raises(KeyboardInterrupt):
        run(pipeline, notes(2), tmp_path)
    assert OmopRun(tmp_path).status() == {"extracted": 2}

    pipeline.builder.build = original
    summary = run(pipeline, notes(2), tmp_path)

    assert (summary.extracted, summary.reused, summary.built) == (0, 2, 2)
    assert extractor.calls == 2


def test_incomplete_parts_are_deleted(tmp_path):
    pipeline, _ = make_pipeline()
    run(pipeline, notes(2), tmp_path)
    ledger = RunLedger(tmp_path / LEDGER_FILE)
    part = ledger.new_part()  # a part whose writing never finished
    ledger.close()
    leftover = tmp_path / "measurement" / _part_name(part)
    leftover.write_bytes(b"partial")

    run(pipeline, notes(2), tmp_path)

    assert not leftover.exists()
    assert len(OmopRun(tmp_path).table("measurement")) == 3


def test_failed_notes_are_recorded_and_retried(tmp_path):
    bad = notes(3)[1]["text"]
    pipeline, extractor = make_pipeline(fail=[bad])

    summary = run(pipeline, notes(3), tmp_path, max_attempts=2)

    assert (summary.built, summary.failed) == (2, 1)
    failures = OmopRun(tmp_path).failures()
    assert failures["note_id"].tolist() == [2]
    assert failures["stage"].tolist() == ["extract"]
    assert "API down" in failures["error"][0]

    # Tried again, and given up after max_attempts.
    assert run(pipeline, notes(3), tmp_path, max_attempts=2).failed == 1
    summary = run(pipeline, notes(3), tmp_path, max_attempts=2)
    assert (summary.gave_up, summary.skipped, summary.failed) == (1, 2, 0)
    assert extractor.calls == 4  # 3, then note 2 once more

    extractor.fail.clear()
    summary = run(pipeline, notes(3), tmp_path, max_attempts=3)
    assert summary.built == 1
    assert OmopRun(tmp_path).status() == {"built": 3}


def test_build_failures_keep_the_extraction(tmp_path):
    pipeline, extractor = make_pipeline()
    original = pipeline.builder.build

    def fails_for_note_2(notes, results, **kwargs):
        if any(n.note_id == 2 for n in notes):
            raise RuntimeError("cannot build note 2")
        return original(notes, results, **kwargs)

    pipeline.builder.build = fails_for_note_2
    summary = run(pipeline, notes(3), tmp_path)
    assert (summary.built, summary.failed) == (2, 1)
    assert OmopRun(tmp_path).failures()["stage"].tolist() == ["build"]

    pipeline.builder.build = original
    summary = run(pipeline, notes(3), tmp_path)
    assert (summary.built, summary.extracted, summary.reused) == (1, 0, 1)
    assert extractor.calls == 3


def test_notes_need_ids(tmp_path):
    pipeline, _ = make_pipeline()
    with pytest.raises(ValueError, match="note_id"):
        run(pipeline, [{"text": "ECOG 1"}], tmp_path)
    with pytest.raises(TypeError, match="plain strings"):
        run(pipeline, ["ECOG 1"], tmp_path)


def test_duplicate_note_ids_in_a_chunk(tmp_path):
    pipeline, extractor = make_pipeline()
    summary = run(pipeline, notes(2) + notes(1), tmp_path, chunk_size=10)
    assert (summary.duplicates, summary.built) == (1, 2)
    assert extractor.calls == 2


# --- rebuilding ---------------------------------------------------------------------


def test_rebuild_makes_no_extraction_calls(tmp_path):
    pipeline, extractor = make_pipeline()
    run(pipeline, notes(4), tmp_path, chunk_size=3)
    before = OmopRun(tmp_path).table("measurement")

    # New mappings: refused until the tables are rebuilt.
    changed = ecog_definitions(measurement_type_concept_id={"value": 0})
    new_pipeline, new_extractor = make_pipeline(changed)
    with pytest.raises(ValueError, match="rebuild"):
        run(new_pipeline, notes(4), tmp_path)

    summary = asyncio.run(new_pipeline.rebuild(tmp_path, chunk_size=3))

    assert new_extractor.calls == 0
    assert (summary.built, summary.reused) == (4, 4)
    after = OmopRun(tmp_path).table("measurement")
    assert sorted(after["measurement_id"]) == sorted(before["measurement_id"])
    assert set(after["measurement_type_concept_id"]) == {0}
    assert len(list((tmp_path / "measurement").glob("part-*.parquet"))) == 2
    # And runs go on with the new mappings.
    assert run(new_pipeline, notes(5), tmp_path).built == 1


def test_other_definitions_need_a_new_folder(tmp_path):
    pipeline, _ = make_pipeline()
    run(pipeline, notes(1), tmp_path)
    defs = ecog_definitions()
    defs.entities[0].definition = "Another definition."
    other, _ = make_pipeline(defs)
    with pytest.raises(ValueError, match="other entity definitions"):
        run(other, notes(1), tmp_path)


# --- sources ------------------------------------------------------------------------


def frame():
    return (
        pd.DataFrame(notes(3))
        .rename(columns={"text": "note_text"})
        .assign(note_date=pd.to_datetime("2025-02-03"), other="ignored")
    )


@pytest.mark.parametrize("kind", ["csv", "parquet", "folder", "dataframe", "sql"])
def test_sources(tmp_path, kind):
    data = frame()
    if kind == "csv":
        source = tmp_path / "notes.csv"
        data.to_csv(source, index=False)
    elif kind == "parquet":
        source = tmp_path / "notes.parquet"
        data.to_parquet(source)
    elif kind == "folder":
        source = tmp_path / "notes"
        source.mkdir()
        data.iloc[:2].to_parquet(source / "a.parquet")
        data.iloc[2:].to_parquet(source / "b.parquet")
    elif kind == "dataframe":
        source = data
    else:
        con = sqlite3.connect(":memory:")
        data.to_sql("note", con, index=False)
        source = notes_from_sql("SELECT * FROM note", con, chunk_size=2)
    pipeline, _ = make_pipeline()

    summary = run(
        pipeline, source, tmp_path / "out", chunk_size=2, columns={"note_text": "text"}
    )

    assert summary.built == 3
    df = OmopRun(tmp_path / "out").table("measurement")
    assert set(df["person_id"]) == {101, 102, 103}
    assert set(df["measurement_date"]) == {date(2025, 2, 3)}


def test_source_without_text_column(tmp_path):
    pipeline, _ = make_pipeline()
    with pytest.raises(ValueError, match="no 'text' column"):
        run(pipeline, frame(), tmp_path)
