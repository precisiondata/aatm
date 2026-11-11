import json
import os
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timezone

import pytest

from aatm.llm.limits import (
    RateLimits,
    get_rate_limits,
    reset_rate_limits,
    set_rate_limits,
)
from aatm.omop import OmopExtractionPipeline, OmopRun, notes_from_sql
from aatm.omop.runner import RunSummary, StageStats
from aatm.omop.shards import (
    ShardedRunError,
    combine_summaries,
    run_sharded,
    shard_condition,
    shard_dir,
    shard_dirs,
    shard_of,
)

from test_omop_mapping import FakeLookup, ecog_definitions
from test_omop_runner import FakeExtractor, make_pipeline, notes, run


def sorted_rows(output, table="measurement"):
    df = output.table(table)
    return df.sort_values(df.columns[0], ignore_index=True)


def run_in_shards(source, parent, count, **kwargs):
    pipeline, extractor = make_pipeline()
    summaries = [
        run(pipeline, source, shard_dir(parent, i, count), shard=(i, count), **kwargs)
        for i in range(count)
    ]
    return summaries, extractor


# --- assigning notes to shards -----------------------------------------------------


def test_shard_of_is_stable_and_even():
    # Pinned: the shard of a note must not change between versions or machines.
    assert [shard_of(i, 4) for i in range(8)] == [shard_of(i, 4) for i in range(8)]
    assert shard_of(12345, 1) == 0
    pinned = [shard_of(i, 7) for i in (*range(10), 10**12)]
    assert pinned == [4, 2, 6, 3, 0, 5, 2, 6, 3, 1, 0]

    patterns = {
        "sequential": range(20_000),
        "stride 10": range(0, 200_000, 10),
        "stride 2**32": range(0, 20_000 * 2**32, 2**32),
        "site offset": range(10**12, 10**12 + 20_000),
    }
    for name, ids in patterns.items():
        for n in (2, 4, 7, 16):
            counts = Counter(shard_of(i, n) for i in ids)
            assert len(counts) == n, name
            assert max(counts.values()) / min(counts.values()) < 1.1, (name, n)


def test_shard_folders(tmp_path):
    assert shard_dir(tmp_path, 2, 8).name == "shard-002-of-008"
    assert shard_dir(tmp_path, 3, 1200).name == "shard-0003-of-1200"
    with pytest.raises(ValueError, match="outside"):
        shard_dir(tmp_path, 4, 4)
    for i in (1, 0):
        shard_dir(tmp_path, i, 2).mkdir()
    (tmp_path / "measurement").mkdir()
    assert [p.name for p in shard_dirs(tmp_path)] == [
        "shard-000-of-002",
        "shard-001-of-002",
    ]
    shard_dir(tmp_path, 0, 3).mkdir()
    with pytest.raises(ValueError, match="different splits"):
        shard_dirs(tmp_path)


# --- shards in the database ------------------------------------------------------------


def test_database_computes_the_same_shards():
    ids = [*range(500), 2**31 - 2, 2**31, 2**40 + 7, 10**12, 2**47 - 1]
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE note (note_id INTEGER)")
    con.executemany("INSERT INTO note VALUES (?)", [(i,) for i in ids])
    for n in (1, 2, 5, 16):
        for index in range(n):
            query = f"SELECT note_id FROM note WHERE {shard_condition(index, n)}"
            found = sorted(row[0] for row in con.execute(query))
            assert found == sorted(i for i in ids if shard_of(i, n) == index)
    assert shard_condition(0, 2, "n.id", cast=None).startswith("(((n.id % ")
    assert "CAST(note_id AS SIGNED)" in shard_condition(0, 2, cast="SIGNED")


def note_db(path, rows):
    import pandas as pd

    con = sqlite3.connect(path)
    frame = pd.DataFrame(rows).rename(columns={"note_id": "nid", "text": "body"})
    frame.to_sql("note", con, index=False)
    return con


def test_sql_source_fetches_only_the_shard(tmp_path):
    source_notes = notes(40)
    con = note_db(":memory:", source_notes)
    source = notes_from_sql("SELECT * FROM note;", con, chunk_size=7)
    columns = {"nid": "note_id", "body": "text"}
    assert source.sql() == "SELECT * FROM note;"
    filtered = source.sql((1, 3), columns)
    assert filtered.startswith("SELECT * FROM (SELECT * FROM note) shard_source WHERE")
    assert "CAST(nid AS BIGINT)" in filtered  # the column renamed to note_id
    assert len(list(con.execute(filtered))) == sum(
        1 for n in source_notes if shard_of(n["note_id"], 3) == 1
    )

    summaries, _ = run_in_shards(source, tmp_path, 3, chunk_size=5, columns=columns)

    assert sum(s.built for s in summaries) == 40  # the source is read again per shard
    assert sorted(OmopRun(tmp_path).provenance()["note_id"].unique()) == list(
        range(1, 41)
    )


# --- shards together equal a single run --------------------------------------------


def test_shards_hold_the_rows_of_a_single_run(tmp_path):
    source = notes(30)
    single, _ = make_pipeline()
    run(single, source, tmp_path / "single", chunk_size=4)

    summaries, extractor = run_in_shards(source, tmp_path / "sharded", 3, chunk_size=4)

    assert extractor.calls == 30  # each note extracted by one shard only
    assert sorted(s.notes for s in summaries) != [0, 0, 30]
    assert sum(s.built for s in summaries) == 30
    whole, sharded = OmopRun(tmp_path / "single"), OmopRun(tmp_path / "sharded")
    assert sharded.sharded and not whole.sharded
    assert sharded.tables == ["measurement"]
    merged = sorted_rows(sharded)
    assert merged.equals(sorted_rows(whole))
    assert merged["measurement_id"].is_unique
    assert len(sharded.provenance()) == len(whole.provenance()) == len(merged)
    assert len(sharded.mappings()) == len(whole.mappings())
    assert sharded.status() == {"built": 30}
    assert len(sharded.summaries()) == 3
    assert sharded.dataset("measurement").count_rows() == len(merged)


def test_shard_reads_only_its_notes_and_resumes(tmp_path):
    source = notes(20)
    folder = shard_dir(tmp_path, 1, 2)
    pipeline, extractor = make_pipeline()

    first = run(pipeline, source, folder, shard=(1, 2), chunk_size=3)
    mine = [n["note_id"] for n in source if shard_of(n["note_id"], 2) == 1]
    assert first.notes == first.built == len(mine)
    assert first.total == 10  # the source's 20 notes, split in two
    assert set(OmopRun(folder).provenance()["note_id"]) == set(mine)

    again = run(pipeline, source, folder, shard=(1, 2), chunk_size=3)
    assert again.skipped == len(mine) and again.extracted == 0
    assert extractor.calls == len(mine)


def test_folders_refuse_another_shard(tmp_path):
    pipeline, _ = make_pipeline()
    folder = tmp_path / "out"
    run(pipeline, notes(4), folder, shard=(0, 2))
    with pytest.raises(ValueError, match="holds shard 0 of 2, not shard 1 of 2"):
        run(pipeline, notes(4), folder, shard=(1, 2))
    with pytest.raises(ValueError, match="not the whole dataset"):
        run(pipeline, notes(4), folder)

    whole = tmp_path / "whole"
    run(pipeline, notes(4), whole)
    with pytest.raises(ValueError, match="holds the whole dataset"):
        run(pipeline, notes(4), whole, shard=(0, 2))
    with pytest.raises(ValueError, match="outside"):
        run(pipeline, notes(4), tmp_path / "x", shard=(2, 2))


def test_parent_of_shards_refuses_an_unsharded_run(tmp_path):
    run_in_shards(notes(6), tmp_path, 2)
    pipeline, _ = make_pipeline()
    with pytest.raises(ValueError, match="sharded run"):
        run(pipeline, notes(6), tmp_path)


def test_rebuild_of_the_parent_rebuilds_every_shard(tmp_path):
    import asyncio

    run_in_shards(notes(10), tmp_path, 2, chunk_size=3)
    before = sorted_rows(OmopRun(tmp_path))
    pipeline, extractor = make_pipeline()

    summary = asyncio.run(pipeline.rebuild(tmp_path))

    assert summary.kind == "rebuild"
    assert summary.built == 10 and extractor.calls == 0
    assert not (tmp_path / "ledger.db").exists()
    assert sorted_rows(OmopRun(tmp_path)).equals(before)


def test_failures_of_all_shards(tmp_path):
    source = notes(10)
    bad = {source[i]["text"] for i in (2, 7)}
    pipeline, _ = make_pipeline(fail=bad)
    for i in range(2):
        run(pipeline, source, shard_dir(tmp_path, i, 2), shard=(i, 2))
    output = OmopRun(tmp_path)
    assert output.failures()["note_id"].tolist() == [3, 8]
    assert output.status() == {"built": 8, "failed": 2}


# --- summaries and rate limits ------------------------------------------------------


def test_combine_summaries():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    a = RunSummary(
        output_dir="a",
        started_at=t0,
        finished_at=t0.replace(minute=10),
        completed=True,
        total=5,
        notes=5,
        built=4,
        failed=1,
        rows={"measurement": 6},
        cost=1.0,
        stages={"extraction": StageStats(items=5, seconds=30.0, cost=1.0)},
    )
    b = RunSummary(
        output_dir="b",
        started_at=t0.replace(minute=1),
        finished_at=t0.replace(minute=20),
        completed=False,
        error="boom",
        total=6,
        notes=6,
        built=6,
        rows={"measurement": 2, "observation": 1},
        stages={"extraction": StageStats(items=6, seconds=40.0)},
    )
    c = combine_summaries([a, b], "parent")
    assert (c.notes, c.built, c.failed, c.total) == (11, 10, 1, 11)
    assert c.rows == {"measurement": 8, "observation": 1}
    assert c.stages["extraction"].items == 11
    assert c.stages["extraction"].seconds == 70.0
    assert c.cost == 1.0
    assert not c.completed and c.error == "boom"
    assert c.seconds == 20 * 60  # wall time, not the sum
    assert c.output_dir == "parent"


def test_rate_limits_divided():
    limits = RateLimits(
        max_concurrency=10, requests_per_minute=1000, tokens_per_minute=9e5
    )
    third = limits.divided(3)
    assert third.max_concurrency == 4  # rounded up
    assert third.requests_per_minute == pytest.approx(1000 / 3)
    assert third.tokens_per_minute == pytest.approx(3e5)
    assert third.max_retries == limits.max_retries
    assert RateLimits().divided(4) == RateLimits()
    assert RateLimits(max_concurrency=2).divided(8).max_concurrency == 1
    with pytest.raises(ValueError):
        limits.divided(0)


# --- run_sharded, in processes --------------------------------------------------------

LIMITS_LOG = "AATM_TEST_LIMITS_LOG"


def build_pipeline():
    """The factory of run_sharded: importable by the worker processes."""
    log = os.environ.get(LIMITS_LOG)
    if log:  # one file per call: appends from several processes may be lost
        limits = get_rate_limits("openai/gpt-5-mini")
        path = os.path.join(log, f"{os.getpid()}-{uuid.uuid4().hex}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(limits.model_dump_json())
    definitions = ecog_definitions()
    return OmopExtractionPipeline(
        definitions, extractor=FakeExtractor(definitions), vocabulary=FakeLookup()
    )


def test_run_sharded_in_processes(tmp_path, monkeypatch):
    source = notes(24)
    log = tmp_path / "limits"
    log.mkdir()
    monkeypatch.setenv(LIMITS_LOG, str(log))
    reset_rate_limits()
    set_rate_limits("openai", requests_per_minute=600, max_concurrency=8)
    try:
        summary = run_sharded(
            build_pipeline, source, tmp_path / "out", num_shards=3, chunk_size=5
        )
    finally:
        reset_rate_limits()

    assert summary.completed and summary.built == 24 and summary.total == 24
    assert summary.rows == {"measurement": 36}
    seen = [json.loads(path.read_text()) for path in log.iterdir()]
    assert len(seen) == 3
    assert {(s["requests_per_minute"], s["max_concurrency"]) for s in seen} == {
        (200.0, 3)
    }
    output = OmopRun(tmp_path / "out")
    single, _ = make_pipeline()
    run(single, source, tmp_path / "single")
    assert sorted_rows(output).equals(sorted_rows(OmopRun(tmp_path / "single")))

    # Run again: every shard resumes, nothing is extracted twice.
    monkeypatch.delenv(LIMITS_LOG)
    again = run_sharded(build_pipeline, source, tmp_path / "out", num_shards=3)
    assert again.skipped == 24 and again.extracted == 0
    with pytest.raises(ValueError, match="split into 3"):
        run_sharded(build_pipeline, source, tmp_path / "out", num_shards=2)


NOTE_DB = "AATM_TEST_NOTE_DB"


def sql_source():
    """A source function: each worker opens its own connection."""
    con = sqlite3.connect(os.environ[NOTE_DB])
    return notes_from_sql("SELECT nid AS note_id, body AS text FROM note", con)


def test_run_sharded_from_a_database(tmp_path, monkeypatch):
    path = tmp_path / "notes.db"
    note_db(path, notes(12)).close()
    monkeypatch.setenv(NOTE_DB, str(path))

    summary = run_sharded(build_pipeline, sql_source, tmp_path / "out", num_shards=2)

    assert summary.completed and summary.built == 12
    assert summary.total is None  # unknown for queries


def test_run_sharded_reports_a_failed_shard(tmp_path):
    out = tmp_path / "out"
    # Shard 0's folder already holds shard 1: shard 0 is refused, shard 1 runs.
    pipeline, _ = make_pipeline()
    run(pipeline, notes(6), shard_dir(out, 0, 2), shard=(1, 2))

    with pytest.raises(ShardedRunError) as caught:
        run_sharded(build_pipeline, notes(6), out, num_shards=2, processes=1)

    assert list(caught.value.errors) == [0]
    assert "holds shard 1 of 2" in caught.value.errors[0]
    summary = caught.value.summary
    assert not summary.completed
    assert summary.built == sum(1 for n in notes(6) if shard_of(n["note_id"], 2) == 1)
