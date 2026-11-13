"""Run observability: tokens per model and stage, cost, progress and saved summaries."""

import asyncio
import json
import logging

import pandas as pd
import pytest

from aatm.concept_search.agent import run_sync
from aatm.llm import ModelPrice, track_usage
from aatm.llm.usage import record_usage
from aatm.omop import OmopExtractionPipeline, OmopRun, RunSummary
from aatm.omop.runner import RUNS, count_notes

from test_omop_coding import FakeCoder, UnitLookup, lab, lab_definitions
from test_omop_mapping import extract


# --- token usage ------------------------------------------------------------------


def test_usage_is_counted_per_model():
    with track_usage() as outer:
        record_usage(100, 10, model_id="openai/a")
        with track_usage() as inner:
            record_usage(200, 20, model_id="openai/a")
            record_usage(5, None, model_id="gemini/b")
        record_usage(1, 1)

    assert (outer.calls, outer.input_tokens, outer.output_tokens) == (4, 306, 31)
    assert set(outer.by_model) == {"openai/a", "gemini/b", "unknown"}
    assert outer.by_model["openai/a"].calls == 2
    assert outer.by_model["openai/a"].input_tokens == 300
    assert inner.calls == 2 and set(inner.by_model) == {"openai/a", "gemini/b"}


def test_cost_from_prices():
    with track_usage() as usage:
        record_usage(1_000_000, 100_000, model_id="openai/a")
        record_usage(2_000_000, 0, model_id="gemini/b")

    assert usage.cost(None) is None
    assert usage.cost({"openai/a": (1.0, 10.0)}) == pytest.approx(2.0)
    assert usage.unpriced({"openai/a": (1.0, 10.0)}) == ["gemini/b"]
    prices = {
        "openai/a": {"input": 1.0, "output": 10.0},
        "gemini/b": ModelPrice(input=0.5, output=1.0),
    }
    assert usage.cost(prices) == pytest.approx(3.0)
    assert usage.unpriced(prices) == []


def test_cost_reported_by_provider():
    with track_usage() as usage:
        record_usage(100, 0, model_id="openrouter/typesafe/jev-1.13", cost=0.002)
        record_usage(100, 0, model_id="openrouter/typesafe/jev-1.13", cost=0.003)
        record_usage(1_000_000, 0, model_id="openai/a")

    assert usage.reported_cost == pytest.approx(0.005)
    # Without prices, only the reported cost counts.
    assert usage.cost(None) == pytest.approx(0.005)
    assert usage.unpriced(None) == ["openai/a"]
    # A price takes precedence over the reported cost.
    prices = {"openai/a": (1.0, 0.0), "openrouter/typesafe/jev-1.13": (10.0, 0.0)}
    assert usage.cost(prices) == pytest.approx(1.0 + 0.002)

    with track_usage() as merged:
        pass
    merged.merge(usage)
    assert merged.by_model["openrouter/typesafe/jev-1.13"].reported_cost == (
        pytest.approx(0.005)
    )


def test_merge_adds_counts():
    with track_usage() as a:
        record_usage(1, 2, model_id="m")
    with track_usage() as b:
        record_usage(3, 4, model_id="m")
        record_usage(5, 6, model_id="n")
    a.merge(b)
    assert (a.calls, a.input_tokens, a.output_tokens) == (3, 9, 12)
    assert a.by_model["m"].calls == 2 and a.by_model["n"].output_tokens == 6


def test_trackers_follow_run_sync_into_its_thread():
    async def call():
        record_usage(7, 3, model_id="m")

    async def main():
        # Inside a running loop, run_sync runs the coroutine in a new thread.
        with track_usage() as usage:
            run_sync(call())
        return usage

    usage = asyncio.run(main())
    assert usage.calls == 1 and usage.input_tokens == 7


# --- runs ---------------------------------------------------------------------------


class MeteredExtractor:
    """Finds "Hb" and "glucose" in every note; reports one call per note."""

    model_id = "fake/extractor"

    def __init__(self, definitions):
        self.definitions = definitions

    async def __call__(self, texts):
        results = []
        for text in texts:
            record_usage(1000, 100, model_id=self.model_id)
            results.append(
                extract(self.definitions, text, [lab("e1", "Hb"), lab("e2", "glucose")])
            )
        return results


class MeteredCoder(FakeCoder):
    """`FakeCoder` that reports one call per text, from the build thread."""

    def code(self, texts, domain=None, vocabularies=None):
        for _ in texts:
            record_usage(500, 50, model_id="fake/coder")
        return super().code(texts, domain=domain, vocabularies=vocabularies)


def lab_pipeline():
    definitions = lab_definitions()
    return OmopExtractionPipeline(
        definitions,
        extractor=MeteredExtractor(definitions),
        vocabulary=UnitLookup(),
        coder=MeteredCoder(),
    )


def lab_notes(n):
    return [{"note_id": i, "text": "Hb 11,2 g/dL, glucose 99 MG/DL"} for i in range(n)]


def test_run_reports_each_stage(tmp_path):
    summary = asyncio.run(
        lab_pipeline().run(
            lab_notes(4),
            tmp_path,
            chunk_size=2,
            prices={"fake/extractor": (1.0, 10.0)},
        )
    )

    assert summary.completed and summary.error is None
    assert summary.total == 4 and summary.kind == "run"
    extraction, coding, building = (
        summary.stages[s] for s in ("extraction", "coding", "building")
    )
    # Extraction overlaps the build of the chunk before: its count stays its own.
    assert extraction.items == 4
    assert set(extraction.usage.by_model) == {"fake/extractor"}
    assert extraction.usage.calls == 4
    assert extraction.cost == pytest.approx(4 * (1000 * 1.0 + 100 * 10.0) / 1e6)
    # Two distinct texts per chunk, coded in the build thread.
    assert coding.items == 4
    assert set(coding.usage.by_model) == {"fake/coder"}
    assert coding.usage.calls == 4
    assert coding.cost == 0 and coding.unpriced == ["fake/coder"]
    assert building.items == 4 and building.errors == 0
    assert summary.usage.calls == 8
    assert summary.cost == pytest.approx(extraction.cost)
    assert summary.started_at is not None and summary.finished_at is not None

    text = str(summary)
    assert "extraction: 4 notes" in text and "coding: 4 texts" in text
    assert "no price for fake/coder" in text

    (saved,) = OmopRun(tmp_path).summaries()
    assert saved == summary


def test_each_call_saves_a_summary(tmp_path):
    pipeline = lab_pipeline()
    asyncio.run(pipeline.run(lab_notes(2), tmp_path))
    again = asyncio.run(pipeline.run(lab_notes(2), tmp_path))
    rebuilt = asyncio.run(pipeline.rebuild(tmp_path))

    assert again.skipped == 2 and again.usage.calls == 0
    assert rebuilt.kind == "rebuild" and rebuilt.total == 2
    assert rebuilt.stages["extraction"].items == 0
    assert rebuilt.stages["building"].items == 2
    summaries = OmopRun(tmp_path).summaries()
    assert [s.kind for s in summaries] == ["run", "run", "rebuild"]
    for path in (tmp_path / RUNS).glob("*.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["completed"] is True


@pytest.fixture
def runner_logs(caplog):
    """Capture the runner's logs, also when `configure_logging` stopped propagation."""
    logger = logging.getLogger("aatm.omop.runner")
    propagate = logger.propagate
    logger.addHandler(caplog.handler)
    logger.propagate = False  # or caplog's root handler sees every record twice
    caplog.set_level(logging.INFO, logger=logger.name)
    yield caplog
    logger.removeHandler(caplog.handler)
    logger.propagate = propagate


def progress_lines(caplog):
    return [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("Progress")
    ]


def test_progress_is_logged_with_time_left(tmp_path, runner_logs):
    asyncio.run(lab_pipeline().run(lab_notes(6), tmp_path, chunk_size=2, log_every=0))
    progress = progress_lines(runner_logs)
    assert len(progress) == 2  # after the builds of chunks 1 and 2; then the summary
    assert "2 of 6 notes done" in progress[0] and "left" in progress[0]
    assert "4 of 6 notes done" in progress[1]
    assert "LLM calls" in progress[0]
    assert any(r.getMessage().startswith("Run finished") for r in runner_logs.records)


def test_progress_is_logged_at_most_every_log_every_seconds(tmp_path, runner_logs):
    asyncio.run(
        lab_pipeline().run(lab_notes(6), tmp_path, chunk_size=1, log_every=3600)
    )
    assert len(progress_lines(runner_logs)) == 1


def test_a_stopped_run_saves_its_summary(tmp_path):
    def source():
        yield from lab_notes(2)
        raise RuntimeError("database gone")

    with pytest.raises(RuntimeError, match="database gone"):
        asyncio.run(lab_pipeline().run(source(), tmp_path, chunk_size=2))

    (saved,) = OmopRun(tmp_path).summaries()
    assert not saved.completed
    assert saved.error == "RuntimeError: database gone"
    assert saved.total is None and saved.built == 2
    assert "stopped (RuntimeError: database gone)" in str(saved)


def test_count_notes(tmp_path):
    frame = pd.DataFrame(lab_notes(3))
    frame.to_parquet(tmp_path / "notes.parquet")
    frame.to_csv(tmp_path / "notes.csv", index=False)

    assert count_notes(frame) == 3
    assert count_notes(lab_notes(5)) == 5
    assert count_notes(tmp_path / "notes.parquet") == 3
    assert count_notes(str(tmp_path / "notes.csv")) is None
    assert count_notes(iter(lab_notes(2))) is None


def test_total_can_be_given(tmp_path):
    summary = asyncio.run(lab_pipeline().run(iter(lab_notes(2)), tmp_path, total=10))
    assert summary.total == 10
    assert summary.eta_seconds is not None and summary.eta_seconds > 0


def test_summary_round_trips_through_json():
    summary = RunSummary(output_dir="x", total=5, notes=2, built=1, seconds=2.0)
    assert RunSummary.model_validate_json(summary.model_dump_json()) == summary
    assert summary.eta_seconds == pytest.approx(8.0)
