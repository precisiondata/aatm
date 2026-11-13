import asyncio
from types import SimpleNamespace

import pytest

from aatm.llm import reset_rate_limits, set_rate_limits
from aatm.omop import OmopRun, RunPlan
from aatm.omop.runner import (
    RunSettings,
    RunSummary,
    extraction_concurrency,
    model_location,
)

from test_omop_runner import make_pipeline, notes, run


@pytest.fixture(autouse=True)
def _reset_limits():
    reset_rate_limits()
    yield
    reset_rate_limits()


def fake(model_id, max_concurrency=None, base_url=None):
    return SimpleNamespace(
        model_id=model_id,
        max_concurrency=max_concurrency,
        llm=SimpleNamespace(base_url=base_url),
    )


# --- extraction concurrency ---------------------------------------------------------


@pytest.mark.parametrize(
    "model_id, base_url, location",
    [
        ("openai/gpt-5-mini", None, "hosted"),
        ("anthropic/claude-haiku-5-5", None, "hosted"),
        ("ollama/llama3.3", None, "local"),
        ("openai-compatible/qwen", "http://localhost:8000/v1", "local"),
        ("openai-compatible/qwen", "http://127.0.0.1:8000/v1", "local"),
        ("openai-compatible/qwen", "https://api.together.xyz/v1", "hosted"),
        ("custom/model", None, None),
    ],
)
def test_model_location(model_id, base_url, location):
    assert model_location(fake(model_id, base_url=base_url)) == location


def test_concurrency_defaults_by_location():
    hosted = extraction_concurrency(fake("openai/gpt-5-mini"))
    assert (hosted.value, hosted.source) == (16, "default")
    local = extraction_concurrency(fake("ollama/llama3.3"))
    assert (local.value, local.source) == (4, "default")
    # A custom extractor with no model id is treated as hosted.
    assert extraction_concurrency(object()).value == 16


def test_concurrency_from_argument_and_rate_limits():
    assert extraction_concurrency(fake("openai/x", 3)).source == "argument"
    set_rate_limits("openai", max_concurrency=2)
    c = extraction_concurrency(fake("openai/x", 3))
    assert (c.value, c.source) == (2, "set_rate_limits")
    assert extraction_concurrency(fake("openai/x", 1)).value == 1


def test_concurrency_derived_from_requests_per_minute():
    set_rate_limits("openai", requests_per_minute=120)
    c = extraction_concurrency(fake("openai/x"))
    assert (c.value, c.source) == (60, "set_rate_limits")
    set_rate_limits("openai", requests_per_minute=100_000)
    assert extraction_concurrency(fake("openai/x")).value == 256
    set_rate_limits("openai", requests_per_minute=2)
    assert extraction_concurrency(fake("openai/x")).value == 4


def test_run_does_not_tie_concurrency_to_chunk_size(tmp_path):
    pipeline, extractor = make_pipeline()
    active, peak = 0, 0
    call = type(extractor).__call__

    async def counting(self, texts):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return await call(self, texts)

    extractor.__class__ = type("Counting", (type(extractor),), {"__call__": counting})
    summary = run(pipeline, notes(40), tmp_path, chunk_size=200)
    assert peak == 16  # the hosted default, not chunk_size
    assert summary.settings == RunSettings(
        max_concurrency=16, chunk_size=200, batch=False, shard="0/1"
    )


# --- plan -------------------------------------------------------------------------


def hosted_pipeline(max_concurrency=None):
    pipeline, extractor = make_pipeline()
    extractor.model_id = "openai/gpt-5-mini"
    if max_concurrency is not None:
        extractor.max_concurrency = max_concurrency
    return pipeline


def test_plan_reports_settings_and_their_sources():
    plan = hosted_pipeline().plan(notes(10), chunk_size=50)
    assert isinstance(plan, RunPlan)
    assert plan.total == 10 and plan.to_extract == 10
    assert plan.setting("max_concurrency").source == "default"
    assert plan.setting("max_concurrency").value == 16
    assert (plan.setting("chunk_size").value, plan.setting("chunk_size").source) == (
        50,
        "argument",
    )
    assert plan.estimate is None
    assert plan.warnings == []
    text = str(plan)
    assert "max_concurrency" in text and "[default]" in text
    assert "warnings: none" in text


def test_plan_shows_rate_limits():
    set_rate_limits("openai", requests_per_minute=600, max_concurrency=32)
    plan = hosted_pipeline().plan(notes(10))
    assert plan.setting("max_concurrency").source == "set_rate_limits"
    assert plan.setting("requests_per_minute").value == 600


def test_plan_warns_about_low_concurrency_with_a_hosted_api():
    plan = hosted_pipeline(max_concurrency=1).plan(notes(10))
    (warning,) = plan.warnings
    assert "max_concurrency=1 with a hosted API" in warning.message
    assert "about 6 notes/min" in warning.message
    assert "max_concurrency=16" in warning.fix


def test_plan_warns_about_shards_with_a_hosted_api():
    plan = hosted_pipeline().plan(notes(10), shard=(0, 4))
    assert any("share one rate limit" in w.message for w in plan.warnings)
    assert plan.total == 3


def test_plan_warns_about_the_agent_coder_without_cache():
    pipeline = hosted_pipeline()

    class AgenticCoder:
        max_concurrency = 8

    pipeline.builder.coder = AgenticCoder()
    plan = pipeline.plan(notes(10))
    assert any("cache=True" in w.fix for w in plan.warnings)
    assert plan.setting("cache").value is False


def test_plan_warns_when_batch_is_not_supported():
    plan = hosted_pipeline().plan(notes(10), batch=True)
    assert any("batch=True needs" in w.message for w in plan.warnings)
    assert plan.setting("batch").value is True
    assert plan.setting("max_concurrency") is None


def test_plan_without_a_count():
    plan = hosted_pipeline().plan(iter(notes(10)))
    assert plan.total is None and plan.to_extract is None
    assert any("total=" in w.fix for w in plan.warnings)
    assert hosted_pipeline().plan(iter(notes(10)), total=10).to_extract == 10


def test_plan_counts_what_an_earlier_run_did(tmp_path):
    pipeline = hosted_pipeline()
    run(pipeline, notes(6), tmp_path)
    plan = pipeline.plan(notes(10), tmp_path)
    assert plan.status == {"built": 6}
    assert plan.to_extract == 4
    # The earlier run serves as the pilot.
    assert plan.estimate is not None and plan.estimate.pilot_notes == 6


def pilot_summary(extracted=100, seconds=50.0, concurrency=10, calls=100):
    summary = RunSummary(
        output_dir="pilot",
        notes=extracted,
        built=extracted,
        extracted=extracted,
        seconds=seconds,
        settings=RunSettings(max_concurrency=concurrency, chunk_size=100),
    )
    ext = summary.stages["extraction"]
    ext.items = extracted
    ext.seconds = seconds
    for _ in range(calls):
        ext.usage.add(1000, 200, "openai/gpt-5-mini")
    summary.stages["building"].seconds = 1.0
    return summary


def test_plan_estimates_from_a_pilot():
    pilot = pilot_summary()  # 5 s per call, 10 in flight
    plan = hosted_pipeline().plan(
        notes(1000), pilot=pilot, prices={"openai/gpt-5-mini": (1.0, 10.0)}
    )
    e = plan.estimate
    assert e.calls == 1000
    assert e.input_tokens == 1_000_000 and e.output_tokens == 200_000
    assert e.cost == pytest.approx(1.0 + 2.0)
    # 1000 calls of 5 s, 16 in flight.
    assert e.seconds == pytest.approx(1000 * 5 / 16)
    assert "5.0 s per extraction call" in e.basis


def test_plan_estimate_is_paced_by_requests_per_minute():
    set_rate_limits("openai", requests_per_minute=60, max_concurrency=16)
    plan = hosted_pipeline().plan(notes(1000), pilot=pilot_summary())
    assert plan.estimate.seconds == pytest.approx(1000.0)
    assert "paced by requests_per_minute" in plan.estimate.basis


def test_batch_estimate_halves_a_live_pilot_cost():
    pipeline = hosted_pipeline()
    pipeline.extractor.submit_batch = lambda *a: None
    plan = pipeline.plan(
        notes(1000),
        batch=True,
        pilot=pilot_summary(),
        prices={"openai/gpt-5-mini": (1.0, 10.0)},
    )
    assert plan.estimate.cost == pytest.approx(1.5)
    assert plan.estimate.seconds is None
    assert plan.warnings == []


def test_plan_reads_a_pilot_folder(tmp_path):
    pipeline = hosted_pipeline()
    run(pipeline, notes(4), tmp_path / "pilot")
    assert OmopRun(tmp_path / "pilot").summaries()[0].settings.max_concurrency == 16
    plan = pipeline.plan(notes(40), tmp_path / "full", pilot=tmp_path / "pilot")
    assert plan.estimate.pilot_notes == 4 and plan.estimate.notes == 40
    assert "estimate (from a pilot of 4 notes)" in str(plan)
