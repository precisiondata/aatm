"""Tests for the shared rate limiter and retries in `aatm.llm.limits`. Offline."""

import asyncio
import time
from types import SimpleNamespace

import anthropic
import httpx
import openai
import pytest
from pydantic import BaseModel

from aatm.data_models.extractors import EntityDefinition, EntityDefinitions
from aatm.extractors import SimpleLLMExtractor
from aatm.llm import (
    LLMResponseError,
    RateLimits,
    get_rate_limiter,
    get_rate_limits,
    reset_rate_limits,
    set_rate_limits,
)
from aatm.llm.anthropic import AnthropicLLM
from aatm.llm.limits import RateLimiter, retry_after_seconds, retry_info
from aatm.llm.openai import OpenAICompatibleLLM, OpenAILLM
from aatm.terminology_mapper import arate_limit

FAST = {"initial_backoff": 0.001, "max_backoff": 0.002}


@pytest.fixture(autouse=True)
def _reset_limits():
    reset_rate_limits()
    yield
    reset_rate_limits()


def http_error(cls, status: int, headers: dict | None = None):
    request = httpx.Request("POST", "https://api.example.test/v1")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls("error", response=response, body=None)


class Answer(BaseModel):
    value: str


# --- error classification ------------------------------------------------------------


@pytest.mark.parametrize(
    "error, retryable, status",
    [
        (http_error(openai.RateLimitError, 429), True, 429),
        (http_error(openai.InternalServerError, 503), True, 503),
        (http_error(openai.BadRequestError, 400), False, 400),
        (http_error(openai.AuthenticationError, 401), False, 401),
        (http_error(anthropic.RateLimitError, 429), True, 429),
        (http_error(anthropic.APIStatusError, 529), True, 529),
        (openai.APITimeoutError(httpx.Request("POST", "https://x.test")), True, None),
        (httpx.ConnectError("refused"), True, None),
        (TimeoutError(), True, None),
        (LLMResponseError("refused"), False, None),
        (ValueError("bad"), False, None),
    ],
)
def test_retry_info_classifies_errors(error, retryable, status):
    got_retryable, got_status, _ = retry_info(error)
    assert (got_retryable, got_status) == (retryable, status)


def test_retry_info_with_google_genai_errors():
    from google.genai import errors

    assert retry_info(errors.ClientError(429, {"error": {"message": "slow"}}))[:2] == (
        True,
        429,
    )
    assert retry_info(errors.ServerError(500, {"error": {"message": "x"}}))[0]
    assert not retry_info(errors.ClientError(400, {"error": {"message": "x"}}))[0]


def test_retry_after_header():
    assert retry_after_seconds(
        http_error(openai.RateLimitError, 429, {"retry-after": "3"})
    ) == pytest.approx(3)
    assert retry_after_seconds(
        http_error(openai.RateLimitError, 429, {"retry-after-ms": "250"})
    ) == pytest.approx(0.25)
    assert retry_after_seconds(http_error(openai.RateLimitError, 429)) is None
    assert retry_after_seconds(ValueError()) is None
    # Very long waits are capped.
    assert (
        retry_after_seconds(
            http_error(openai.RateLimitError, 429, {"retry-after": "100000"})
        )
        == 600
    )


# --- configuration --------------------------------------------------------------------


def test_no_limits_by_default_but_retries_on():
    limits = get_rate_limits("openai/gpt-5")
    assert limits.max_concurrency is None
    assert limits.requests_per_minute is None
    assert limits.tokens_per_minute is None
    assert limits.max_retries > 0


def test_limits_resolve_by_model_then_provider_then_default():
    assert get_rate_limits("openai/gpt-5") == RateLimits()
    set_rate_limits("*", max_concurrency=4)
    set_rate_limits("openai", requests_per_minute=100)
    set_rate_limits("openai/gpt-5", max_concurrency=32)
    assert get_rate_limits("gemini/x").max_concurrency == 4
    assert get_rate_limits("openai/gpt-5-mini").requests_per_minute == 100
    assert get_rate_limits("openai/gpt-5").max_concurrency == 32


def test_limiter_is_shared_and_follows_new_limits():
    limiter = get_rate_limiter("openai/gpt-5")
    assert get_rate_limiter("openai/gpt-5") is limiter
    assert get_rate_limiter("openai/other") is not limiter
    set_rate_limits("openai", max_concurrency=2)
    assert limiter.limits.max_concurrency == 2


def test_set_rate_limits_validates():
    with pytest.raises(ValueError):
        set_rate_limits("*", max_concurrency=0)
    with pytest.raises(ValueError):
        set_rate_limits("*", unknown=1)


# --- windows --------------------------------------------------------------------------


def test_requests_per_minute_window():
    limiter = RateLimiter("m", RateLimits(requests_per_minute=2))
    limiter._requests.extend([100.0, 110.0])
    # The oldest call leaves the window at 160.
    assert limiter._wait_seconds(120.0, 0) == pytest.approx(40)
    assert limiter._wait_seconds(161.0, 0) <= 0


def test_tokens_per_minute_window():
    limiter = RateLimiter("m", RateLimits(tokens_per_minute=1000))
    limiter._tokens.extend([[100.0, 600.0], [110.0, 300.0]])
    assert limiter._wait_seconds(120.0, 100) <= 0
    # 1,100 tokens: the first reservation must leave the window.
    assert limiter._wait_seconds(120.0, 200) == pytest.approx(40)
    # 1,700 tokens: both must leave.
    assert limiter._wait_seconds(120.0, 800) == pytest.approx(50)
    # A call larger than the whole budget still runs once the window is empty.
    limiter._tokens.clear()
    assert limiter._wait_seconds(120.0, 5000) <= 0


def test_pause_holds_new_calls():
    limiter = RateLimiter("m", RateLimits())
    limiter.pause(5)
    assert limiter._wait_seconds(time.monotonic(), 0) > 4


def test_token_reservation_is_corrected_with_reported_usage():
    from aatm.llm.usage import record_usage

    limiter = get_rate_limiter("fake/model")
    set_rate_limits("fake", tokens_per_minute=10_000)

    async def call():
        record_usage(40, 60)
        return "ok"

    assert asyncio.run(limiter.run(call, estimated_tokens=5)) == "ok"
    assert [t for _, t in limiter._tokens] == [100]


# --- running calls -------------------------------------------------------------------


def run_with(limiter: RateLimiter, outcomes: list):
    """Run a call that raises or returns each outcome in turn; return (result, calls)."""
    calls = []

    async def call():
        calls.append(1)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return asyncio.run(limiter.run(call)), len(calls)


def test_retries_transient_errors_then_succeeds():
    limiter = get_rate_limiter("openai/m")
    set_rate_limits("*", **FAST)
    outcomes = [
        http_error(openai.RateLimitError, 429, {"retry-after-ms": "1"}),
        http_error(openai.InternalServerError, 500),
        TimeoutError(),
        "done",
    ]
    assert run_with(limiter, outcomes) == ("done", 4)


def test_rate_limit_pauses_the_model():
    limiter = get_rate_limiter("openai/m")
    set_rate_limits("*", **FAST)
    before = time.monotonic()
    run_with(
        limiter, [http_error(openai.RateLimitError, 429, {"retry-after-ms": "1"}), 1]
    )
    assert limiter._paused_until >= before


def test_non_retryable_errors_are_raised_at_once():
    limiter = get_rate_limiter("openai/m")
    set_rate_limits("*", **FAST)
    calls = []

    async def call():
        calls.append(1)
        raise http_error(openai.BadRequestError, 400)

    with pytest.raises(openai.BadRequestError):
        asyncio.run(limiter.run(call))
    assert len(calls) == 1


def test_retries_run_out():
    limiter = get_rate_limiter("openai/m")
    set_rate_limits("*", max_retries=2, **FAST)
    calls = []

    async def call():
        calls.append(1)
        raise http_error(openai.InternalServerError, 502)

    with pytest.raises(openai.InternalServerError):
        asyncio.run(limiter.run(call))
    assert len(calls) == 3


def test_max_concurrency_is_enforced():
    set_rate_limits("*", max_concurrency=3)
    limiter = get_rate_limiter("openai/m")
    active = peak = 0

    async def call():
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1

    async def main():
        await asyncio.gather(*(limiter.run(call) for _ in range(12)))

    asyncio.run(main())
    assert peak == 3


def test_limiter_works_across_event_loops():
    limiter = get_rate_limiter("openai/m")

    async def call():
        return 1

    assert asyncio.run(limiter.run(call)) == 1
    assert asyncio.run(limiter.run(call)) == 1


# --- adapters --------------------------------------------------------------------------


def test_sdk_retries_are_disabled():
    assert OpenAILLM("gpt-5", api_key="k").client.max_retries == 0
    compatible = OpenAICompatibleLLM("m", base_url="http://localhost:1/v1")
    assert compatible.client.max_retries == 0
    assert AnthropicLLM("claude-opus-5-5", api_key="k").client.max_retries == 0


def test_openai_adapter_retries_through_the_limiter():
    set_rate_limits("*", **FAST)
    llm = OpenAILLM("gpt-5", api_key="k")
    calls = []

    async def parse(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise http_error(openai.RateLimitError, 429)
        return SimpleNamespace(
            output_parsed=Answer(value="x"),
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            status="completed",
            incomplete_details=None,
        )

    llm.client = SimpleNamespace(responses=SimpleNamespace(parse=parse))  # type: ignore[assignment]
    result = asyncio.run(llm.generate(system="s", user="u", output_type=Answer))
    assert result == Answer(value="x")
    assert len(calls) == 2


def test_empty_output_is_not_retried():
    set_rate_limits("*", **FAST)
    llm = OpenAILLM("gpt-5", api_key="k")
    calls = []

    async def parse(**kwargs):
        calls.append(1)
        return SimpleNamespace(
            output_parsed=None, usage=None, status="incomplete", incomplete_details=None
        )

    llm.client = SimpleNamespace(responses=SimpleNamespace(parse=parse))  # type: ignore[assignment]
    with pytest.raises(LLMResponseError):
        asyncio.run(llm.generate(system="s", user="u", output_type=Answer))
    assert len(calls) == 1


# --- extractor and mapper -----------------------------------------------------------


class FailingLLM:
    model_id = "fake/failing"

    async def generate(self, *, system, user, output_type):
        raise LLMResponseError("no output")


def definitions() -> EntityDefinitions:
    return EntityDefinitions(
        entities=[EntityDefinition(name="Condition name", definition="A condition.")]
    )


def test_extractor_records_errors_per_text():
    extractor = SimpleLLMExtractor(
        entity_definitions=definitions(), llm=FailingLLM(), on_error="record"
    )
    results = asyncio.run(extractor(["Note one.", "Note two."]))
    assert [r.error for r in results] == ["LLMResponseError: no output"] * 2
    assert all(not r.validated_entities for r in results)


def test_extractor_raises_errors_by_default():
    extractor = SimpleLLMExtractor(entity_definitions=definitions(), llm=FailingLLM())
    assert extractor.semaphore is None
    with pytest.raises(LLMResponseError):
        asyncio.run(extractor(["Note."]))


def test_extractor_validates_arguments():
    with pytest.raises(ValueError):
        SimpleLLMExtractor(
            entity_definitions=definitions(), llm=FailingLLM(), on_error="skip"
        )
    with pytest.raises(ValueError):
        SimpleLLMExtractor(
            entity_definitions=definitions(), llm=FailingLLM(), max_concurrency=0
        )


def test_async_mapper_rate_limit_does_not_block_the_loop():
    ticks = []

    async def ticker():
        for _ in range(5):
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    finished = []

    async def wait():
        await arate_limit(1, time.monotonic() + 0.1, 60)
        finished.append(time.monotonic())

    async def main():
        await asyncio.gather(wait(), ticker())

    asyncio.run(main())
    # The ticker kept running while the rate limit waited.
    assert sum(t < finished[0] for t in ticks) == 5
