"""Tests for the provider batch endpoints (`aatm.llm.batch`), with fake clients."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from google.genai import types
from pydantic import BaseModel

from aatm.llm import LLMSettings, track_usage
from aatm.llm.anthropic import AnthropicLLM
from aatm.llm.batch import BatchRequest, supports_batch
from aatm.llm.gemini import GEMINI_INLINE_LIMIT, GeminiLLM
from aatm.llm.openai import OpenAICompatibleLLM, OpenAILLM
from aatm.llm.usage import TokenUsage, batch_model_id

KEY = "test-key"


class Out(BaseModel):
    value: int


def requests(n=2):
    return [
        BatchRequest(custom_id=f"note-{i}", system="sys", user=f"user {i}")
        for i in range(1, n + 1)
    ]


def ids(n=2):
    return [f"note-{i}" for i in range(1, n + 1)]


def run(coro):
    return asyncio.run(coro)


def test_supports_batch():
    assert supports_batch(OpenAILLM("gpt-5-mini", api_key=KEY))
    assert supports_batch(AnthropicLLM("claude-haiku-5-5", api_key=KEY))
    assert supports_batch(GeminiLLM("gemini-3-flash-preview", api_key=KEY))
    assert not supports_batch(OpenAICompatibleLLM("llama", base_url="http://x/v1"))


def test_requests_are_checked():
    llm = OpenAILLM("gpt-5-mini", api_key=KEY)
    with pytest.raises(ValueError, match="at least one"):
        run(llm.submit_batch([], Out))
    bad = [BatchRequest(custom_id="note 1", system="s", user="u")]
    with pytest.raises(ValueError, match="Invalid custom_id"):
        run(llm.submit_batch(bad, Out))
    with pytest.raises(ValueError, match="Duplicate"):
        run(llm.submit_batch(requests(1) * 2, Out))


# --- usage and cost -------------------------------------------------------------------


def test_batch_usage_is_priced_at_a_discount():
    usage = TokenUsage()
    usage.add(1_000_000, 1_000_000, "openai/gpt-5-mini")
    usage.add(1_000_000, 1_000_000, batch_model_id("openai/gpt-5-mini"))
    assert batch_model_id("openai/gpt-5-mini") == "openai/gpt-5-mini@batch"
    assert batch_model_id("openai/gpt-5-mini@batch") == "openai/gpt-5-mini@batch"
    prices = {"openai/gpt-5-mini": (0.25, 2.0)}
    assert usage.cost(prices) == pytest.approx(2.25 + 1.125)
    assert usage.unpriced(prices) == []
    # A price for the batch id wins over the discount.
    prices["openai/gpt-5-mini@batch"] = (0.1, 0.1)
    assert usage.cost(prices) == pytest.approx(2.25 + 0.2)
    assert usage.unpriced({"other/model": (1, 1)}) == [
        "openai/gpt-5-mini",
        "openai/gpt-5-mini@batch",
    ]


# --- OpenAI ---------------------------------------------------------------------------


def openai_line(custom_id, value=None, status=200, error=None, refusal=None):
    content = []
    if refusal is not None:
        content.append({"type": "refusal", "refusal": refusal})
    if value is not None:
        content.append({"type": "output_text", "text": json.dumps({"value": value})})
    body = {
        "status": "completed",
        "output": [{"type": "message", "content": content}],
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    if status != 200:
        body = {"error": {"message": error}}
    return json.dumps(
        {"custom_id": custom_id, "response": {"status_code": status, "body": body}}
    )


class FakeOpenAI:
    def __init__(self, status="completed", output=(), errors=(), batch_errors=None):
        self.status = status
        self.files_content = {"out": "\n".join(output), "err": "\n".join(errors)}
        self.uploaded = None
        self.created = None
        self.batch_errors = batch_errors
        self.files = SimpleNamespace(create=self._upload, content=self._content)
        self.batches = SimpleNamespace(create=self._create, retrieve=self._retrieve)

    async def _upload(self, file, purpose):
        self.uploaded = (file, purpose)
        return SimpleNamespace(id="file-1")

    async def _create(self, **kwargs):
        self.created = kwargs
        return SimpleNamespace(id="batch_1")

    async def _retrieve(self, batch_id):
        return SimpleNamespace(
            status=self.status,
            output_file_id="out" if self.files_content["out"] else None,
            error_file_id="err" if self.files_content["err"] else None,
            errors=self.batch_errors,
        )

    async def _content(self, file_id):
        return SimpleNamespace(text=self.files_content[file_id])


def test_openai_submit_writes_responses_requests():
    llm = OpenAILLM(
        "gpt-5-mini",
        api_key=KEY,
        settings=LLMSettings(reasoning_effort="low", max_tokens=500, timeout=30),
    )
    llm.client = FakeOpenAI()
    assert run(llm.submit_batch(requests(), Out)) == "batch_1"
    (name, data), purpose = llm.client.uploaded
    assert purpose == "batch"
    lines = [json.loads(line) for line in data.decode().splitlines()]
    assert [line["custom_id"] for line in lines] == ids()
    assert lines[0]["url"] == "/v1/responses"
    body = lines[0]["body"]
    assert body["model"] == "gpt-5-mini"
    assert body["instructions"] == "sys" and body["input"] == "user 1"
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert body["reasoning"] == {"effort": "low"}
    assert body["max_output_tokens"] == 500
    assert "timeout" not in body
    assert llm.client.created == {
        "input_file_id": "file-1",
        "endpoint": "/v1/responses",
        "completion_window": "24h",
    }


def test_openai_poll():
    llm = OpenAILLM("gpt-5-mini", api_key=KEY)
    llm.client = FakeOpenAI(status="in_progress")
    assert run(llm.poll_batch("batch_1", ids(), Out)) is None

    llm.client = FakeOpenAI(
        output=[openai_line("note-1", value=3), openai_line("note-3", refusal="no")],
        errors=[openai_line("note-2", status=400, error="bad schema")],
    )
    with track_usage() as usage:
        results = run(llm.poll_batch("batch_1", ids(4), Out))
    assert [r.custom_id for r in results] == ids(4)
    assert results[0].output == Out(value=3) and results[0].error is None
    assert "bad schema" in results[1].error
    assert "refused" in results[2].error
    assert "without a result" in results[3].error
    assert usage.by_model["openai/gpt-5-mini@batch"].input_tokens == 20


def test_openai_failed_batch_reports_its_errors():
    llm = OpenAILLM("gpt-5-mini", api_key=KEY)
    llm.client = FakeOpenAI(
        status="failed",
        batch_errors=SimpleNamespace(data=[SimpleNamespace(message="file too big")]),
    )
    (result,) = run(llm.poll_batch("batch_1", ids(1), Out))
    assert "failed" in result.error and "file too big" in result.error


# --- Anthropic ------------------------------------------------------------------------


class AsyncEntries:
    def __init__(self, entries):
        self.entries = list(entries)

    def __aiter__(self):
        self._it = iter(self.entries)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


def anthropic_entry(custom_id, value=None, kind="succeeded", stop="end_turn"):
    if kind != "succeeded":
        error = SimpleNamespace(error=SimpleNamespace(message="overloaded"))
        return SimpleNamespace(
            custom_id=custom_id, result=SimpleNamespace(type=kind, error=error)
        )
    text = json.dumps({"value": value}) if value is not None else ""
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop,
        stop_details=SimpleNamespace(category="cyber"),
        usage=SimpleNamespace(input_tokens=7, output_tokens=1),
    )
    return SimpleNamespace(
        custom_id=custom_id, result=SimpleNamespace(type=kind, message=message)
    )


class FakeAnthropic:
    def __init__(self, status="ended", entries=()):
        self.status = status
        self.entries = entries
        self.created = None
        batches = SimpleNamespace(
            create=self._create, retrieve=self._retrieve, results=self._results
        )
        self.messages = SimpleNamespace(batches=batches)

    async def _create(self, requests):
        self.created = requests
        return SimpleNamespace(id="msgbatch_1")

    async def _retrieve(self, batch_id):
        return SimpleNamespace(processing_status=self.status)

    async def _results(self, batch_id):
        return AsyncEntries(self.entries)


def test_anthropic_submit():
    llm = AnthropicLLM(
        "claude-haiku-5-5", api_key=KEY, settings=LLMSettings(reasoning_effort="low")
    )
    llm.client = FakeAnthropic()
    assert run(llm.submit_batch(requests(), Out)) == "msgbatch_1"
    first = llm.client.created[0]
    assert first["custom_id"] == "note-1"
    params = first["params"]
    assert params["model"] == "claude-haiku-5-5"
    assert params["system"] == "sys"
    assert params["messages"] == [{"role": "user", "content": "user 1"}]
    assert params["max_tokens"] == 16000
    assert params["output_config"]["effort"] == "low"
    output_format = params["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["additionalProperties"] is False


def test_anthropic_poll():
    llm = AnthropicLLM("claude-haiku-5-5", api_key=KEY)
    llm.client = FakeAnthropic(status="in_progress")
    assert run(llm.poll_batch("msgbatch_1", ids(), Out)) is None

    llm.client = FakeAnthropic(
        entries=[
            anthropic_entry("note-1", value=5),
            anthropic_entry("note-2", kind="errored"),
            anthropic_entry("note-3", stop="refusal"),
            anthropic_entry("note-4", kind="expired"),
        ]
    )
    with track_usage() as usage:
        results = run(llm.poll_batch("msgbatch_1", ids(5), Out))
    assert results[0].output == Out(value=5)
    assert "errored" in results[1].error and "overloaded" in results[1].error
    assert "declined" in results[2].error
    assert "expired" in results[3].error
    assert "without a result" in results[4].error
    assert usage.by_model["anthropic/claude-haiku-5-5@batch"].calls == 2


# --- Gemini ---------------------------------------------------------------------------


def gemini_response(value):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=json.dumps({"value": value}))],
                )
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=9, candidates_token_count=1, thoughts_token_count=4
        ),
    )


class FakeGemini:
    def __init__(self, state=types.JobState.JOB_STATE_SUCCEEDED, responses=()):
        self.state = state
        self.responses = list(responses)
        self.created = None
        self.aio = SimpleNamespace(
            batches=SimpleNamespace(create=self._create, get=self._get)
        )

    async def _create(self, model, src, config):
        self.created = (model, src, config)
        return types.BatchJob(name="batches/123")

    async def _get(self, name):
        return types.BatchJob(
            name=name,
            state=self.state,
            dest=types.BatchJobDestination(inlined_responses=self.responses),
        )


def test_gemini_submit():
    llm = GeminiLLM("gemini-3-flash-preview", api_key=KEY)
    llm.client = FakeGemini()
    assert run(llm.submit_batch(requests(), Out)) == "batches/123"
    model, src, config = llm.client.created
    assert model == "gemini-3-flash-preview"
    assert [r.metadata for r in src] == [{"key": "note-1"}, {"key": "note-2"}]
    assert src[0].contents == "user 1"
    assert src[0].config.system_instruction == "sys"
    assert src[0].config.response_json_schema["required"] == ["value"]


def test_gemini_submit_refuses_oversized_batches():
    llm = GeminiLLM("gemini-3-flash-preview", api_key=KEY)
    llm.client = FakeGemini()
    big = [BatchRequest(custom_id="a", system="s", user="x" * GEMINI_INLINE_LIMIT)]
    with pytest.raises(ValueError, match="smaller chunk_size"):
        run(llm.submit_batch(big, Out))


def test_gemini_poll():
    llm = GeminiLLM("gemini-3-flash-preview", api_key=KEY)
    llm.client = FakeGemini(state=types.JobState.JOB_STATE_RUNNING)
    assert run(llm.poll_batch("batches/123", ids(), Out)) is None

    # Matched by the key sent as metadata, whatever the order.
    llm.client = FakeGemini(
        state=types.JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
        responses=[
            types.InlinedResponse(
                metadata={"key": "note-2"}, error=types.JobError(message="quota")
            ),
            types.InlinedResponse(
                metadata={"key": "note-1"}, response=gemini_response(4)
            ),
        ],
    )
    with track_usage() as usage:
        results = run(llm.poll_batch("batches/123", ids(3), Out))
    assert results[0].output == Out(value=4)
    assert "quota" in results[1].error
    assert "without a result" in results[2].error
    assert usage.by_model["gemini/gemini-3-flash-preview@batch"].output_tokens == 5

    # Without metadata, by position.
    llm.client = FakeGemini(
        responses=[
            types.InlinedResponse(response=gemini_response(1)),
            types.InlinedResponse(response=gemini_response(2)),
        ]
    )
    results = run(llm.poll_batch("batches/123", ids(2), Out))
    assert [r.output for r in results] == [Out(value=1), Out(value=2)]

    llm.client = FakeGemini(state=types.JobState.JOB_STATE_EXPIRED)
    (result,) = run(llm.poll_batch("batches/123", ids(1), Out))
    assert "JOB_STATE_EXPIRED" in result.error
