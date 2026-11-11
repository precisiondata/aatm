"""Tests for the decision models in `aatm.llm.decisions`.

Offline: the HTTP backends run against ``httpx.MockTransport``. Live calls run only
when ``AATM_LIVE_LLM_TESTS=1`` is set (they cost money) and the provider's key is
present.
"""

import asyncio
import json
import logging
import os
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError

from aatm.llm import (
    Decision,
    DecisionError,
    DecisionModel,
    DecisionSettings,
    LLMDecisionModel,
    Question,
    available_decision_providers,
    create_decision_model,
    register_decision_provider,
    reset_rate_limits,
    set_rate_limits,
    track_usage,
)
from aatm.llm.jev import OPENROUTER_DECISIONS_URL, DecisionHTTPError, JevDecisionModel
from aatm.llm.openai_decisions import OpenAIDecisionModel


@pytest.fixture(autouse=True)
def _limits():
    reset_rate_limits()
    yield
    reset_rate_limits()


SPECIMEN = Question.choice(
    "specimen",
    "Which specimen was the test done on?",
    {"serum": "Serum or plasma", "urine": "Urine", "unknown": None},
)
CERTAINTY = Question.score(
    "certainty", "How certain is the mapping?", ["low", "medium", "high"]
)
IS_LAB = Question.predicate("is_lab", "Is the text the name of a lab test?")
QUESTIONS = [SPECIMEN, CERTAINTY, IS_LAB]
STATE = {"text": "creatinine", "unit": "mg/dl"}


class Recorder:
    """A mock transport that records requests and replays responses in order."""

    def __init__(self, *responses: httpx.Response):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def body(self, i: int = -1) -> dict:
        return json.loads(self.requests[i].content)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


# --- Questions ---------------------------------------------------------------------------


def test_question_constructors():
    assert SPECIMEN.values == ["serum", "urine", "unknown"]
    assert SPECIMEN.options[0].description == "Serum or plasma"
    assert CERTAINTY.kind == "score" and CERTAINTY.values == ["low", "medium", "high"]
    assert IS_LAB.options == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(name="1bad", kind="predicate", instructions="x"),
        dict(name="has-dash", kind="predicate", instructions="x"),
        dict(name="q", kind="choice", instructions="x", options=[{"value": "a"}]),
        dict(
            name="q",
            kind="choice",
            instructions="x",
            options=[{"value": "a"}, {"value": "a"}],
        ),
        dict(
            name="q",
            kind="score",
            instructions="x",
            options=[{"value": str(i)} for i in range(11)],
        ),
        dict(name="q", kind="predicate", instructions="x", options=[{"value": "a"}]),
        dict(name="q", kind="predicate", instructions=""),
    ],
)
def test_question_validation(kwargs):
    with pytest.raises(ValidationError):
        Question(**kwargs)


def test_choice_accepts_up_to_255_options():
    Question.choice("q", "x", [str(i) for i in range(255)])
    with pytest.raises(ValidationError):
        Question.choice("q", "x", [str(i) for i in range(256)])


def test_decide_rejects_bad_question_lists():
    model = LLMDecisionModel(FakeLLM({}))
    with pytest.raises(ValueError, match="at least one"):
        asyncio.run(model.decide("x", []))
    with pytest.raises(ValueError, match="unique"):
        asyncio.run(model.decide("x", [IS_LAB, IS_LAB]))


# --- OpenAI Decisions API ----------------------------------------------------------------

OPENAI_RESPONSE = {
    "model": "gpt-6-luna-2026-10-01",
    "answers": [
        {
            "type": "choice",
            "name": "specimen",
            "choice": "serum",
            "confidence": 0.8,
            "probabilities": [
                {"value": "serum", "probability": 0.8},
                {"value": "urine", "probability": 0.15},
                {"value": "unknown", "probability": 0.05},
            ],
        },
        {
            "type": "score",
            "name": "certainty",
            "score": 1.6,
            "confidence": 0.6,
            "probabilities": [
                {"value": 0, "label": "low", "probability": 0.1},
                {"value": 1, "label": "medium", "probability": 0.2},
                {"value": 2, "label": "high", "probability": 0.7},
            ],
        },
        {"type": "predicate", "name": "is_lab", "probability": 0.97},
    ],
    "usage": {"input_tokens": 120, "output_tokens": 0, "total_tokens": 120},
}


def openai_model(recorder: Recorder, **kwargs) -> OpenAIDecisionModel:
    client = AsyncOpenAI(api_key="test", max_retries=0, http_client=recorder.client())
    return OpenAIDecisionModel("gpt-6-luna", client=client, **kwargs)


def test_openai_request_and_answers():
    recorder = Recorder(httpx.Response(200, json=OPENAI_RESPONSE))
    model = openai_model(
        recorder, settings=DecisionSettings(provider_kwargs={"prompt_cache_key": "k"})
    )
    with track_usage() as usage:
        decision = asyncio.run(model.decide(STATE, QUESTIONS))

    request = recorder.requests[0]
    assert request.url == "https://api.openai.com/v1/decisions"
    body = recorder.body()
    assert body["model"] == "gpt-6-luna"
    assert body["prompt_cache_key"] == "k"
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_json", "data": STATE}]}
    ]
    assert body["questions"] == [
        {
            "type": "choice",
            "name": "specimen",
            "instructions": "Which specimen was the test done on?",
            "choices": [
                {"value": "serum", "description": "Serum or plasma"},
                {"value": "urine", "description": "Urine"},
                {"value": "unknown"},
            ],
        },
        {
            "type": "score",
            "name": "certainty",
            "instructions": "How certain is the mapping?",
            "levels": [{"label": "low"}, {"label": "medium"}, {"label": "high"}],
        },
        {
            "type": "predicate",
            "name": "is_lab",
            "instructions": "Is the text the name of a lab test?",
        },
    ]

    assert isinstance(decision, Decision)
    assert decision.model == "gpt-6-luna-2026-10-01"
    specimen = decision["specimen"]
    assert specimen.value == "serum" and specimen.confidence == 0.8
    assert specimen.probabilities["urine"] == pytest.approx(0.15)
    assert specimen.probability == pytest.approx(0.8)
    certainty = decision["certainty"]
    assert certainty.value == pytest.approx(1.6) and certainty.label == "high"
    assert certainty.probability == pytest.approx(0.7)
    is_lab = decision["is_lab"]
    assert is_lab.value is True and is_lab.probability == pytest.approx(0.97)
    assert is_lab.probabilities["false"] == pytest.approx(0.03)

    assert usage.by_model["openai/gpt-6-luna"].input_tokens == 120


def test_openai_text_state_and_refusal():
    response = {
        "model": "gpt-6-luna",
        "answers": [{"type": "refusal", "name": "is_lab"}],
        "usage": {},
    }
    recorder = Recorder(httpx.Response(200, json=response))
    decision = asyncio.run(openai_model(recorder).decide("Hb 12 g/dl", [IS_LAB]))
    assert recorder.body()["input"] == "Hb 12 g/dl"
    answer = decision["is_lab"]
    assert answer.refused and answer.value is None and answer.probability is None


@pytest.mark.parametrize(
    "answers, match",
    [
        ([], "0 answers for 1"),
        ([{"type": "choice", "choice": "x"}], "with"),
        ([{"type": "predicate"}], "malformed"),
    ],
)
def test_openai_bad_answers(answers, match):
    recorder = Recorder(httpx.Response(200, json={"model": "m", "answers": answers}))
    with pytest.raises(DecisionError, match=match):
        asyncio.run(openai_model(recorder).decide("x", [IS_LAB]))


def test_openai_retries_rate_limits():
    set_rate_limits("openai/gpt-6-luna", initial_backoff=0.01)
    ok = {"model": "m", "answers": [{"type": "predicate", "probability": 0.2}]}
    recorder = Recorder(
        httpx.Response(429, json={"error": {"message": "slow down"}}),
        httpx.Response(200, json=ok),
    )
    decision = asyncio.run(openai_model(recorder).decide("x", [IS_LAB]))
    assert len(recorder.requests) == 2
    assert decision["is_lab"].value is False


# --- Jev-shaped endpoints ----------------------------------------------------------------

JEV_RESPONSE = {
    "id": "dec-1",
    "model": "typesafe/jev-1.13.0",
    "provider": "TypeSafe",
    "answers": {
        "specimen": {
            "type": "choice",
            "choice": "serum",
            "confidence": 0.7,
            "probabilities": {"serum": 0.7, "urine": 0.2, "unknown": 0.1},
        },
        "certainty": {
            "type": "score",
            "score": 0.4,
            "confidence": 0.5,
            "legend": {"0": "low", "1": "medium", "2": "high"},
            "probabilities": {"0": 0.65, "1": 0.3, "2": 0.05},
        },
        "is_lab": {"type": "noul", "noul": 0.9},
    },
    "usage": {"input_tokens": 80, "output_tokens": 3, "cost": 0.0000034},
}


def jev_model(recorder: Recorder, **kwargs) -> JevDecisionModel:
    return JevDecisionModel(
        "typesafe/jev-1.13",
        url=OPENROUTER_DECISIONS_URL,
        api_key="key",
        provider="openrouter",
        http_client=recorder.client(),
        **kwargs,
    )


def test_jev_request_and_answers():
    recorder = Recorder(httpx.Response(200, json=JEV_RESPONSE))
    with track_usage() as usage:
        decision = asyncio.run(jev_model(recorder).decide(STATE, QUESTIONS))

    request = recorder.requests[0]
    assert str(request.url) == OPENROUTER_DECISIONS_URL
    assert request.headers["authorization"] == "Bearer key"
    assert recorder.body() == {
        "model": "typesafe/jev-1.13",
        "state": STATE,
        "questions": {
            "specimen": {
                "type": "choice",
                "instructions": "Which specimen was the test done on?",
                "criteria": {
                    "serum": "Serum or plasma",
                    "urine": "Urine",
                    "unknown": "unknown",
                },
            },
            "certainty": {
                "type": "score",
                "instructions": "How certain is the mapping?",
                "criteria": ["low", "medium", "high"],
            },
            "is_lab": {
                "type": "noul",
                "instructions": "Is the text the name of a lab test?",
            },
        },
    }

    assert decision.model == "typesafe/jev-1.13.0"
    assert decision["specimen"].value == "serum"
    assert decision["specimen"].probabilities == {
        "serum": 0.7,
        "urine": 0.2,
        "unknown": 0.1,
    }
    certainty = decision["certainty"]
    assert certainty.label == "low" and certainty.value == pytest.approx(0.4)
    assert certainty.probabilities == {"low": 0.65, "medium": 0.3, "high": 0.05}
    assert decision["is_lab"].value is True
    assert decision["is_lab"].probability == pytest.approx(0.9)

    model_usage = usage.by_model["openrouter/typesafe/jev-1.13"]
    assert model_usage.input_tokens == 80
    assert model_usage.reported_cost == pytest.approx(0.0000034)
    assert usage.cost(None) == pytest.approx(0.0000034)


def test_jev_score_probabilities_keyed_by_criterion():
    question = Question.score(
        "severity", "How severe?", {"mild": "Mild", "severe": "Severe"}
    )
    response = {
        "model": "m",
        "answers": {
            "severity": {
                "type": "score",
                "score": 0.9,
                "probabilities": {"mild: Mild": 0.1, "severe: Severe": 0.9},
            }
        },
    }
    recorder = Recorder(httpx.Response(200, json=response))
    decision = asyncio.run(jev_model(recorder).decide("x", [question]))
    assert recorder.body()["questions"]["severity"]["criteria"] == [
        "mild: Mild",
        "severe: Severe",
    ]
    assert decision["severity"].probabilities == {"mild": 0.1, "severe": 0.9}
    assert decision["severity"].label == "severe"


@pytest.mark.parametrize(
    "answers, match",
    [
        ({}, "no answer for 'is_lab'"),
        ({"is_lab": {"type": "choice", "choice": "x"}}, "with"),
        ({"is_lab": {"type": "noul"}}, "malformed"),
    ],
)
def test_jev_bad_answers(answers, match):
    recorder = Recorder(httpx.Response(200, json={"model": "m", "answers": answers}))
    with pytest.raises(DecisionError, match=match):
        asyncio.run(jev_model(recorder).decide("x", [IS_LAB]))


def test_jev_client_errors_are_not_retried():
    recorder = Recorder(httpx.Response(400, json={"error": "bad request"}))
    with pytest.raises(DecisionHTTPError) as error:
        asyncio.run(jev_model(recorder).decide("x", [IS_LAB]))
    assert error.value.status_code == 400
    assert len(recorder.requests) == 1


def test_jev_server_errors_are_retried():
    set_rate_limits("openrouter", initial_backoff=0.01)
    ok = {"model": "m", "answers": {"is_lab": {"type": "noul", "noul": 0.1}}}
    recorder = Recorder(httpx.Response(503), httpx.Response(200, json=ok))
    decision = asyncio.run(jev_model(recorder).decide("x", [IS_LAB]))
    assert len(recorder.requests) == 2
    assert decision["is_lab"].value is False


# --- Any StructuredLLM -------------------------------------------------------------------


class FakeLLM:
    """Answers with fixed values, keyed by question name, and records prompts."""

    model_id = "fake/model"

    def __init__(self, answers: dict):
        self.answers = answers
        self.calls: list[dict] = []

    async def generate(self, *, system, user, output_type):
        self.calls.append({"system": system, "user": user, "schema": output_type})
        return output_type.model_validate(self.answers)


def test_llm_decision_model():
    llm = FakeLLM({"specimen": "urine", "certainty": "medium", "is_lab": True})
    model = LLMDecisionModel(llm)
    assert isinstance(model, DecisionModel)
    assert model.model_id == "llm/fake/model"

    decision = asyncio.run(model.decide(STATE, QUESTIONS))

    user = llm.calls[0]["user"]
    assert '"text": "creatinine"' in user
    assert "- serum: Serum or plasma" in user
    assert "Levels, from lowest to highest:" in user
    schema = llm.calls[0]["schema"].model_json_schema()
    assert set(schema["properties"]) == {"specimen", "certainty", "is_lab"}
    assert schema["properties"]["specimen"]["enum"] == ["serum", "urine", "unknown"]

    assert decision.model == "fake/model"
    assert decision["specimen"].value == "urine"
    assert decision["specimen"].probabilities == {}
    assert decision["specimen"].probability is None
    assert decision["certainty"].value == 1.0
    assert decision["certainty"].label == "medium"
    assert decision["is_lab"].value is True and decision["is_lab"].confidence is None


def test_llm_decision_model_rejects_values_outside_the_options():
    llm = FakeLLM({"specimen": "blood"})
    with pytest.raises(ValidationError):
        asyncio.run(LLMDecisionModel(llm).decide("x", [SPECIMEN]))


# --- Registry ----------------------------------------------------------------------------


def test_create_decision_model_dispatches_on_prefix(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-key")

    openai = create_decision_model("openai/gpt-6-luna", settings={"timeout": 5})
    assert isinstance(openai, OpenAIDecisionModel)
    assert openai.settings.timeout == 5

    openrouter = create_decision_model("openrouter/typesafe/jev-1.13")
    assert isinstance(openrouter, JevDecisionModel)
    assert openrouter.model_id == "openrouter/typesafe/jev-1.13"
    assert openrouter.url == OPENROUTER_DECISIONS_URL
    assert openrouter.api_key == "or-key"

    eden = create_decision_model(
        "jev-compatible/typesafe/jev-1.13",
        base_url="https://api.edenai.run/v3/alpha/decisions",
        api_key="eden",
    )
    assert isinstance(eden, JevDecisionModel) and eden.api_key == "eden"
    assert eden.url == "https://api.edenai.run/v3/alpha/decisions"

    llm = create_decision_model("llm/openai/gpt-5-mini")
    assert isinstance(llm, LLMDecisionModel)
    assert llm.model_id == "llm/openai/gpt-5-mini"


@pytest.mark.parametrize(
    "model_id, kwargs, match",
    [
        ("gpt-6-luna", {}, "prefixed"),
        ("openai/", {}, "no model name"),
        ("nope/x", {}, "Unknown decision provider"),
        ("jev-compatible/jev-1.13", {}, "base_url"),
        ("openai/gpt-6-luna", {"settings": 3}, "settings"),
    ],
)
def test_create_decision_model_errors(model_id, kwargs, match):
    with pytest.raises(ValueError, match=match):
        create_decision_model(model_id, api_key="k", **kwargs)


def test_register_decision_provider():
    register_decision_provider("fake", lambda model, **_: LLMDecisionModel(FakeLLM({})))
    assert "fake" in available_decision_providers()
    assert isinstance(create_decision_model("fake/x"), LLMDecisionModel)
    with pytest.raises(ValueError):
        register_decision_provider("a/b", lambda model, **_: None)


def test_moving_aliases_warn(caplog):
    # `aatm.logs.setup_logging`, called by other tests, stops the "aatm" logger
    # from propagating to the root logger caplog listens on.
    logger = logging.getLogger("aatm.llm.decisions")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="aatm.llm.decisions"):
            create_decision_model("openrouter/typesafe/jev-latest", api_key="k")
        assert "moving alias" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="aatm.llm.decisions"):
            create_decision_model("openrouter/typesafe/jev-1.13", api_key="k")
        assert caplog.text == ""
    finally:
        logger.removeHandler(caplog.handler)


def test_alpha_endpoints_warn_once(caplog, monkeypatch):
    from aatm.llm import jev

    monkeypatch.setattr(jev, "_WARNED_ALPHA", set())
    logger = logging.getLogger("aatm.llm.jev")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="aatm.llm.jev"):
            create_decision_model("openrouter/typesafe/jev-1.13", api_key="k")
            create_decision_model("openrouter/typesafe/jev-1.14", api_key="k")
            create_decision_model(
                "jev-compatible/jev-1.13",
                base_url="https://decisions.example.org/v1/decide",
            )
    finally:
        logger.removeHandler(caplog.handler)
    # caplog may see a record twice: through the handler and through propagation.
    alpha = {
        (r.created, r.getMessage())
        for r in caplog.records
        if "alpha endpoint" in r.getMessage()
    }
    assert len(alpha) == 1
    assert OPENROUTER_DECISIONS_URL in next(iter(alpha))[1]


class FakeDecisionsResource:
    """`client.decisions` of openai >= 3: records kwargs, returns a model."""

    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(model_dump=lambda: self.response)


def test_openai_uses_the_sdk_decisions_method_when_available():
    resource = FakeDecisionsResource(OPENAI_RESPONSE)
    client = SimpleNamespace(decisions=resource)
    model = OpenAIDecisionModel(
        "gpt-6-luna",
        client=client,  # type: ignore[arg-type]
        settings=DecisionSettings(
            timeout=7, provider_kwargs={"prompt_cache_key": "k", "store": False}
        ),
    )
    decision = asyncio.run(model.decide("creatinine", QUESTIONS))

    call = resource.calls[0]
    assert call["model"] == "gpt-6-luna" and call["input"] == "creatinine"
    assert [q["name"] for q in call["questions"]] == ["specimen", "certainty", "is_lab"]
    assert call["prompt_cache_key"] == "k"
    assert call["extra_body"] == {"store": False}
    assert call["timeout"] == 7
    assert decision["specimen"].value == "serum"
    assert decision["is_lab"].probability == pytest.approx(0.97)


# --- Live --------------------------------------------------------------------------------

LIVE = os.getenv("AATM_LIVE_LLM_TESTS") == "1"


@pytest.mark.skipif(not LIVE, reason="set AATM_LIVE_LLM_TESTS=1 to call providers")
@pytest.mark.parametrize(
    "model_id, key_env",
    [
        ("openai/gpt-6-luna", "OPENAI_API_KEY"),
        ("openrouter/typesafe/jev-1.13", "OPENROUTER_API_KEY"),
    ],
)
def test_live_decision(model_id, key_env):
    if not os.getenv(key_env):
        pytest.skip(f"{key_env} is not set")
    model = create_decision_model(model_id)
    # A name with a result is not "the name of a lab test": ask about the result.
    reports_result = Question.predicate(
        "reports_result", "Does the text report a laboratory test result?"
    )
    decision = asyncio.run(
        model.decide(
            {"text": "serum creatinine 1.4 mg/dl"},
            [SPECIMEN, CERTAINTY, reports_result],
        )
    )
    assert decision["reports_result"].value is True
    assert decision["reports_result"].probability > 0.5
    assert decision["specimen"].value == "serum"
    assert decision["specimen"].probabilities
    assert decision["certainty"].label in CERTAINTY.values
