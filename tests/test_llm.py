"""Tests for the provider adapters in `aatm.llm` and their use by the extractor.

Offline by default. Live calls to the real providers run only when
``AATM_LIVE_LLM_TESTS=1`` is set (they cost money) and the provider's key is present.
"""

import asyncio
import logging
import os

import pytest
from pydantic import BaseModel, ValidationError

from aatm.data_models.extractors import (
    EntityDefinition,
    EntityDefinitions,
    ExtractedEntityCandidate,
    ExtractedEntityCandidates,
)
from aatm.extractors import SimpleLLMExtractor
from aatm.llm import (
    LLMSettings,
    StructuredLLM,
    available_providers,
    create_llm,
    register_provider,
)
from aatm.llm.anthropic import AnthropicLLM
from aatm.llm.gemini import GeminiLLM, gemini_response_schema
from aatm.llm.openai import OpenAICompatibleLLM, OpenAILLM

DUMMY_KEY = "test-key"


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


class FakeLLM:
    """Returns a fixed answer and records the prompts it receives."""

    model_id = "fake/model"

    def __init__(self, answer: BaseModel):
        self.answer = answer
        self.calls: list[dict] = []

    async def generate(self, *, system, user, output_type):
        self.calls.append({"system": system, "user": user, "output_type": output_type})
        return self.answer


# --- registry -----------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id, cls, model",
    [
        ("openai/gpt-5", OpenAILLM, "gpt-5"),
        ("gemini/gemini-3-flash-preview", GeminiLLM, "gemini-3-flash-preview"),
        ("anthropic/claude-opus-5-5", AnthropicLLM, "claude-opus-5-5"),
        ("openrouter/qwen/qwen3-235b", OpenAICompatibleLLM, "qwen/qwen3-235b"),
    ],
)
def test_create_llm_dispatches_on_prefix(model_id, cls, model):
    llm = create_llm(model_id, api_key=DUMMY_KEY)
    assert isinstance(llm, cls)
    assert isinstance(llm, StructuredLLM)
    assert llm.model == model
    assert llm.model_id == model_id


@pytest.mark.parametrize("model_id", ["gpt-5", "openai/"])
def test_create_llm_rejects_malformed_ids(model_id):
    with pytest.raises(ValueError):
        create_llm(model_id)


def test_create_llm_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider 'mistral'"):
        create_llm("mistral/large")


def test_ollama_uses_local_default_url():
    llm = create_llm("ollama/llama3.3")
    assert str(llm.client.base_url).startswith("http://localhost:11434/v1")


def test_openrouter_reads_its_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    llm = create_llm("openrouter/some/model")
    assert llm.client.api_key == "or-key"


def test_openai_compatible_requires_base_url():
    with pytest.raises(ValueError, match="base_url"):
        create_llm("openai-compatible/my-model")
    llm = create_llm("openai-compatible/my-model", base_url="http://localhost:8000/v1")
    assert str(llm.client.base_url).startswith("http://localhost:8000/v1")


@pytest.mark.parametrize("model_id", ["gemini/x", "anthropic/x"])
def test_base_url_rejected_for_native_providers(model_id):
    with pytest.raises(ValueError, match="base_url"):
        create_llm(model_id, api_key=DUMMY_KEY, base_url="http://localhost")


def test_register_provider():
    fake = FakeLLM(ExtractedEntityCandidates(entities=[]))
    register_provider("fake-test", lambda model, **_: fake)
    try:
        assert create_llm("fake-test/anything") is fake
        assert "fake-test" in available_providers()
    finally:
        from aatm.llm import registry

        registry._PROVIDERS.pop("fake-test")


def test_settings_accept_dict_and_reject_unknown_fields():
    llm = create_llm("openai/gpt-5", api_key=DUMMY_KEY, settings={"temperature": 0})
    assert llm.settings.temperature == 0
    with pytest.raises(ValidationError):
        create_llm("openai/gpt-5", api_key=DUMMY_KEY, settings={"temprature": 0})
    with pytest.raises(ValueError):
        create_llm("openai/gpt-5", api_key=DUMMY_KEY, settings=42)  # type: ignore[arg-type]


# --- settings mapping -----------------------------------------------------------

SETTINGS = LLMSettings(
    temperature=0.2,
    max_tokens=1000,
    reasoning_effort="low",
    timeout=30,
    provider_kwargs={"seed": 1},
)


def test_openai_request_kwargs():
    llm = OpenAILLM("gpt-5", api_key=DUMMY_KEY, settings=SETTINGS)
    assert llm._request_kwargs() == {
        "temperature": 0.2,
        "max_output_tokens": 1000,
        "reasoning": {"effort": "low"},
        "timeout": 30,
        "seed": 1,
    }
    assert OpenAILLM("gpt-5", api_key=DUMMY_KEY)._request_kwargs() == {}


def test_openai_compatible_request_kwargs():
    llm = OpenAICompatibleLLM(
        "m", base_url="http://localhost:8000/v1", settings=SETTINGS
    )
    assert llm._request_kwargs() == {
        "temperature": 0.2,
        "max_completion_tokens": 1000,
        "reasoning_effort": "low",
        "timeout": 30,
        "seed": 1,
    }


def test_anthropic_request_kwargs(caplog):
    # Attach the handler directly: aatm.configure_logging() (used by other tests)
    # stops the "aatm" logger from propagating to the root logger caplog listens on.
    logger = logging.getLogger("aatm.llm.anthropic")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="aatm.llm.anthropic"):
            llm = AnthropicLLM("claude-opus-5-5", api_key=DUMMY_KEY, settings=SETTINGS)
    finally:
        logger.removeHandler(caplog.handler)
    assert "temperature" in caplog.text
    assert llm._request_kwargs() == {
        "max_tokens": 1000,
        "output_config": {"effort": "low"},
        "seed": 1,
    }
    default = AnthropicLLM("claude-opus-5-5", api_key=DUMMY_KEY)._request_kwargs()
    assert default == {"max_tokens": 16000}


def test_gemini_config():
    llm = GeminiLLM(
        "gemini-3-flash-preview",
        api_key=DUMMY_KEY,
        settings=SETTINGS.model_copy(update={"provider_kwargs": {}}),
    )
    config = llm._config("system prompt", ExtractedEntityCandidates)
    assert config.system_instruction == "system prompt"
    assert config.response_mime_type == "application/json"
    assert config.temperature == 0.2
    assert config.max_output_tokens == 1000
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level is not None
    assert config.thinking_config.thinking_level.value == "LOW"


def test_gemini_schema_drops_unsupported_keywords():
    schema = gemini_response_schema(ExtractedEntityCandidates.model_json_schema())
    entity = schema["$defs"]["ExtractedEntityCandidate"]
    assert set(entity["properties"]) == {
        "id",
        "text",
        "key",
        "line",
        "occurrence",
        "parents",
        "attributes",
    }
    flat = str(schema)
    for keyword in ("minLength", "pattern", "'default'"):
        assert keyword not in flat
    assert entity["properties"]["occurrence"]["anyOf"][0]["minimum"] == 1


# --- extractor ------------------------------------------------------------------

NOTE = "Problems:\n* Hypertension since 2019\n"


def test_extractor_uses_injected_llm(definitions):
    answer = ExtractedEntityCandidates(
        entities=[
            ExtractedEntityCandidate(
                id="e1", text="Hypertension", key="condition_name", line="L2"
            ),
            ExtractedEntityCandidate(
                id="e2",
                text="2019",
                key="verbatim_start_date",
                line="L2",
                parents=["e1"],
            ),
        ]
    )
    fake = FakeLLM(answer)
    extractor = SimpleLLMExtractor(entity_definitions=definitions, llm=fake)

    results = asyncio.run(extractor([NOTE, NOTE]))

    assert len(results) == 2
    assert [e.text for e in results[0].validated_entities] == ["Hypertension", "2019"]
    assert results[0].unmatched_entities == []
    call = fake.calls[0]
    assert call["output_type"] is ExtractedEntityCandidates
    assert "[L2] * Hypertension since 2019" in call["user"]
    assert "condition_name" in call["system"]
    assert "{entity_definitions}" not in call["system"]


def test_extractor_builds_llm_from_model_id(definitions):
    extractor = SimpleLLMExtractor(
        model_id="anthropic/claude-opus-5-5",
        entity_definitions=definitions,
        api_key=DUMMY_KEY,
        settings={"max_tokens": 4000},
    )
    assert isinstance(extractor.llm, AnthropicLLM)
    assert extractor.model_id == "anthropic/claude-opus-5-5"
    assert extractor.llm.settings.max_tokens == 4000


def test_extractor_argument_errors(definitions):
    with pytest.raises(ValueError, match="model_id or llm"):
        SimpleLLMExtractor(entity_definitions=definitions)
    with pytest.raises(ValueError, match="entity_definitions"):
        SimpleLLMExtractor(model_id="openai/gpt-5")
    with pytest.raises(ValueError, match="list of strings"):
        extractor = SimpleLLMExtractor(
            entity_definitions=definitions,
            llm=FakeLLM(ExtractedEntityCandidates(entities=[])),
        )
        asyncio.run(extractor("not a list"))  # type: ignore[arg-type]


# --- live (opt-in) --------------------------------------------------------------

LIVE_MODELS = [
    ("openai/gpt-5-mini", "OPENAI_API_KEY"),
    ("gemini/gemini-3-flash-preview", "GOOGLE_API_KEY"),
    ("anthropic/claude-haiku-5-5", "ANTHROPIC_API_KEY"),
]


@pytest.mark.parametrize("model_id, key_env", LIVE_MODELS)
def test_live_extraction(model_id, key_env, definitions):
    if os.getenv("AATM_LIVE_LLM_TESTS") != "1":
        pytest.skip("set AATM_LIVE_LLM_TESTS=1 to call real providers")
    if not os.getenv(key_env):
        pytest.skip(f"{key_env} is not set")
    extractor = SimpleLLMExtractor(model_id=model_id, entity_definitions=definitions)
    result = asyncio.run(extractor([NOTE]))[0]
    texts = {e.text for e in result.validated_entities}
    assert "Hypertension" in texts
