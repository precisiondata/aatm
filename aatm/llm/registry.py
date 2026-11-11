"""Build a `StructuredLLM` from a provider-prefixed model id."""

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

from aatm.llm.base import LLMSettings, StructuredLLM

if TYPE_CHECKING:
    from aatm.llm.decisions import DecisionModel

ProviderFactory = Callable[..., StructuredLLM]
"""Called as ``factory(model, api_key=..., base_url=..., settings=...)``."""


def _openai(model: str, *, api_key, base_url, settings) -> StructuredLLM:
    from aatm.llm.openai import OpenAILLM

    return OpenAILLM(model, api_key=api_key, base_url=base_url, settings=settings)


def _gemini(model: str, *, api_key, base_url, settings) -> StructuredLLM:
    from aatm.llm.gemini import GeminiLLM

    if base_url is not None:
        raise ValueError("base_url is not supported for the gemini provider.")
    return GeminiLLM(model, api_key=api_key, settings=settings)


def _anthropic(model: str, *, api_key, base_url, settings) -> StructuredLLM:
    from aatm.llm.anthropic import AnthropicLLM

    if base_url is not None:
        raise ValueError("base_url is not supported for the anthropic provider.")
    return AnthropicLLM(model, api_key=api_key, settings=settings)


def _compatible(
    provider: str, default_base_url: Optional[str], key_env: Optional[str]
) -> ProviderFactory:
    def factory(model: str, *, api_key, base_url, settings) -> StructuredLLM:
        from aatm.llm.openai import OpenAICompatibleLLM

        url = base_url or default_base_url
        if url is None:
            raise ValueError(
                f"The {provider} provider needs a base_url, "
                "e.g. base_url='http://localhost:8000/v1'."
            )
        key = api_key or (os.getenv(key_env) if key_env else None)
        return OpenAICompatibleLLM(
            model, base_url=url, api_key=key, settings=settings, provider=provider
        )

    return factory


_PROVIDERS: dict[str, ProviderFactory] = {
    "openai": _openai,
    "gemini": _gemini,
    "anthropic": _anthropic,
    "openai-compatible": _compatible("openai-compatible", None, None),
    "ollama": _compatible("ollama", "http://localhost:11434/v1", None),
    "openrouter": _compatible(
        "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"
    ),
}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register (or replace) a provider prefix usable in model ids."""
    if not name or "/" in name:
        raise ValueError(f"Invalid provider name: {name!r}")
    _PROVIDERS[name] = factory


def available_providers() -> list[str]:
    """Return the registered provider prefixes."""
    return sorted(_PROVIDERS)


def create_llm(
    model_id: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    settings: Optional[LLMSettings | dict[str, Any]] = None,
) -> StructuredLLM:
    """Create an LLM client from a ``provider/model`` id.

    Args:
        model_id: Provider-prefixed model id, e.g. ``"openai/gpt-5"``,
            ``"gemini/gemini-3-flash-preview"``, ``"anthropic/claude-opus-5-5"`` or
            ``"ollama/llama3.3"``. Only the first ``/`` separates the provider, so
            model names may contain slashes (``"openrouter/qwen/qwen3-235b"``).
        api_key: API key. If omitted, the provider's usual environment variable is used.
        base_url: Server URL for OpenAI and OpenAI-compatible providers.
        settings: Generation settings, as an `LLMSettings` or a compatible dict.
    """
    if "/" not in model_id:
        raise ValueError(
            "model_id must be prefixed with a provider, e.g. 'openai/<model>'. "
            f"Got {model_id!r}. Available providers: {', '.join(available_providers())}"
        )
    provider, model = model_id.split("/", 1)
    if not model:
        raise ValueError(f"model_id has no model name: {model_id!r}")
    factory = _PROVIDERS.get(provider)
    if factory is None:
        raise ValueError(
            f"Unknown provider {provider!r} in {model_id!r}. "
            f"Available providers: {', '.join(available_providers())}. "
            "Use 'openai-compatible/<model>' with base_url for other "
            "OpenAI-compatible servers, or register one with register_provider()."
        )

    if settings is None:
        llm_settings = LLMSettings()
    elif isinstance(settings, LLMSettings):
        llm_settings = settings
    elif isinstance(settings, dict):
        llm_settings = LLMSettings(**settings)
    else:
        raise ValueError("settings must be an LLMSettings object or a compatible dict")

    return factory(model, api_key=api_key, base_url=base_url, settings=llm_settings)


# --- Decision models -------------------------------------------------------------------

DecisionFactory = Callable[..., "DecisionModel"]
"""Called as ``factory(model, api_key=..., base_url=..., settings=...)``."""


def _openai_decisions(model: str, *, api_key, base_url, settings) -> "DecisionModel":
    from aatm.llm.openai_decisions import OpenAIDecisionModel

    return OpenAIDecisionModel(
        model, api_key=api_key, base_url=base_url, settings=settings
    )


def _gemini_decisions(model: str, *, api_key, base_url, settings) -> "DecisionModel":
    from aatm.llm.gemini_decisions import GeminiDecisionModel

    if base_url is not None:
        raise ValueError("base_url is not supported for the gemini provider.")
    return GeminiDecisionModel(model, api_key=api_key, settings=settings)


def _openrouter_decisions(
    model: str, *, api_key, base_url, settings
) -> "DecisionModel":
    from aatm.llm.jev import OPENROUTER_DECISIONS_URL, JevDecisionModel

    return JevDecisionModel(
        model,
        url=base_url or OPENROUTER_DECISIONS_URL,
        api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
        provider="openrouter",
        settings=settings,
    )


def _jev_compatible(model: str, *, api_key, base_url, settings) -> "DecisionModel":
    from aatm.llm.jev import JevDecisionModel

    if base_url is None:
        raise ValueError(
            "The jev-compatible provider needs base_url: the full URL of the "
            "decisions endpoint, e.g. 'https://api.edenai.run/v3/alpha/decisions'."
        )
    return JevDecisionModel(
        model,
        url=base_url,
        api_key=api_key,
        provider="jev-compatible",
        settings=settings,
    )


def _llm_decisions(model: str, *, api_key, base_url, settings) -> "DecisionModel":
    from aatm.llm.decisions import LLMDecisionModel

    return LLMDecisionModel(create_llm(model, api_key=api_key, base_url=base_url))


_DECISION_PROVIDERS: dict[str, DecisionFactory] = {
    "openai": _openai_decisions,
    "gemini": _gemini_decisions,
    "openrouter": _openrouter_decisions,
    "jev-compatible": _jev_compatible,
    "llm": _llm_decisions,
}


def register_decision_provider(name: str, factory: DecisionFactory) -> None:
    """Register (or replace) a provider prefix usable in decision model ids."""
    if not name or "/" in name:
        raise ValueError(f"Invalid provider name: {name!r}")
    _DECISION_PROVIDERS[name] = factory


def available_decision_providers() -> list[str]:
    """Return the registered decision provider prefixes."""
    return sorted(_DECISION_PROVIDERS)


def create_decision_model(
    model_id: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    settings: Optional[Any] = None,
) -> "DecisionModel":
    """Create a decision model from a ``provider/model`` id.

    Args:
        model_id: Provider-prefixed model id:

            - ``"openai/gpt-6-luna"``: the OpenAI Decisions API;
            - ``"gemini/gemini-decision-1"``: the Gemini Decisions API (key from
              ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``);
            - ``"openrouter/typesafe/jev-1.13"``: OpenRouter's decisions endpoint
              (key from ``OPENROUTER_API_KEY``);
            - ``"jev-compatible/<model>"``: any endpoint with Jev's request shape,
              with ``base_url`` the full endpoint URL;
            - ``"llm/<llm id>"``, e.g. ``"llm/anthropic/claude-haiku-5-5"``: any
              `create_llm` model through structured output, **without
              probabilities**.
        api_key: API key. If omitted, the provider's usual environment variable is used.
        base_url: Server URL (OpenAI), or the endpoint URL (OpenRouter,
            ``jev-compatible``), or the LLM's server URL (``llm/``).
        settings: A `DecisionSettings` or a compatible dict. Ignored by ``llm/``
            models; configure the LLM with `create_llm` and `LLMDecisionModel` instead.
    """
    from aatm.llm.decisions import DecisionSettings

    if "/" not in model_id:
        raise ValueError(
            "model_id must be prefixed with a provider, e.g. 'openai/gpt-6-luna'. "
            f"Got {model_id!r}. Available providers: "
            f"{', '.join(available_decision_providers())}"
        )
    provider, model = model_id.split("/", 1)
    if not model:
        raise ValueError(f"model_id has no model name: {model_id!r}")
    factory = _DECISION_PROVIDERS.get(provider)
    if factory is None:
        raise ValueError(
            f"Unknown decision provider {provider!r} in {model_id!r}. Available "
            f"providers: {', '.join(available_decision_providers())}. Use "
            "'llm/<llm id>' to answer with any LLM, or register one with "
            "register_decision_provider()."
        )
    if settings is None:
        decision_settings = DecisionSettings()
    elif isinstance(settings, DecisionSettings):
        decision_settings = settings
    elif isinstance(settings, dict):
        decision_settings = DecisionSettings(**settings)
    else:
        raise ValueError(
            "settings must be a DecisionSettings object or a compatible dict"
        )
    return factory(
        model, api_key=api_key, base_url=base_url, settings=decision_settings
    )
