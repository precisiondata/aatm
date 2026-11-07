"""Anthropic Claude adapter.

Requires the optional dependency: ``pip install "aatm[anthropic]"``.
"""

import logging
from typing import Any, Optional

from aatm.llm.base import LLMResponseError, LLMSettings, T

try:
    import anthropic
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError(
        "Anthropic models need the 'anthropic' package. "
        'Install it with: pip install "aatm[anthropic]"'
    ) from e

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 16000


class AnthropicLLM:
    """Claude models through the Messages API with native structured outputs.

    Credentials are resolved by the Anthropic SDK (``ANTHROPIC_API_KEY`` and the other
    sources it supports) unless an API key is passed.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        settings: Optional[LLMSettings] = None,
    ):
        self.model = model
        self.model_id = f"anthropic/{model}"
        self.settings = settings or LLMSettings()
        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if self.settings.timeout is not None:
            client_kwargs["timeout"] = self.settings.timeout
        self.client = anthropic.AsyncAnthropic(**client_kwargs)
        if self.settings.temperature is not None:
            logger.warning(
                "temperature is not supported by current Anthropic models and is ignored."
            )

    def _request_kwargs(self) -> dict[str, Any]:
        s = self.settings
        kwargs: dict[str, Any] = {"max_tokens": s.max_tokens or DEFAULT_MAX_TOKENS}
        if s.reasoning_effort is not None:
            kwargs["output_config"] = {"effort": s.reasoning_effort}
        return {**kwargs, **s.provider_kwargs}

    async def generate(self, *, system: str, user: str, output_type: type[T]) -> T:
        message = await self.client.messages.parse(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=output_type,
            **self._request_kwargs(),
        )
        if message.stop_reason == "refusal":
            details = message.stop_details
            raise LLMResponseError(
                f"{self.model_id} declined the request "
                f"(category: {getattr(details, 'category', None)})."
            )
        if message.parsed_output is None:
            raise LLMResponseError(
                f"{self.model_id} returned no structured output "
                f"(stop reason: {message.stop_reason})."
            )
        return message.parsed_output
