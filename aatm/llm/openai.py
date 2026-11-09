"""OpenAI and OpenAI-compatible adapters."""

from typing import Any, Optional

from openai import AsyncOpenAI

from aatm.llm.base import LLMResponseError, LLMSettings, T
from aatm.llm.usage import record_usage, usage_attr


class OpenAILLM:
    """OpenAI models through the Responses API with strict structured outputs."""

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        settings: Optional[LLMSettings] = None,
    ):
        self.model = model
        self.model_id = f"openai/{model}"
        self.settings = settings or LLMSettings()
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _request_kwargs(self) -> dict[str, Any]:
        s = self.settings
        kwargs: dict[str, Any] = {}
        if s.temperature is not None:
            kwargs["temperature"] = s.temperature
        if s.max_tokens is not None:
            kwargs["max_output_tokens"] = s.max_tokens
        if s.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": s.reasoning_effort}
        if s.timeout is not None:
            kwargs["timeout"] = s.timeout
        return {**kwargs, **s.provider_kwargs}

    async def generate(self, *, system: str, user: str, output_type: type[T]) -> T:
        response = await self.client.responses.parse(
            model=self.model,
            instructions=system,
            input=user,
            text_format=output_type,
            **self._request_kwargs(),
        )
        usage = getattr(response, "usage", None)
        record_usage(
            usage_attr(usage, "input_tokens"), usage_attr(usage, "output_tokens")
        )
        if response.output_parsed is None:
            raise LLMResponseError(
                f"{self.model_id} returned no structured output "
                f"(status: {response.status}, incomplete: {response.incomplete_details})."
            )
        return response.output_parsed


class OpenAICompatibleLLM:
    """Any server exposing an OpenAI-compatible Chat Completions API.

    Covers vLLM, Ollama, OpenRouter, LM Studio, Together, Groq and similar services.
    Structured output is requested with ``response_format`` (JSON schema); the server
    and model must support it.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        settings: Optional[LLMSettings] = None,
        provider: str = "openai-compatible",
    ):
        self.model = model
        self.model_id = f"{provider}/{model}"
        self.settings = settings or LLMSettings()
        # Local servers usually need no key, but the OpenAI client requires one.
        self.client = AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url)

    def _request_kwargs(self) -> dict[str, Any]:
        s = self.settings
        kwargs: dict[str, Any] = {}
        if s.temperature is not None:
            kwargs["temperature"] = s.temperature
        if s.max_tokens is not None:
            kwargs["max_completion_tokens"] = s.max_tokens
        if s.reasoning_effort is not None:
            kwargs["reasoning_effort"] = s.reasoning_effort
        if s.timeout is not None:
            kwargs["timeout"] = s.timeout
        return {**kwargs, **s.provider_kwargs}

    async def generate(self, *, system: str, user: str, output_type: type[T]) -> T:
        completion = await self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=output_type,
            **self._request_kwargs(),
        )
        usage = getattr(completion, "usage", None)
        record_usage(
            usage_attr(usage, "prompt_tokens"), usage_attr(usage, "completion_tokens")
        )
        if not completion.choices:
            raise LLMResponseError(f"{self.model_id} returned no choices.")
        message = completion.choices[0].message
        if message.parsed is None:
            raise LLMResponseError(
                f"{self.model_id} returned no structured output "
                f"(finish reason: {completion.choices[0].finish_reason}, "
                f"refusal: {message.refusal})."
            )
        return message.parsed
