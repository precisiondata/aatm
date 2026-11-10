"""Anthropic Claude adapter.

Requires the optional dependency: ``pip install "aatm[anthropic]"``.
"""

import logging
from typing import Any, Optional, Sequence

from pydantic import BaseModel

from aatm.llm.base import LLMResponseError, LLMSettings, T
from aatm.llm.batch import (
    BatchRequest,
    BatchResult,
    batch_call,
    check_requests,
    complete,
    parse_output,
)
from aatm.llm.limits import estimate_tokens, get_rate_limiter
from aatm.llm.usage import batch_model_id, record_usage, usage_attr

try:
    import anthropic
    from anthropic.lib._parse._transform import transform_schema
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
        # Retries are handled by the shared limiter (`aatm.llm.limits`).
        client_kwargs: dict[str, Any] = {"max_retries": 0}
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
        return await get_rate_limiter(self.model_id).run(
            lambda: self._generate(system=system, user=user, output_type=output_type),
            estimated_tokens=estimate_tokens(system, user),
        )

    async def _generate(self, *, system: str, user: str, output_type: type[T]) -> T:
        message = await self.client.messages.parse(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=output_type,
            **self._request_kwargs(),
        )
        usage = getattr(message, "usage", None)
        record_usage(
            usage_attr(usage, "input_tokens"),
            usage_attr(usage, "output_tokens"),
            model_id=self.model_id,
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

    # --- Message Batches ------------------------------------------------------------

    def _batch_params(
        self, request: BatchRequest, output_type: type[BaseModel]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
            **self._request_kwargs(),
        }
        output_format = {
            "type": "json_schema",
            "schema": transform_schema(output_type.model_json_schema()),
        }
        params["output_config"] = {
            **params.get("output_config", {}),
            "format": output_format,
        }
        return params

    async def submit_batch(
        self, requests: Sequence[BatchRequest], output_type: type[BaseModel]
    ) -> str:
        """Start a Message Batches job with the requests."""
        check_requests(requests)
        batch_requests: list[Any] = [
            {"custom_id": r.custom_id, "params": self._batch_params(r, output_type)}
            for r in requests
        ]
        batch = await batch_call(
            self.model_id,
            lambda: self.client.messages.batches.create(requests=batch_requests),
        )
        return batch.id

    async def poll_batch(
        self, batch_id: str, custom_ids: Sequence[str], output_type: type[BaseModel]
    ) -> Optional[list[BatchResult]]:
        batch = await batch_call(
            self.model_id, lambda: self.client.messages.batches.retrieve(batch_id)
        )
        if batch.processing_status != "ended":
            return None

        async def read() -> dict[str, BatchResult]:
            entries = await self.client.messages.batches.results(batch_id)
            found: dict[str, BatchResult] = {}
            async for entry in entries:
                found[entry.custom_id] = self._batch_entry(entry, output_type)
            return found

        results = await batch_call(self.model_id, read)
        return complete(custom_ids, results, f"batch {batch_id} ended without a result")

    def _batch_entry(self, entry: Any, output_type: type[BaseModel]) -> BatchResult:
        custom_id = entry.custom_id
        result = entry.result
        if result.type != "succeeded":
            error = getattr(getattr(result, "error", None), "error", None)
            message = getattr(error, "message", None) or result.type
            return BatchResult(
                custom_id=custom_id,
                error=f"{self.model_id} batch request {result.type}: {message}",
            )
        message = result.message
        usage = getattr(message, "usage", None)
        record_usage(
            usage_attr(usage, "input_tokens"),
            usage_attr(usage, "output_tokens"),
            model_id=batch_model_id(self.model_id),
        )
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            return BatchResult(
                custom_id=custom_id,
                error=f"{self.model_id} declined the request "
                f"(category: {getattr(details, 'category', None)}).",
            )
        text = "".join(block.text for block in message.content if block.type == "text")
        try:
            output = parse_output(self.model_id, text, output_type)
        except LLMResponseError as e:
            return BatchResult(
                custom_id=custom_id,
                error=f"{e} (stop reason: {message.stop_reason})",
            )
        return BatchResult(custom_id=custom_id, output=output)
