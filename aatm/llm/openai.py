"""OpenAI and OpenAI-compatible adapters."""

from typing import Any, Optional, Sequence

import json

from openai import AsyncOpenAI
from openai.lib._parsing._responses import type_to_text_format_param
from pydantic import BaseModel

from aatm.llm.base import LLMResponseError, LLMSettings, PerEventLoop, T
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
        # Retries are handled by the shared limiter (`aatm.llm.limits`). One client
        # per event loop: openai 3.x binds its connection pool to the first loop.
        self._clients = PerEventLoop(
            lambda: AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        )

    @property
    def client(self) -> AsyncOpenAI:
        """The client of the running event loop."""
        return self._clients.get()

    @client.setter
    def client(self, value: AsyncOpenAI) -> None:
        self._clients.set(value)

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
        return await get_rate_limiter(self.model_id).run(
            lambda: self._generate(system=system, user=user, output_type=output_type),
            estimated_tokens=estimate_tokens(system, user),
        )

    async def _generate(self, *, system: str, user: str, output_type: type[T]) -> T:
        response = await self.client.responses.parse(
            model=self.model,
            instructions=system,
            input=user,
            text_format=output_type,
            **self._request_kwargs(),
        )
        usage = getattr(response, "usage", None)
        record_usage(
            usage_attr(usage, "input_tokens"),
            usage_attr(usage, "output_tokens"),
            model_id=self.model_id,
        )
        if response.output_parsed is None:
            raise LLMResponseError(
                f"{self.model_id} returned no structured output "
                f"(status: {response.status}, incomplete: {response.incomplete_details})."
            )
        return response.output_parsed

    # --- Batch API ------------------------------------------------------------------

    def _batch_body(
        self, request: BatchRequest, output_type: type[BaseModel]
    ) -> dict[str, Any]:
        kwargs = self._request_kwargs()
        kwargs.pop("timeout", None)  # a client option, not part of the request
        return {
            "model": self.model,
            "instructions": request.system,
            "input": request.user,
            "text": {"format": type_to_text_format_param(output_type)},
            **kwargs,
        }

    async def submit_batch(
        self, requests: Sequence[BatchRequest], output_type: type[BaseModel]
    ) -> str:
        """Upload the requests as a JSONL file and start a Batch API job."""
        check_requests(requests)
        lines = [
            json.dumps(
                {
                    "custom_id": r.custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": self._batch_body(r, output_type),
                },
                ensure_ascii=False,
            )
            for r in requests
        ]
        data = ("\n".join(lines) + "\n").encode("utf-8")
        file = await batch_call(
            self.model_id,
            lambda: self.client.files.create(
                file=("aatm-batch.jsonl", data), purpose="batch"
            ),
        )
        batch = await batch_call(
            self.model_id,
            lambda: self.client.batches.create(
                input_file_id=file.id,
                endpoint="/v1/responses",
                completion_window="24h",
            ),
        )
        return batch.id

    async def poll_batch(
        self, batch_id: str, custom_ids: Sequence[str], output_type: type[BaseModel]
    ) -> Optional[list[BatchResult]]:
        batch = await batch_call(
            self.model_id, lambda: self.client.batches.retrieve(batch_id)
        )
        if batch.status in _OPENAI_RUNNING:
            return None
        results: dict[str, BatchResult] = {}
        for file_id in (batch.output_file_id, batch.error_file_id):
            if not file_id:
                continue
            content = await batch_call(
                self.model_id, lambda: self.client.files.content(file_id)
            )
            for line in content.text.splitlines():
                if line.strip():
                    result = self._batch_line(json.loads(line), output_type)
                    results[result.custom_id] = result
        reason = f"batch {batch_id} ended ({batch.status}) without a result"
        errors = getattr(getattr(batch, "errors", None), "data", None)
        if errors:
            reason += ": " + "; ".join(str(getattr(e, "message", e)) for e in errors)
        return complete(custom_ids, results, reason)

    def _batch_line(
        self, line: dict[str, Any], output_type: type[BaseModel]
    ) -> BatchResult:
        custom_id = line.get("custom_id", "")
        response = line.get("response") or {}
        body = response.get("body") or {}
        usage = body.get("usage") or {}
        if usage:
            record_usage(
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                model_id=batch_model_id(self.model_id),
            )
        error = line.get("error") or body.get("error")
        if error or response.get("status_code") != 200:
            message = error.get("message") if isinstance(error, dict) else error
            return BatchResult(
                custom_id=custom_id,
                error=f"{self.model_id} batch request failed "
                f"(status {response.get('status_code')}): {message}",
            )
        texts: list[str] = []
        for item in body.get("output") or []:
            if item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if part.get("type") == "refusal":
                    return BatchResult(
                        custom_id=custom_id,
                        error=f"{self.model_id} refused: {part.get('refusal')}",
                    )
                if part.get("type") == "output_text":
                    texts.append(part.get("text") or "")
        try:
            output = parse_output(self.model_id, "".join(texts), output_type)
        except LLMResponseError as e:
            return BatchResult(
                custom_id=custom_id,
                error=f"{e} (status: {body.get('status')}, "
                f"incomplete: {body.get('incomplete_details')})",
            )
        return BatchResult(custom_id=custom_id, output=output)


_OPENAI_RUNNING = {"validating", "in_progress", "finalizing", "cancelling"}


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
        # Local servers usually need no key, but the OpenAI client requires one. One
        # client per event loop: openai 3.x binds its connection pool to the first.
        self._clients = PerEventLoop(
            lambda: AsyncOpenAI(
                api_key=api_key or "not-needed", base_url=base_url, max_retries=0
            )
        )

    @property
    def client(self) -> AsyncOpenAI:
        """The client of the running event loop."""
        return self._clients.get()

    @client.setter
    def client(self, value: AsyncOpenAI) -> None:
        self._clients.set(value)

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
        return await get_rate_limiter(self.model_id).run(
            lambda: self._generate(system=system, user=user, output_type=output_type),
            estimated_tokens=estimate_tokens(system, user),
        )

    async def _generate(self, *, system: str, user: str, output_type: type[T]) -> T:
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
            usage_attr(usage, "prompt_tokens"),
            usage_attr(usage, "completion_tokens"),
            model_id=self.model_id,
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
