"""Google Gemini adapter."""

from typing import Any, Optional, Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from aatm.llm.base import LLMResponseError, LLMSettings, T
from aatm.llm.batch import (
    BatchRequest,
    BatchResult,
    batch_call,
    check_requests,
    complete,
)
from aatm.llm.limits import estimate_tokens, get_rate_limiter
from aatm.llm.usage import batch_model_id, record_usage, usage_attr


# JSON Schema keywords kept in the schema sent to Gemini. Others (pattern, minLength,
# default, ...) are dropped to stay within the subset Gemini accepts; the returned
# JSON is validated against the full Pydantic model afterwards.
_GEMINI_SCHEMA_KEYS = {
    "$defs",
    "$ref",
    "additionalProperties",
    "anyOf",
    "description",
    "enum",
    "format",
    "items",
    "maxItems",
    "maximum",
    "minItems",
    "minimum",
    "prefixItems",
    "properties",
    "required",
    "title",
    "type",
}


def gemini_response_schema(schema: Any) -> Any:
    """Return a copy of a JSON schema restricted to keywords Gemini supports."""
    if isinstance(schema, list):
        return [gemini_response_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _GEMINI_SCHEMA_KEYS:
            continue
        if key in ("properties", "$defs"):
            # Keys of these maps are property / definition names, not keywords.
            cleaned[key] = {
                name: gemini_response_schema(sub) for name, sub in value.items()
            }
        else:
            cleaned[key] = gemini_response_schema(value)
    return cleaned


class GeminiLLM:
    """Gemini models through the Google GenAI SDK with a JSON response schema.

    The API key is read from ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``) unless passed.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        settings: Optional[LLMSettings] = None,
    ):
        self.model = model
        self.model_id = f"gemini/{model}"
        self.settings = settings or LLMSettings()
        http_options = None
        if self.settings.timeout is not None:
            http_options = types.HttpOptions(timeout=int(self.settings.timeout * 1000))
        self.client = genai.Client(api_key=api_key, http_options=http_options)

    def _config(self, system: str, output_type: type[T]) -> types.GenerateContentConfig:
        s = self.settings
        kwargs: dict[str, Any] = {
            "system_instruction": system,
            "response_mime_type": "application/json",
            "response_json_schema": gemini_response_schema(
                output_type.model_json_schema()
            ),
        }
        if s.temperature is not None:
            kwargs["temperature"] = s.temperature
        if s.max_tokens is not None:
            kwargs["max_output_tokens"] = s.max_tokens
        if s.reasoning_effort is not None:
            # Gemini has no level above "high".
            level = (
                "high" if s.reasoning_effort in ("xhigh", "max") else s.reasoning_effort
            )
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(level.upper())
            )
        return types.GenerateContentConfig(**{**kwargs, **s.provider_kwargs})

    async def generate(self, *, system: str, user: str, output_type: type[T]) -> T:
        return await get_rate_limiter(self.model_id).run(
            lambda: self._generate(system=system, user=user, output_type=output_type),
            estimated_tokens=estimate_tokens(system, user),
        )

    async def _generate(self, *, system: str, user: str, output_type: type[T]) -> T:
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=user,
            config=self._config(system, output_type),
        )
        return self._parse(response, output_type, self.model_id)

    def _parse(
        self,
        response: types.GenerateContentResponse,
        output_type: type[T],
        usage_id: str,
    ) -> T:
        """Record the usage of a response and validate its JSON text."""
        usage = getattr(response, "usage_metadata", None)
        output_tokens = usage_attr(usage, "candidates_token_count")
        thoughts = usage_attr(usage, "thoughts_token_count")
        if thoughts is not None:
            output_tokens = (output_tokens or 0) + thoughts
        record_usage(
            usage_attr(usage, "prompt_token_count"),
            output_tokens,
            model_id=usage_id,
        )
        if not response.text:
            reason = None
            if response.candidates:
                reason = response.candidates[0].finish_reason
            raise LLMResponseError(
                f"{self.model_id} returned an empty response (finish reason: {reason}, "
                f"prompt feedback: {response.prompt_feedback})."
            )
        try:
            return output_type.model_validate_json(response.text)
        except ValidationError as e:
            raise LLMResponseError(
                f"{self.model_id} returned output that does not match the schema: {e}"
            ) from e

    # --- batch mode -----------------------------------------------------------------

    async def submit_batch(
        self, requests: Sequence[BatchRequest], output_type: type[BaseModel]
    ) -> str:
        """Start a batch job with the requests inlined.

        Inlined requests are limited to about 20 MB in all: lower the chunk size of
        the run if this raises.
        """
        check_requests(requests)
        size = sum(len(r.system.encode()) + len(r.user.encode()) for r in requests)
        if size > GEMINI_INLINE_LIMIT:
            raise ValueError(
                f"The {len(requests)} requests hold {size / 1e6:.1f} MB of prompts; "
                f"Gemini takes at most {GEMINI_INLINE_LIMIT / 1e6:.0f} MB in a batch "
                "of inlined requests. Use a smaller chunk_size"
            )
        inlined = [
            types.InlinedRequest(
                contents=r.user,
                config=self._config(r.system, output_type),
                metadata={"key": r.custom_id},
            )
            for r in requests
        ]
        job = await batch_call(
            self.model_id,
            lambda: self.client.aio.batches.create(
                model=self.model,
                src=inlined,
                config=types.CreateBatchJobConfig(display_name="aatm-extraction"),
            ),
        )
        if not job.name:
            raise LLMResponseError(f"{self.model_id} returned a batch job without id")
        return job.name

    async def poll_batch(
        self, batch_id: str, custom_ids: Sequence[str], output_type: type[BaseModel]
    ) -> Optional[list[BatchResult]]:
        job = await batch_call(
            self.model_id, lambda: self.client.aio.batches.get(name=batch_id)
        )
        if job.state not in _GEMINI_ENDED:
            return None
        responses = (job.dest.inlined_responses if job.dest else None) or []
        results: dict[str, BatchResult] = {}
        # Responses carry the key sent as metadata; without any, they are matched
        # by position, as they keep the order of the requests.
        by_position = not any((item.metadata or {}).get("key") for item in responses)
        for position, item in enumerate(responses):
            if by_position:
                key = custom_ids[position] if position < len(custom_ids) else None
            else:
                key = (item.metadata or {}).get("key")
            if key is None:
                continue
            if item.error is not None or item.response is None:
                message = getattr(item.error, "message", None) or "no response"
                results[key] = BatchResult(
                    custom_id=key,
                    error=f"{self.model_id} batch request failed: {message}",
                )
                continue
            try:
                output = self._parse(
                    item.response, output_type, batch_model_id(self.model_id)
                )
            except LLMResponseError as e:
                results[key] = BatchResult(custom_id=key, error=str(e))
            else:
                results[key] = BatchResult(custom_id=key, output=output)
        state = getattr(job.state, "value", job.state)
        reason = f"batch {batch_id} ended ({state}) without a result"
        if job.error is not None:
            reason += f": {getattr(job.error, 'message', job.error)}"
        return complete(custom_ids, results, reason)


#: Bytes of prompts accepted in one batch of inlined requests (the API takes 20 MB
#: per request, framing included).
GEMINI_INLINE_LIMIT = 19_000_000

_GEMINI_ENDED = {
    types.JobState.JOB_STATE_SUCCEEDED,
    types.JobState.JOB_STATE_FAILED,
    types.JobState.JOB_STATE_CANCELLED,
    types.JobState.JOB_STATE_EXPIRED,
    types.JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
}
