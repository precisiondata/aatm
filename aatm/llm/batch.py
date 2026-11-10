"""Provider batch endpoints: many prompts sent as one job, about half the price.

OpenAI (Batch API), Anthropic (Message Batches) and Gemini (batch mode) take a list
of requests, run them within 24 hours and charge about 50% of the live price. A
`BatchLLM` submits requests and polls the job:

```python
from aatm.llm import create_llm
from aatm.llm.batch import BatchRequest, supports_batch

llm = create_llm("openai/gpt-5-mini")
assert supports_batch(llm)
batch_id = await llm.submit_batch(
    [BatchRequest(custom_id="note-1", system="...", user="...")], MyModel
)
results = await llm.poll_batch(batch_id, ["note-1"], MyModel)  # None while running
results[0].output, results[0].error
```

Usage of batch calls is recorded under the model id with ``@batch`` appended
(``"openai/gpt-5-mini@batch"``). `aatm.llm.usage.TokenUsage.cost` prices it at the
model's price times `BATCH_DISCOUNT` unless a price is given for the ``@batch`` id.

`aatm.omop.OmopExtractionPipeline.run` uses this with ``batch=True``.
"""

from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

import re

from pydantic import BaseModel, ValidationError

from aatm.llm.base import LLMResponseError, T
from aatm.llm.limits import get_rate_limiter
from aatm.llm.usage import BATCH_DISCOUNT, BATCH_SUFFIX, batch_model_id

__all__ = [
    "BATCH_DISCOUNT",
    "BATCH_SUFFIX",
    "BatchLLM",
    "BatchRequest",
    "BatchResult",
    "batch_model_id",
    "supports_batch",
]

R = TypeVar("R")

# Anthropic's rule for custom ids; the strictest of the three providers.
_CUSTOM_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class BatchRequest(BaseModel):
    """One prompt of a batch.

    Attributes:
        custom_id: The key of the request, unique within the batch: letters, digits,
            ``_`` and ``-``, at most 64 characters.
        system: The system prompt.
        user: The user message.
    """

    custom_id: str
    system: str
    user: str


class BatchResult(BaseModel):
    """The outcome of one request of a finished batch: an output or an error."""

    custom_id: str
    output: Optional[Any] = None
    error: Optional[str] = None


@runtime_checkable
class BatchLLM(Protocol):
    """An LLM that can also run prompts as a provider batch job."""

    model_id: str

    async def submit_batch(
        self, requests: Sequence[BatchRequest], output_type: type[BaseModel]
    ) -> str:
        """Submit the requests as one job; return its id."""
        ...

    async def poll_batch(
        self, batch_id: str, custom_ids: Sequence[str], output_type: type[BaseModel]
    ) -> Optional[list[BatchResult]]:
        """``None`` while the job runs; once it ended, one result per custom id.

        A request the job returned nothing for (the job failed, expired or was
        cancelled) gets an error.
        """
        ...


def supports_batch(llm: Any) -> bool:
    """Whether `llm` can run batch jobs."""
    return isinstance(llm, BatchLLM)


def check_requests(requests: Sequence[BatchRequest]) -> None:
    if not requests:
        raise ValueError("A batch needs at least one request")
    seen: set[str] = set()
    for request in requests:
        if not _CUSTOM_ID.match(request.custom_id):
            raise ValueError(
                f"Invalid custom_id {request.custom_id!r}: use letters, digits, '_' "
                "and '-', at most 64 characters"
            )
        if request.custom_id in seen:
            raise ValueError(f"Duplicate custom_id {request.custom_id!r}")
        seen.add(request.custom_id)


async def batch_call(model_id: str, call: Callable[[], Awaitable[R]]) -> R:
    """Call a batch endpoint, retrying rate-limit and transient errors.

    Uses the limiter of the ``@batch`` model id, so batch calls do not take from the
    live calls' rate limits.
    """
    return await get_rate_limiter(batch_model_id(model_id)).run(call)


def parse_output(model_id: str, text: Optional[str], output_type: type[T]) -> T:
    """Validate the JSON text of a batch response against `output_type`."""
    if not text:
        raise LLMResponseError(f"{model_id} returned an empty response.")
    try:
        return output_type.model_validate_json(text)
    except ValidationError as e:
        raise LLMResponseError(
            f"{model_id} returned output that does not match the schema: {e}"
        ) from e


def complete(
    custom_ids: Sequence[str],
    results: dict[str, BatchResult],
    missing: str,
) -> list[BatchResult]:
    """One result per custom id, in order; those not returned get `missing` as error."""
    return [
        results.get(custom_id) or BatchResult(custom_id=custom_id, error=missing)
        for custom_id in custom_ids
    ]
