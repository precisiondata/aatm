"""Count the tokens of LLM calls made inside a block of code, and estimate their cost.

```python
from aatm.llm.usage import track_usage

with track_usage() as usage:
    await llm.generate(system="...", user="...", output_type=MyModel)
usage.calls, usage.input_tokens, usage.output_tokens
usage.by_model                     # the same counts per model id
usage.cost({"openai/gpt-5-mini": (0.25, 2.00)})  # USD per million input/output tokens
```

Each adapter reports the usage of its calls with `record_usage`. Tracking is
per context (`contextvars`), so concurrent asyncio tasks each count their own calls.
Trackers can be nested: a call counts in every enclosing tracker. Threads started
with `asyncio.to_thread` inherit the trackers of the caller.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

import threading

from pydantic import BaseModel

UNKNOWN_MODEL = "unknown"
#: Appended to the model id of calls made through a provider batch endpoint.
BATCH_SUFFIX = "@batch"
#: Share of the live price that batch calls cost, when no ``@batch`` price is given.
BATCH_DISCOUNT = 0.5


def batch_model_id(model_id: str) -> str:
    """The id under which batch calls of `model_id` are counted."""
    return model_id if model_id.endswith(BATCH_SUFFIX) else model_id + BATCH_SUFFIX


# A tracker may be shared by the event loop and a worker thread.
_LOCK = threading.Lock()


class ModelPrice(BaseModel):
    """The price of a model, in a currency of your choice (usually USD).

    Attributes:
        input: Price per million input (prompt) tokens.
        output: Price per million output tokens, reasoning tokens included.
    """

    input: float
    output: float


#: Prices per model id, e.g. ``{"openai/gpt-5-mini": (0.25, 2.0)}``: a `ModelPrice`,
#: a dict with ``input`` and ``output``, or a pair (input, output), per million tokens.
Prices = Mapping[str, Union[ModelPrice, Mapping[str, float], Sequence[float]]]


def model_prices(prices: Optional[Prices]) -> dict[str, ModelPrice]:
    """Normalize `Prices` to `ModelPrice` objects."""
    normalized: dict[str, ModelPrice] = {}
    for model_id, price in (prices or {}).items():
        if isinstance(price, ModelPrice):
            normalized[model_id] = price
        elif isinstance(price, Mapping):
            normalized[model_id] = ModelPrice.model_validate(price)
        else:
            input_price, output_price = price
            normalized[model_id] = ModelPrice(input=input_price, output=output_price)
    return normalized


def price_of(model_id: str, prices: dict[str, ModelPrice]) -> Optional[ModelPrice]:
    """The price of `model_id` in normalized `prices`, or ``None``.

    Batch calls (``<model>@batch``) without a price of their own cost the model's
    price times `BATCH_DISCOUNT`.
    """
    price = prices.get(model_id)
    if price is not None or not model_id.endswith(BATCH_SUFFIX):
        return price
    live = prices.get(model_id[: -len(BATCH_SUFFIX)])
    if live is None:
        return None
    return ModelPrice(
        input=live.input * BATCH_DISCOUNT, output=live.output * BATCH_DISCOUNT
    )


class ModelTokens(BaseModel):
    """Calls and tokens of one model.

    Attributes:
        calls: Number of LLM calls.
        input_tokens: Prompt tokens, as reported by the provider.
        output_tokens: Output tokens, reasoning tokens included when the provider
            counts them as output.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def cost_at(self, price: ModelPrice) -> float:
        """The cost of these tokens at `price`."""
        return (
            self.input_tokens * price.input + self.output_tokens * price.output
        ) / 1e6


class TokenUsage(ModelTokens):
    """Calls and tokens counted by a `track_usage` block, in all and per model.

    Attributes:
        by_model: The counts per model id (e.g. ``"openai/gpt-5-mini"``). Calls
            recorded without a model id count under ``"unknown"``.
    """

    by_model: dict[str, ModelTokens] = {}

    def add(
        self,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        model_id: Optional[str] = None,
    ) -> None:
        with _LOCK:
            model = self.by_model.get(model_id or UNKNOWN_MODEL)
            if model is None:
                model = self.by_model[model_id or UNKNOWN_MODEL] = ModelTokens()
            for counts in (self, model):
                counts.calls += 1
                counts.input_tokens += input_tokens or 0
                counts.output_tokens += output_tokens or 0

    def merge(self, other: "TokenUsage") -> None:
        """Add the counts of another tracker to this one."""
        with _LOCK:
            self.calls += other.calls
            self.input_tokens += other.input_tokens
            self.output_tokens += other.output_tokens
            for model_id, counts in other.by_model.items():
                mine = self.by_model.setdefault(model_id, ModelTokens())
                mine.calls += counts.calls
                mine.input_tokens += counts.input_tokens
                mine.output_tokens += counts.output_tokens

    def cost(self, prices: Optional[Prices]) -> Optional[float]:
        """The estimated cost of the calls, or ``None`` when no prices are given.

        Models without a price are left out: see `unpriced`. Batch calls
        (``<model>@batch``) cost the model's price times `BATCH_DISCOUNT`, unless
        `prices` has the ``@batch`` id.
        """
        normalized = model_prices(prices)
        if not normalized:
            return None
        total = 0.0
        for model_id, counts in self.by_model.items():
            price = price_of(model_id, normalized)
            if price is not None:
                total += counts.cost_at(price)
        return total

    def unpriced(self, prices: Optional[Prices]) -> list[str]:
        """The models called that have no price in `prices`."""
        normalized = model_prices(prices)
        return sorted(m for m in self.by_model if price_of(m, normalized) is None)


_TRACKERS: ContextVar[tuple[TokenUsage, ...]] = ContextVar(
    "aatm_llm_usage_trackers", default=()
)


@contextmanager
def track_usage() -> Iterator[TokenUsage]:
    """Count the LLM calls and tokens made inside the ``with`` block."""
    usage = TokenUsage()
    token = _TRACKERS.set((*_TRACKERS.get(), usage))
    try:
        yield usage
    finally:
        _TRACKERS.reset(token)


def record_usage(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    model_id: Optional[str] = None,
) -> None:
    """Add one call to every active tracker. Called by the provider adapters."""
    for usage in _TRACKERS.get():
        usage.add(input_tokens, output_tokens, model_id)


def usage_attr(obj: Any, *names: str) -> Optional[int]:
    """The first integer attribute found among `names`, or ``None``."""
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, int):
            return value
    return None
