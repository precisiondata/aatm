"""Count the tokens of LLM calls made inside a block of code.

```python
from aatm.llm.usage import track_usage

with track_usage() as usage:
    await llm.generate(system="...", user="...", output_type=MyModel)
usage.calls, usage.input_tokens, usage.output_tokens
```

Each adapter reports the usage of its calls with `record_usage`. Tracking is
per context (`contextvars`), so concurrent asyncio tasks each count their own calls.
Trackers can be nested: a call counts in every enclosing tracker.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Calls and tokens counted by a `track_usage` block.

    Attributes:
        calls: Number of LLM calls.
        input_tokens: Prompt tokens, as reported by the provider.
        output_tokens: Output tokens, reasoning tokens included when the provider
            counts them as output.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: Optional[int], output_tokens: Optional[int]) -> None:
        self.calls += 1
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0


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


def record_usage(input_tokens: Optional[int], output_tokens: Optional[int]) -> None:
    """Add one call to every active tracker. Called by the provider adapters."""
    for usage in _TRACKERS.get():
        usage.add(input_tokens, output_tokens)


def usage_attr(obj: Any, *names: str) -> Optional[int]:
    """The first integer attribute found among `names`, or ``None``."""
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, int):
            return value
    return None
