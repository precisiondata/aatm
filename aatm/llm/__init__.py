"""Provider-neutral structured-output LLM clients.

```python
from aatm.llm import create_llm

llm = create_llm("anthropic/claude-opus-5-5", settings={"reasoning_effort": "low"})
result = await llm.generate(system="...", user="...", output_type=MyModel)
```

Supported prefixes: ``openai``, ``gemini``, ``anthropic`` (needs ``aatm[anthropic]``),
``ollama``, ``openrouter`` and ``openai-compatible`` (any OpenAI-compatible server,
with ``base_url``). More can be added with `register_provider`.

Every call goes through one limiter per model, with retries on rate limits and
transient errors; set its limits with `set_rate_limits` (see `aatm.llm.limits`).
"""

from aatm.llm.limits import (
    RateLimiter,
    RateLimits,
    get_rate_limiter,
    get_rate_limits,
    reset_rate_limits,
    set_rate_limits,
)
from aatm.llm.base import LLMResponseError, LLMSettings, ReasoningEffort, StructuredLLM
from aatm.llm.registry import available_providers, create_llm, register_provider
from aatm.llm.usage import TokenUsage, track_usage

__all__ = [
    "LLMResponseError",
    "LLMSettings",
    "RateLimiter",
    "RateLimits",
    "ReasoningEffort",
    "StructuredLLM",
    "TokenUsage",
    "available_providers",
    "create_llm",
    "get_rate_limiter",
    "get_rate_limits",
    "register_provider",
    "reset_rate_limits",
    "set_rate_limits",
    "track_usage",
]
