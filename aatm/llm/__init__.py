"""Provider-neutral structured-output LLM clients.

```python
from aatm.llm import create_llm

llm = create_llm("anthropic/claude-opus-5-5", settings={"reasoning_effort": "low"})
result = await llm.generate(system="...", user="...", output_type=MyModel)
```

Supported prefixes: ``openai``, ``gemini``, ``anthropic`` (needs ``aatm[anthropic]``),
``ollama``, ``openrouter`` and ``openai-compatible`` (any OpenAI-compatible server,
with ``base_url``). More can be added with `register_provider`.

Decision models answer closed-set questions (choice, score, yes/no) with
probabilities instead of writing text: see `aatm.llm.decisions` and
`create_decision_model`.

Calls are retried on rate limits and transient errors. No rate limits are set by
default; set them, per model, provider or for all, with `set_rate_limits`
(see `aatm.llm.limits`).
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
from aatm.llm.decisions import (
    Answer,
    Decision,
    DecisionError,
    DecisionModel,
    DecisionSettings,
    LLMDecisionModel,
    Option,
    Question,
)
from aatm.llm.registry import (
    available_decision_providers,
    available_providers,
    create_decision_model,
    create_llm,
    register_decision_provider,
    register_provider,
)
from aatm.llm.usage import ModelPrice, ModelTokens, TokenUsage, track_usage

__all__ = [
    "Answer",
    "Decision",
    "DecisionError",
    "DecisionModel",
    "DecisionSettings",
    "LLMDecisionModel",
    "Option",
    "Question",
    "available_decision_providers",
    "create_decision_model",
    "register_decision_provider",
    "LLMResponseError",
    "LLMSettings",
    "RateLimiter",
    "RateLimits",
    "ReasoningEffort",
    "StructuredLLM",
    "ModelPrice",
    "ModelTokens",
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
