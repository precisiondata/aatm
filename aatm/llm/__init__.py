"""Provider-neutral structured-output LLM clients.

```python
from aatm.llm import create_llm

llm = create_llm("anthropic/claude-opus-5-5", settings={"reasoning_effort": "low"})
result = await llm.generate(system="...", user="...", output_type=MyModel)
```

Supported prefixes: ``openai``, ``gemini``, ``anthropic`` (needs ``aatm[anthropic]``),
``ollama``, ``openrouter`` and ``openai-compatible`` (any OpenAI-compatible server,
with ``base_url``). More can be added with `register_provider`.
"""

from aatm.llm.base import LLMResponseError, LLMSettings, ReasoningEffort, StructuredLLM
from aatm.llm.registry import available_providers, create_llm, register_provider

__all__ = [
    "LLMResponseError",
    "LLMSettings",
    "ReasoningEffort",
    "StructuredLLM",
    "available_providers",
    "create_llm",
    "register_provider",
]
