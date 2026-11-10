"""LLM-based entity extractors.

`SimpleLLMExtractor` extracts entities described by `EntityDefinitions` from clinical
documents with one structured-output LLM call per document, then validates the result
with `aatm.data_models.extractors.validate_extracted_entity`.

Extraction is asynchronous; await the extractor (or use `asyncio.run` in scripts):

```python
from aatm.data_models.extractors import EntityDefinitions
from aatm.extractors import SimpleLLMExtractor

extractor = SimpleLLMExtractor(
    model_id="openai/gpt-5",  # or "gemini/...", "anthropic/...", "ollama/...", ...
    entity_definitions=EntityDefinitions.from_predefined("condition_occurrence"),
)
results = await extractor([clinical_note])
records = results[0].to_records()
```

Model ids are prefixed with a provider; see `aatm.llm.create_llm` for the list. The API
key is read from the provider's usual environment variable unless passed explicitly.
"""

import asyncio
import contextlib
import logging
from typing import Any, List, Literal, Optional

from aatm.data_models.extractors import (
    EntityDefinitions,
    EntityValidationResult,
    ExtractedEntityCandidates,
    format_segments_for_prompt,
    segment_text,
    validate_extracted_entity,
)
from aatm.llm import LLMSettings, StructuredLLM, create_llm

logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = """
# Goal

Your task is to extract relevant entities from a clinical note according to the entity definition provided.

# Instructions

You will receive the text of a clinical document split into numbered lines. Each line starts with its line id in brackets, for example `[L12]`.

You must extract spans that are directly correlated with one of the entities and the definition provided below. Each with a name, a key and a definition.

If the entity definition has a parent entity key, you must include the spans of the parent entity in the extracted entities.

For every extracted span:
- `id` is a short id that is unique within your response, for example `e1`, `e2`, `e3`.
- `parents` is the list of ids of the parent entities of this span. Each mention of a parent entity is a separate entity with its own id, so link each span to the specific mention it belongs to. A span may have several parents when it applies to several of them, for example one date for two conditions. Leave `parents` empty for entities without a parent key.
- `line` is the id of the line where this specific mention appears, for example `L12`. When the same text appears on several lines, choose the line of the mention you mean, and extract it once per line if several mentions are relevant.
- `text` must be copied exactly from that line, character by character. Do not include the `[L12]` line id in the text.
- `occurrence` is only needed when the same text appears more than once on that same line: it is the 1-based position of the mention you mean. Otherwise leave it null.
- `attributes` is only used for entities whose definition lists attributes. For each attribute the note states for this specific mention, give its `key` and, as `value`, the one label whose description matches best, written exactly as defined. Leave out any attribute the note does not state; never guess. Leave `attributes` empty for entities without attributes.

Spans that cannot be found on the given line will be considered invalid.

# Response format

Your response must be in JSON format following the schema below:

{output_format}

# Entity definitions

{entity_definitions}
""".strip()


class SimpleLLMExtractor:
    """Extract entities from texts with a single structured-output LLM call per text.

    The text is split into numbered lines (`segment_text`) before being sent to the
    model, which must return the line id of each mention. The output is validated with
    `validate_extracted_entity`, which grounds each span to one position on its line and
    checks keys and parent references.

    Args:
        model_id: Provider-prefixed model id, e.g. ``"openai/gpt-5"``,
            ``"gemini/gemini-3-flash-preview"``, ``"anthropic/claude-opus-5-5"`` or
            ``"ollama/llama3.3"``. See `aatm.llm.create_llm`. Ignored when ``llm`` is
            given.
        entity_definitions: Definitions of the entities to extract.
        instructions: System prompt template. The placeholders ``{output_format}`` and
            ``{entity_definitions}`` are replaced; other braces are kept as-is.
        output_type: Structured output model requested from the LLM.
        api_key: Optional API key. If omitted, the provider's default environment
            variable (e.g. ``OPENAI_API_KEY``) is used.
        base_url: Server URL for OpenAI-compatible providers (vLLM, Ollama, ...).
        settings: Generation settings, as an `aatm.llm.LLMSettings` or a compatible
            dict, e.g. ``{"temperature": 0, "max_tokens": 8000}``.
        max_concurrency: Maximum number of concurrent model calls made by this
            extractor. ``None`` (default) leaves the cap to the model's shared rate
            limiter (`aatm.llm.set_rate_limits`), which also retries rate-limit and
            transient errors.
        max_segment_chars: If set, lines longer than this are split at sentence
            boundaries into separately numbered segments before being sent to the
            model. ``None`` (default) keeps every line whole.
        llm: A ready-made `aatm.llm.StructuredLLM` to use instead of building one
            from ``model_id``, e.g. a custom provider.
        on_error: What to do when a text cannot be extracted once retries are spent:
            ``"raise"`` (default) raises the error; ``"record"`` returns an empty
            result for that text with the error in its ``error`` field, so one note
            does not stop a batch.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        entity_definitions: Optional[EntityDefinitions] = None,
        instructions: str = DEFAULT_INSTRUCTION,
        output_type: type[ExtractedEntityCandidates] = ExtractedEntityCandidates,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        settings: Optional[LLMSettings | dict[str, Any]] = None,
        max_concurrency: Optional[int] = None,
        max_segment_chars: Optional[int] = None,
        llm: Optional[StructuredLLM] = None,
        on_error: Literal["raise", "record"] = "raise",
    ):
        if entity_definitions is None:
            raise ValueError("entity_definitions is required")
        if on_error not in ("raise", "record"):
            raise ValueError("on_error must be 'raise' or 'record'")
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1 or None")
        if llm is None:
            if model_id is None:
                raise ValueError("Either model_id or llm must be given")
            llm = create_llm(
                model_id, api_key=api_key, base_url=base_url, settings=settings
            )

        self.llm = llm
        self.model_id = llm.model_id
        self.semaphore: Optional[asyncio.Semaphore] = (
            asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
        )
        self.on_error = on_error
        self.entity_definitions = entity_definitions
        self.max_segment_chars = max_segment_chars

        # Plain replacement instead of str.format so that literal braces in custom
        # instructions do not break prompt construction.
        self.instructions = instructions.replace(
            "{output_format}", str(output_type.model_json_schema())
        ).replace("{entity_definitions}", entity_definitions.to_prompt_str())
        self.output_type = output_type

    async def extract(self, text: str) -> EntityValidationResult:
        segments = segment_text(text, max_segment_chars=self.max_segment_chars)

        try:
            async with self.semaphore or contextlib.nullcontext():
                candidates = await self.llm.generate(
                    system=self.instructions,
                    user=format_segments_for_prompt(segments),
                    output_type=self.output_type,
                )
        except Exception as e:
            if self.on_error == "raise":
                raise
            logger.warning("Extraction failed with %s: %s", self.model_id, e)
            return EntityValidationResult(error=f"{type(e).__name__}: {e}")

        return validate_extracted_entity(
            text=text,
            extracted_entities=candidates,
            entity_definitions=self.entity_definitions,
            segments=segments,
        )

    async def __call__(self, texts: List[str]) -> List[EntityValidationResult]:
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a list of strings.")

        if not texts:
            return []

        return list(await asyncio.gather(*(self.extract(t) for t in texts)))
