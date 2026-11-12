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
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Sequence

from aatm.data_models.extractors import (
    EntityDefinitions,
    EntityValidationResult,
    ExtractedEntityCandidates,
    TextSegment,
    format_segments_for_prompt,
    segment_text,
    validate_extracted_entity,
)
from aatm.llm import LLMSettings, StructuredLLM, create_llm
from aatm.llm.batch import BatchLLM, BatchRequest, supports_batch

if TYPE_CHECKING:
    from aatm.extractors.attribute_labeller import DecisionAttributeLabeller

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
            extractor. ``None`` (default) sets no cap: all texts are sent at once,
            within any limits set for the model with `aatm.llm.set_rate_limits`
            (none by default). Rate-limit and transient errors are retried either way.
        max_segment_chars: If set, lines longer than this are split at sentence
            boundaries into separately numbered segments before being sent to the
            model. ``None`` (default) keeps every line whole.
        llm: A ready-made `aatm.llm.StructuredLLM` to use instead of building one
            from ``model_id``, e.g. a custom provider.
        on_error: What to do when a text cannot be extracted once retries are spent:
            ``"raise"`` (default) raises the error; ``"record"`` returns an empty
            result for that text with the error in its ``error`` field, so one note
            does not stop a batch.
        attribute_labeller: Opt in to labelling attributes with a decision model,
            with probabilities, instead of (or as a check on) this extractor's own
            labels; see `aatm.extractors.DecisionAttributeLabeller`. A labeller
            error counts as an extraction error (see ``on_error``).
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
        attribute_labeller: Optional["DecisionAttributeLabeller"] = None,
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
        self.max_concurrency = max_concurrency
        self.semaphore: Optional[asyncio.Semaphore] = (
            asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None
        )
        self.on_error = on_error
        self.entity_definitions = entity_definitions
        self.max_segment_chars = max_segment_chars
        self.attribute_labeller = attribute_labeller
        # The definitions the model sees and its output is validated against: in
        # the labeller's replace mode, without the attributes it labels.
        self._prompt_definitions = (
            attribute_labeller.prompt_definitions(entity_definitions)
            if attribute_labeller is not None
            else entity_definitions
        )

        # Plain replacement instead of str.format so that literal braces in custom
        # instructions do not break prompt construction.
        self.instructions = instructions.replace(
            "{output_format}", str(output_type.model_json_schema())
        ).replace("{entity_definitions}", self._prompt_definitions.to_prompt_str())
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

        result = self._validate(text, candidates, segments)
        if self.attribute_labeller is None:
            return result
        try:
            return await self._label(result, segments)
        except Exception as e:
            if self.on_error == "raise":
                raise
            logger.warning("Attribute labelling failed: %s", e)
            return EntityValidationResult(error=f"{type(e).__name__}: {e}")

    def _validate(
        self,
        text: str,
        candidates: ExtractedEntityCandidates,
        segments: List[TextSegment],
    ) -> EntityValidationResult:
        if self.attribute_labeller is not None:
            candidates = self.attribute_labeller.strip_candidates(
                candidates, self.entity_definitions
            )
        return validate_extracted_entity(
            text=text,
            extracted_entities=candidates,
            entity_definitions=self._prompt_definitions,
            segments=segments,
        )

    async def _label(
        self, result: EntityValidationResult, segments: List[TextSegment]
    ) -> EntityValidationResult:
        assert self.attribute_labeller is not None
        return await self.attribute_labeller.label(
            result, segments, self.entity_definitions
        )

    # --- provider batch jobs ------------------------------------------------------

    @property
    def supports_batch(self) -> bool:
        """Whether the model can run provider batch jobs (OpenAI, Anthropic, Gemini)."""
        return supports_batch(self.llm)

    def _batch_llm(self) -> BatchLLM:
        if not supports_batch(self.llm):
            raise ValueError(
                f"{self.model_id} has no batch endpoint here: batch extraction "
                "supports the openai, anthropic and gemini providers"
            )
        return self.llm  # type: ignore[return-value]

    async def submit_batch(self, items: Sequence[tuple[str, str]]) -> str:
        """Submit texts as one provider batch job; return the job id.

        Args:
            items: ``(key, text)`` pairs. Keys are unique within the batch, made of
                letters, digits, ``_`` and ``-``.
        """
        requests = [
            BatchRequest(
                custom_id=key,
                system=self.instructions,
                user=format_segments_for_prompt(
                    segment_text(text, max_segment_chars=self.max_segment_chars)
                ),
            )
            for key, text in items
        ]
        return await self._batch_llm().submit_batch(requests, self.output_type)

    async def poll_batch(
        self, batch_id: str, items: Sequence[tuple[str, str]]
    ) -> Optional[dict[str, EntityValidationResult]]:
        """``None`` while the job runs; then the validated result of each key.

        `items` are the ``(key, text)`` pairs submitted. A request that failed gets
        a result with its ``error`` set, whatever ``on_error`` is.

        With an ``attribute_labeller``, the attributes are labelled here, with live
        decision calls, before the results are returned. A run that stops while
        they are labelled has not stored the results, so it polls the finished job
        again and labels them then.
        """
        keys = [key for key, _ in items]
        batch = await self._batch_llm().poll_batch(batch_id, keys, self.output_type)
        if batch is None:
            return None
        results: dict[str, EntityValidationResult] = {}
        segments_of: dict[str, List[TextSegment]] = {}
        for (key, text), item in zip(items, batch):
            if item.error is not None or item.output is None:
                results[key] = EntityValidationResult(error=item.error or "no output")
                continue
            try:
                segments = segment_text(text, max_segment_chars=self.max_segment_chars)
                results[key] = self._validate(text, item.output, segments)
                segments_of[key] = segments
            except Exception as e:  # noqa: BLE001 - one bad output must not stop a batch
                results[key] = EntityValidationResult(error=f"{type(e).__name__}: {e}")
        if self.attribute_labeller is not None:
            labelled = await asyncio.gather(
                *(self._label(results[k], s) for k, s in segments_of.items()),
                return_exceptions=True,
            )
            for key, outcome in zip(segments_of, labelled):
                if isinstance(outcome, BaseException):
                    error = f"{type(outcome).__name__}: {outcome}"
                    results[key] = EntityValidationResult(error=error)
                else:
                    results[key] = outcome
        return results

    async def __call__(self, texts: List[str]) -> List[EntityValidationResult]:
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a list of strings.")

        if not texts:
            return []

        return list(await asyncio.gather(*(self.extract(t) for t in texts)))
