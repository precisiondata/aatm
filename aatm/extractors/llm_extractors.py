import asyncio
from typing import Any, List, Optional
from agents import Agent, ModelSettings, OpenAIResponsesModel, Runner
from openai import AsyncOpenAI
from pydantic import BaseModel

from aatm.data_models.extractors import (
    EntityDefinitions,
    EntityValidationResult,
    ExtractedEntityCandidates,
    format_segments_for_prompt,
    segment_text,
    validate_extracted_entity,
)

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

Spans that cannot be found on the given line will be considered invalid.

# Response format

Your response must be in JSON format following the schema below:

{output_format}

# Entity definitions

{entity_definitions}
""".strip()


class SimpleLLMExtractor:
    """Extract entities from texts with a single LLM agent call per text.

    The text is split into numbered lines (`segment_text`) before being sent to the
    model, which must return the line id of each mention. The output is validated with
    `validate_extracted_entity`, which grounds each span to one position on its line and
    checks keys and parent references.

    Args:
        model_id: Provider-prefixed model id, e.g. ``"openai/gpt-5"``. Only the
            ``openai/`` provider is currently supported.
        entity_definitions: Definitions of the entities to extract.
        instructions: System prompt template. The placeholders ``{output_format}`` and
            ``{entity_definitions}`` are replaced; other braces are kept as-is.
        output_type: Structured output model for the agent.
        name: Name given to the agent.
        api_key: Optional API key. If omitted, the provider's default environment
            variable (e.g. ``OPENAI_API_KEY``) is used.
        model_settings: Agent model settings, as a `ModelSettings` or a compatible dict.
        max_concurrency: Maximum number of concurrent model calls.
        max_segment_chars: If set, lines longer than this are split at sentence
            boundaries into separately numbered segments before being sent to the
            model. ``None`` (default) keeps every line whole.
        agent_args: Extra keyword arguments forwarded to `agents.Agent`.
    """

    def __init__(
        self,
        model_id: str,
        entity_definitions: EntityDefinitions,
        instructions: str = DEFAULT_INSTRUCTION,
        output_type: type[ExtractedEntityCandidates] = ExtractedEntityCandidates,
        name: str = "Extractor",
        api_key: Optional[str] = None,
        model_settings: Optional[ModelSettings | dict[str, Any]] = None,
        max_concurrency: int = 1,
        max_segment_chars: Optional[int] = None,
        agent_args: Optional[dict[str, Any]] = None,
    ):
        if "/" not in model_id:
            raise ValueError(
                f"model_id must be prefixed with a provider, e.g. 'openai/<model>'. Got {model_id!r}"
            )
        self.provider, self.model_name = model_id.split("/", 1)
        if self.provider != "openai":
            raise NotImplementedError(
                "Support for providers other than OpenAI is not implemented yet."
            )

        self.model_id = model_id
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.entity_definitions = entity_definitions
        self.name = name
        self.max_segment_chars = max_segment_chars

        self.model_settings: ModelSettings
        if isinstance(model_settings, dict):
            self.model_settings = ModelSettings(**model_settings)
        elif isinstance(model_settings, ModelSettings):
            self.model_settings = model_settings
        elif model_settings is None:
            self.model_settings = ModelSettings()
        else:
            raise ValueError(
                "model_settings must be a ModelSettings object or a compatible dict"
            )

        # Plain replacement instead of str.format so that literal braces in custom
        # instructions do not break prompt construction.
        self.instructions = instructions.replace(
            "{output_format}", str(output_type.model_json_schema())
        ).replace("{entity_definitions}", entity_definitions.to_prompt_str())
        self.output_type: type[BaseModel] = output_type
        self.agent_args: dict[str, Any] = agent_args or {}

    def make_agent(self) -> Agent:
        model: str | OpenAIResponsesModel = self.model_name
        if self.api_key is not None:
            model = OpenAIResponsesModel(
                model=self.model_name,
                openai_client=AsyncOpenAI(api_key=self.api_key),
            )

        return Agent(
            name=self.name,
            instructions=self.instructions,
            model=model,
            model_settings=self.model_settings,
            output_type=self.output_type,
            **self.agent_args,
        )

    async def extract(self, text: str) -> EntityValidationResult:
        segments = segment_text(text, max_segment_chars=self.max_segment_chars)

        async with self.semaphore:
            agent = self.make_agent()
            run_result = await Runner.run(
                agent, input=format_segments_for_prompt(segments)
            )

        return validate_extracted_entity(
            text=text,
            extracted_entities=run_result.final_output_as(ExtractedEntityCandidates),
            entity_definitions=self.entity_definitions,
            segments=segments,
        )

    async def __call__(self, texts: List[str]) -> List[EntityValidationResult]:
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a list of strings.")

        if not texts:
            return []

        return list(await asyncio.gather(*(self.extract(t) for t in texts)))
