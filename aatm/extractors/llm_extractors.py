import asyncio
from typing import Any, List, Optional
from agents import Agent, ModelSettings, OpenAIResponsesModel, Runner
from openai import AsyncOpenAI
from pydantic import BaseModel

from aatm.data_models.extractors import (
    EntityDefinitions,
    EntityValidationResult,
    ExtractedEntityCandidates,
    validate_extracted_entity,
)

DEFAULT_INSTRUCTION = """
# Goal

Your task is to extract relevant entities from a clinical note according to the entity definition provided.

# Instructions

You will receive the text related to a clinical document.

You must extract spans that are directly correlated with one of the entities and the definition provided below. Each with a name, a key and a definition.

If the entity definition has a parent entity key, you must include the spans of the parent entity in the extracted entities.

The span must match exactly with a span from the clinical document provided. Spans that cannot be found in the document text will be considered invalid.

# Response format

Your response must be in JSON format following the schema below:

{output_format}

# Entity definitions

{entity_definitions}
""".strip()


class SimpleLLMExtractor:
    """Extract entities from texts with a single LLM agent call per text.

    The model output is validated with `validate_extracted_entity`, which keeps only
    spans found verbatim in the text and checks keys and parent references.

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
        async with self.semaphore:
            agent = self.make_agent()
            run_result = await Runner.run(agent, input=text)

        return validate_extracted_entity(
            text=text,
            extracted_entities=run_result.final_output_as(ExtractedEntityCandidates),
            entity_definitions=self.entity_definitions,
        )

    async def __call__(self, texts: List[str]) -> List[EntityValidationResult]:
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a list of strings.")

        if not texts:
            return []

        return list(await asyncio.gather(*(self.extract(t) for t in texts)))
