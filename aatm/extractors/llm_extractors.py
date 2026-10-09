import asyncio
from typing import List, Optional
from agents import Agent, ModelSettings, Runner
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
    def __init__(
        self,
        model_id: str,
        entity_definitions: EntityDefinitions,
        instructions: str = DEFAULT_INSTRUCTION,
        output_type: type[BaseModel] = ExtractedEntityCandidates,
        name: str = "Extractor",
        api_key: Optional[str] = None,
        model_settings: Optional[ModelSettings] = None,
        max_concurrency: int = 1,
        agent_args: Optional[dict] = {},
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.entity_definitions = entity_definitions

        # agent args
        self.name = name

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

        self.instructions = instructions.format(
            output_format=output_type.model_json_schema(),
            entity_definitions=entity_definitions.to_prompt_str(),
        )
        self.output_type = output_type
        self.agent_args = agent_args

    def make_agent(self) -> Agent:
        if self.model_id.startswith("openai/"):
            agent = Agent(
                name="Extractor",
                instructions=self.instructions,
                model=self.model_id.split("/")[-1],
                model_settings=self.model_settings,
                output_type=self.output_type,
                **self.agent_args,
            )
        else:
            raise NotImplementedError(
                "Support for providers other than OpenAI is not implemented yet."
            )

        return agent

    async def extract(self, text: str) -> EntityValidationResult:
        async with self.semaphore:
            agent = self.make_agent()
            result = await Runner.run(agent, input=text)

            result = validate_extracted_entity(
                text=text,
                extracted_entities=result.final_output,
                entity_definitions=self.entity_definitions,
            )
            return result

    async def __call__(self, texts: List[str]) -> List[EntityValidationResult]:
        if not isinstance(texts, list) or not isinstance(texts[0], str):
            raise ValueError("Input must be a list of strings.")

        aio_tasks = [self.extract(t) for t in texts]
        results = await asyncio.gather(*aio_tasks)
        return results
