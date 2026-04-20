from typing import List
from fastapi import FastAPI
from aatm.api.config import APIConfig
from aatm.api.data_models import TerminologyMappingRequest
from aatm.data_models import MappedSourceConcept
from aatm.terminology_mapper import TerminologyMapper
from aatm.time import Timer

app = FastAPI()

api_config = APIConfig.load_from_disk()

PIPELINE_COMPONENTS_REGISTRY = {}


@app.post("/map", response_model=List[MappedSourceConcept])
def map(request: TerminologyMappingRequest):

    # Load terminology mapper
    tm_key = (
        request.translator_id,
        request.retriever_id,
        request.selector_id,
        request.reranker_id,
    )
    if tm_key in PIPELINE_COMPONENTS_REGISTRY:
        tm = PIPELINE_COMPONENTS_REGISTRY[tm_key]
    else:
        tm = TerminologyMapper.from_task_request(request, api_config)
        PIPELINE_COMPONENTS_REGISTRY[tm_key] = tm

    # Run terminology mapping
    mapped_concepts = tm.map(
        request.source_concepts,
        save_to_disk=False,
        return_as="mapped_source_concepts",
    )
    return mapped_concepts
