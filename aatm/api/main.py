from typing import List
from fastapi import FastAPI
from aatm.api.data_models import TerminologyMappingRequest
from aatm.data_models import MappedSourceConcept
from aatm.terminology_mapper import TerminologyMapper

app = FastAPI()


@app.post("/map", response_model=List[MappedSourceConcept])
def map(request: TerminologyMappingRequest):
    # Load terminology mapper
    tm = TerminologyMapper.from_task_request(request)

    # Run terminology mapping
    mapped_concepts = tm.map(
        request.source_concepts, save_to_disk=False, return_as="mapped_source_concepts"
    )
    return mapped_concepts
