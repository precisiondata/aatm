from typing import List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

from aatm.data_models import SourceConcept


class TerminologyMappingRequest(BaseModel):
    source_concepts: List[SourceConcept]
    translator_id: Optional[str] = Field(None, examples=["gemini-2.5-flash"])
    retriever_id: Optional[str] = Field(None, examples=["embeddinggemma-300M"])
    selector_id: Optional[str] = Field(None, examples=["first-result-selector"])
    reranker_id: Optional[str] = Field(None, examples=[None])

    @field_validator("source_concepts", mode="after")
    def validate_source_concepts(cls, v: List[SourceConcept]):
        if len(v) == 0:
            raise HTTPException(status_code=400, detail="No source concepts provided")

        incomplete_source_concepts = [
            source_concept
            for source_concept in v
            if source_concept.source_code_description is None
        ]
        if len(incomplete_source_concepts) > 0:
            raise HTTPException(
                status_code=400,
                detail=f"The field source_code_description is required for all source concepts. A total of {len(incomplete_source_concepts)} source concepts are missing this field.",
            )

        return v
