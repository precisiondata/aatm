"""Request models for the terminology mapping API.

This module defines Pydantic models used to validate incoming API requests for
terminology mapping and retrieval. It includes request schemas for mapping
source concepts through the terminology pipeline and for performing retriever-
based searches.
"""

from typing import Any, List, Optional
from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

from aatm.data_models import SourceConcept


class TerminologyMappingRequest(BaseModel):
    """Request model for terminology mapping operations.

    This model encapsulates the list of source concepts to be mapped together with
    the optional identifiers of the pipeline components used during the mapping
    workflow.

    Attributes:
        source_concepts: Source concepts to map to a target terminology.
        translator_id: Optional identifier of the translator component.
        retriever_id: Optional identifier of the retriever component.
        selector_id: Optional identifier of the selector component.
        reranker_id: Optional identifier of the reranker component.
    """

    source_concepts: List[SourceConcept]
    translator_id: Optional[str] = Field(None, examples=["gemini-2.5-flash"])
    retriever_id: Optional[str] = Field(None, examples=["embeddinggemma-300M"])
    selector_id: Optional[str] = Field(None, examples=["first-result-selector"])
    reranker_id: Optional[str] = Field(None, examples=[None])

    @field_validator("source_concepts", mode="after")
    def validate_source_concepts(cls, v: List[SourceConcept]):
        """Validate the list of source concepts provided in the request.

        This validator ensures that at least one source concept is provided and that
        all source concepts include a `source_code_description` value.

        Args:
            v: List of source concepts to validate.

        Returns:
            The validated list of source concepts.

        Raises:
            HTTPException: If no source concepts are provided.
            HTTPException: If one or more source concepts are missing the
                `source_code_description` field.
        """
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


class SearchRequest(BaseModel):
    """Request model for retriever-based search operations.

    Attributes:
        queries: List of query strings to search for.
        retriever_id: Identifier of the retriever to use.
        top_k: Maximum number of results to return per query.
        where: Optional metadata filter applied during retrieval.
    """

    queries: List[str] = Field(..., examples=[["Cardiovascular disease"]])
    retriever_id: str = Field(..., examples=["embeddinggemma-300M"])
    top_k: int = Field(10, examples=[10])
    where: dict[str, Any] | None = Field(None, examples=[None])
