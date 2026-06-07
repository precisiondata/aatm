"""FastAPI application for terminology mapping and retrieval workflows.

This module defines API endpoints for terminology mapping and retrieval, along
with an in-memory least-recently-used registry for caching pipeline components.
The registry avoids repeatedly instantiating expensive retrievers and
terminology mappers across requests.
"""

from collections import OrderedDict
from typing import List
from fastapi import FastAPI
from aatm.api.config import APIConfig
from aatm.api.data_models import SearchRequest, TerminologyMappingRequest
from aatm.data_models import MappedSourceConcept, RetrieverResults
from aatm.pipeline import PipelineBaseClass
from aatm.registries.retrievers import load_retriever
from aatm.retrievers import BaseRetriever, ChromaDBRetriever
from aatm.terminology_mapper import TerminologyMapper

app = FastAPI()

api_config = APIConfig.load_from_disk()


class ComponentRegistry:
    """Least-recently-used in-memory registry for pipeline components.

    This registry stores instantiated terminology mappers and other pipeline
    components keyed by configuration tuples. When the registry reaches its maximum
    capacity, the least recently used item is evicted.

    Args:
        max_size: Maximum number of cached components to retain.

    Attributes:
        max_size: Maximum number of cached components.
        _store: Ordered mapping from cache keys to instantiated pipeline
            components.
    """

    def __init__(self, max_size: int = 10):
        """Initialize the component registry.

        Args:
            max_size: Maximum number of cached components to retain before evicting
                the least recently used entry.
        """
        self.max_size = max_size
        self._store: OrderedDict[tuple | str, TerminologyMapper | PipelineBaseClass] = (
            OrderedDict()
        )

    def get(self, key: tuple | str) -> TerminologyMapper | PipelineBaseClass | None:
        """Retrieve a cached component by key.

        If the key is present, the corresponding component is marked as recently used
        before being returned.

        Args:
            key: Cache key identifying the component.

        Returns:
            The cached `TerminologyMapper` or `PipelineBaseClass` instance associated
            with the key, or `None` if the key is not present.
        """
        if key not in self._store:
            return None

        # Mark as recently used
        self._store.move_to_end(key)
        return self._store[key]

    def set(
        self, key: tuple | str, value: TerminologyMapper | PipelineBaseClass
    ) -> None:
        """Store a component in the registry.

        If the key already exists, the existing entry is updated and marked as recently
        used. If the registry is full, the least recently used entry is removed before
        adding the new component.

        Args:
            key: Cache key identifying the component.
            value: Instantiated terminology mapper or pipeline component to cache.

        Returns:
            None.
        """
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return

        if len(self._store) >= self.max_size:
            # Remove least recently used item
            self._store.popitem(last=False)

        self._store[key] = value


PIPELINE_COMPONENTS_REGISTRY = ComponentRegistry(max_size=20)


@app.post("/map", response_model=List[MappedSourceConcept])
def map(request: TerminologyMappingRequest):
    """Map source concepts to target terminology concepts.

    This endpoint retrieves or creates a `TerminologyMapper` instance based on the
    requested pipeline component identifiers, then runs the mapping workflow over
    the provided source concepts.

    Args:
        request: Terminology mapping request containing source concepts and the
            identifiers of the translator, retriever, selector, and reranker
            components.

    Returns:
        A list of mapped source concepts produced by the terminology mapping
            pipeline.
    """
    tm_key = (
        request.translator_id,
        request.retriever_id,
        request.selector_id,
        request.reranker_id,
    )

    tm = PIPELINE_COMPONENTS_REGISTRY.get(tm_key)
    if tm is None:
        tm = TerminologyMapper.from_task_request(request, api_config)
        PIPELINE_COMPONENTS_REGISTRY.set(tm_key, tm)

    assert isinstance(tm, TerminologyMapper), (
        f"Expected TerminologyMapper, but got {type(tm)}"
    )

    mapped_concepts = tm.map(
        request.source_concepts,  # type: ignore[arg-type]
        save_to_disk=False,
        return_as="mapped_source_concepts",
    )
    return mapped_concepts


@app.post("/search", response_model=RetrieverResults)
def search(request: SearchRequest) -> RetrieverResults:
    """Search terminology candidates using a configured retriever.

    This endpoint retrieves or creates a retriever instance based on the requested
    retriever identifier, then executes the retrieval operation with the request
    parameters.

    Args:
        request: Search request containing the retriever identifier and query
            parameters.

    Returns:
        A `RetrieverResults` object containing the retrieval results.
    """
    retriever_key = f"retriever-{request.retriever_id}"
    retriever = PIPELINE_COMPONENTS_REGISTRY.get(retriever_key)

    assert isinstance(retriever, BaseRetriever) or retriever is None, (
        f"Expected BaseRetriever or None, but got {type(retriever)}"
    )
    if retriever is None:
        retriever = load_retriever(request.retriever_id)
        PIPELINE_COMPONENTS_REGISTRY.set(retriever_key, retriever)

    return retriever(**request.model_dump())
