from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import chromadb
from chromadb import EmbeddingFunction

from aatm.data_models import RetrievedExpressionMetadata, RetrieverResults, Translation
from aatm.embedding_functions import (
    GemmaEmbeddingFunction,
    GemmaEmbeddingModels,
    GoogleEmbeddingFunction,
    OpenAIEmbeddingFunction,
    OpenAIEmbeddingModels,
    Qwen3EmbeddingFunction,
    Qwen3EmbeddingModels,
)
from aatm.pipeline import PipelineBaseClass


@dataclass(slots=True, frozen=True)
class ChromaRetrieverModelSpec:
    """Configuration for a ChromaDB-backed retriever model."""

    name: str
    model_id: str
    embedding_function_cls: type[EmbeddingFunction]
    collection_name: str = "expressions"
    rate_limit: int | None = None

    @property
    def chromadb_path(self) -> str:
        return str(Path(".aatm/chroma_vector_dbs") / self.name)

    @property
    def output_path(self) -> str:
        return str(Path("output") / self.name)

    def to_dict(self) -> dict[str, Any]:
        """Optional compatibility helper if some older code still expects a dict."""
        return {
            "model_id": self.model_id,
            "embedding_function": self.embedding_function_cls,
            "collection_name": self.collection_name,
            "chromadb_path": self.chromadb_path,
            "output_path": self.output_path,
            "rate_limit": self.rate_limit,
        }


MODEL_SPECS: list[ChromaRetrieverModelSpec] = [
    ChromaRetrieverModelSpec(
        name="qwen3-06B",
        model_id=Qwen3EmbeddingModels.QWEN3_06B.value,
        embedding_function_cls=Qwen3EmbeddingFunction,
    ),
    ChromaRetrieverModelSpec(
        name="qwen3-4B",
        model_id=Qwen3EmbeddingModels.QWEN3_4B.value,
        embedding_function_cls=Qwen3EmbeddingFunction,
    ),
    ChromaRetrieverModelSpec(
        name="gemini-embedding-001",
        model_id="gemini-embedding-001",
        embedding_function_cls=GoogleEmbeddingFunction,
        rate_limit=3000,
    ),
    ChromaRetrieverModelSpec(
        name="embeddinggemma-300M",
        model_id=GemmaEmbeddingModels.EMBEDDING_GEMMA_300M.value,
        embedding_function_cls=GemmaEmbeddingFunction,
        rate_limit=1000,
    ),
    ChromaRetrieverModelSpec(
        name="text-embedding-3-small",
        model_id=OpenAIEmbeddingModels.TEXT_EMBEDDING_3_SMALL.value,
        embedding_function_cls=OpenAIEmbeddingFunction,
    ),
    ChromaRetrieverModelSpec(
        name="text-embedding-3-large",
        model_id=OpenAIEmbeddingModels.TEXT_EMBEDDING_3_LARGE.value,
        embedding_function_cls=OpenAIEmbeddingFunction,
    ),
]

CHROMADB_RETRIEVER_MODEL_REGISTRY: dict[str, ChromaRetrieverModelSpec] = {
    spec.name: spec for spec in MODEL_SPECS
}


def get_chromadb_retriever_model_spec(model_name: str) -> ChromaRetrieverModelSpec:
    """Return the model spec for a registered retriever model."""
    try:
        return CHROMADB_RETRIEVER_MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(CHROMADB_RETRIEVER_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown retriever model '{model_name}'. Available models: {available}"
        ) from exc


def load_chromadb_retriever(model_name: str) -> "ChromaDBRetriever":
    """Instantiate a ChromaDBRetriever from a registered model name."""
    spec = get_chromadb_retriever_model_spec(model_name)

    client = chromadb.PersistentClient(spec.chromadb_path)

    retriever = ChromaDBRetriever(
        client=client,
        collection_name=spec.collection_name,
        embedding_function=spec.embedding_function_cls(model=spec.model_id),
    )
    return retriever


class BaseRetriever(PipelineBaseClass, ABC):
    @abstractmethod
    def retrieve(self, queries: List[str]) -> RetrieverResults:
        pass

    def __call__(
        self, queries: str | Translation | List[str] | List[Translation]
    ) -> RetrieverResults:
        if isinstance(queries, str):
            queries = [queries]
        elif isinstance(queries, Translation):
            queries = [queries.text]
        elif (
            isinstance(queries, list)
            and queries
            and isinstance(queries[0], Translation)
        ):
            queries = [t.text for t in queries]

        assert isinstance(queries, list) and queries and isinstance(queries[0], str), (
            "queries must be a string, a Translation, a list of strings, or a list "
            f"of Translation objects. Got {queries}."
        )

        return self.retrieve(queries)


class ChromaDBRetriever(BaseRetriever):
    def __init__(
        self,
        client: chromadb.ClientAPI,
        collection_name: str,
        embedding_function: EmbeddingFunction,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ):
        self.client = client
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.collection = self.client.get_or_create_collection(
            collection_name,
            embedding_function=embedding_function,
        )
        self.top_k = top_k
        self.where = where

    def retrieve(
        self,
        queries: List[str],
        where: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> RetrieverResults:
        results = self.collection.query(
            query_texts=queries,
            n_results=top_k if top_k is not None else self.top_k,
            where=where if where is not None else self.where,
        )

        processed_results = []
        for query_id, _ in enumerate(queries):
            query_results = []
            for distance, metadata in zip(
                results["distances"][query_id],
                results["metadatas"][query_id],
            ):
                query_results.append(
                    RetrievedExpressionMetadata(distance=distance, **metadata)
                )
            processed_results.append(query_results)

        return RetrieverResults(
            results=processed_results,
            queries=queries,
        )
