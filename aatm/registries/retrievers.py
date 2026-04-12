"""
Register and instantiate ChromaDB-backed retriever configurations.

This module defines the registry of supported embedding models used to build
ChromaDB retrievers. Each registry entry stores the information required to
construct a retriever, including the model identifier, embedding function
class, collection name, and optional rate limit metadata.

The module also provides a factory function for loading a configured
``ChromaDBRetriever`` instance from a registered model name.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from ..embedding_functions import (
    GemmaEmbeddingFunction,
    GemmaEmbeddingModels,
    GoogleEmbeddingFunction,
    OpenAIEmbeddingFunction,
    OpenAIEmbeddingModels,
    Qwen3EmbeddingFunction,
    Qwen3EmbeddingModels,
)
from ..retrievers import ChromaDBRetriever


@dataclass(slots=True, frozen=True)
class ChromaRetrieverModelSpec:
    """Store the configuration required for a ChromaDB-backed retriever.

    Instances of this dataclass describe a retriever model specification,
    including the embedding model identifier, the embedding function class used
    to encode queries, the target ChromaDB collection name, and optional rate
    limit metadata.
    """

    name: str
    """Unique registry name used to identify the retriever model."""

    model_id: str
    """Identifier of the embedding model used by the retriever."""

    embedding_function_cls: type[chromadb.EmbeddingFunction]
    """Embedding function class used to encode queries for retrieval."""

    collection_name: str = "expressions"
    """Name of the ChromaDB collection queried by the retriever."""

    rate_limit: int | None = None
    """Optional maximum throughput in items per minute for embedding generation."""

    @property
    def chromadb_path(self) -> str:
        """Return the filesystem path for this retriever's ChromaDB database.

        The path is derived from the retriever name and points to the local
        directory where the persistent ChromaDB collection is stored.

        Returns:
            The string path to the retriever's ChromaDB persistence directory.
        """
        return str(Path(".aatm/chroma_vector_dbs") / self.name)

    @property
    def output_path(self) -> str:
        """Return the default output directory associated with this retriever.

        This path can be used by downstream workflows to store outputs related
        to the retriever configuration.

        Returns:
            The string path to the default output directory for this retriever.
        """
        return str(Path("output") / self.name)

    def to_dict(self) -> dict[str, Any]:
        """Convert the retriever specification to a dictionary.

        This helper provides a dictionary representation of the retriever
        configuration for compatibility with code paths that expect
        mapping-based access instead of attribute-based access.

        Returns:
            A dictionary containing the retriever model identifier, embedding
                function class, collection name, ChromaDB path, output path, and
                rate limit.
        """
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
        embedding_function_cls=GemmaEmbeddingFunction,  # noqa: F821
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


def load_retriever(model_name: str) -> "ChromaDBRetriever":
    """Instantiate a registered ChromaDB retriever by model name.

    This function looks up a retriever specification in the model registry,
    creates a persistent ChromaDB client for the configured storage path,
    instantiates the corresponding embedding function, and returns a configured
    ``ChromaDBRetriever``.

    Args:
        model_name: Registry name of the retriever model to instantiate.

    Returns:
        A configured ``ChromaDBRetriever`` instance for the requested model.

    Raises:
        ValueError: If the provided model name is not present in the retriever
            registry.
    """
    try:
        spec = CHROMADB_RETRIEVER_MODEL_REGISTRY[model_name]
    except KeyError as e:
        available = ", ".join(sorted(CHROMADB_RETRIEVER_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown retriever model '{model_name}'. Available models: {available}"
        ) from e

    client = chromadb.PersistentClient(spec.chromadb_path)

    retriever = ChromaDBRetriever(
        client=client,
        collection_name=spec.collection_name,
        embedding_function=spec.embedding_function_cls(model=spec.model_id),
    )
    return retriever
