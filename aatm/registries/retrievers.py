"""
Register and instantiate vector-database-backed retriever configurations.

This module defines the registry of supported embedding models used to build
retrievers. Each registry entry stores the information required to construct a
retriever, including the model identifier, embedding function class, collection
name, and optional rate limit metadata.

The module also provides a factory function, `load_retriever`, which returns a
``ChromaDBRetriever`` or a ``PineconeRetriever`` for a registered model name,
depending on the configured vector database (see `aatm.vector_db`).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError

from ..device import get_device
from ..embedding_functions import (
    GemmaEmbeddingFunction,
    GemmaEmbeddingModels,
    GoogleEmbeddingFunction,
    OpenAIEmbeddingFunction,
    OpenAIEmbeddingModels,
    Qwen3EmbeddingFunction,
    Qwen3EmbeddingModels,
)
from ..precision import COLLECTION_DTYPE_KEY, dtype_for_queries
from ..retrievers import ChromaDBRetriever, PineconeRetriever, VectorStoreRetriever
from ..vector_db import (
    ChromaConnection,
    PineconeConnection,
    VectorDbConnection,
    resolve_vector_db_mode,
)


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
    def cloud_collection_name(self) -> str:
        """Return the collection name used in Chroma Cloud.

        A local database holds one model's index, so its collection is simply
        `collection_name`. In Chroma Cloud one database holds every model's
        index, so the collection name includes the registry name.

        Returns:
            The collection name, e.g. ``expressions-qwen3-06B``.
        """
        return f"{self.collection_name}-{self.name}"

    def chroma_location(
        self, vector_db: str | None = None, local_path: str | None = None
    ) -> tuple[ChromaConnection, str]:
        """Return where this retriever's index lives and its collection name.

        Args:
            vector_db: ``local`` or ``cloud``. If ``None``, ``AATM_VECTOR_DB``
                decides, and otherwise ``local``. See `aatm.vector_db`.
            local_path: Directory of the local database. Defaults to
                `chromadb_path`. Ignored in cloud mode.

        Returns:
            The connection and the collection name to use with it.

        Raises:
            VectorDatabaseConfigError: If the configuration is invalid.
        """
        connection = ChromaConnection.from_env(
            local_path=local_path or self.chromadb_path, mode=vector_db
        )
        if connection.is_cloud:
            return connection, self.cloud_collection_name
        return connection, self.collection_name

    def pinecone_index_name(self, connection: PineconeConnection) -> str:
        """Return the name of this model's Pinecone index.

        Every Pinecone index has a fixed dimension, so each embedding model
        gets its own, named ``<PINECONE_INDEX_PREFIX>-<name>`` and made valid
        for Pinecone (see `aatm.vector_db.pinecone_index_name`).
        """
        return connection.index_name(self.name)

    def pinecone_location(self) -> tuple[PineconeConnection, str]:
        """Return the Pinecone connection and this model's index name.

        Raises:
            VectorDatabaseConfigError: If ``PINECONE_API_KEY`` is not set.
        """
        connection = PineconeConnection.from_env()
        return connection, self.pinecone_index_name(connection)

    def vector_db_location(
        self, vector_db: str | None = None, local_path: str | None = None
    ) -> tuple[VectorDbConnection, str]:
        """Return where this retriever's index lives, for any vector database.

        Args:
            vector_db: ``local``, ``cloud`` or ``pinecone``. If ``None``,
                ``AATM_VECTOR_DB`` decides, and otherwise ``local``.
            local_path: Directory of the local database. Defaults to
                `chromadb_path`. Only used in local mode.

        Returns:
            The connection and the collection or index name.

        Raises:
            VectorDatabaseConfigError: If the configuration is invalid.
        """
        if resolve_vector_db_mode(vector_db) == "pinecone":
            return self.pinecone_location()
        return self.chroma_location(vector_db, local_path)

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


def load_retriever(
    model_name: str, vector_db: str | None = None, **retriever_kwargs: Any
) -> VectorStoreRetriever:
    """Instantiate a registered retriever by model name.

    This function looks up a retriever specification in the model registry,
    opens the configured vector database (a local persistent ChromaDB
    database, Chroma Cloud or Pinecone, see `aatm.vector_db`), instantiates the
    corresponding embedding function, and returns a configured retriever.

    Args:
        model_name: Registry name of the retriever model to instantiate.
        vector_db: ``local``, ``cloud`` or ``pinecone``. If ``None``,
            ``AATM_VECTOR_DB`` decides, and otherwise ``local``.
        **retriever_kwargs: Passed on to the retriever, e.g. ``top_k``,
            ``filter_strategy`` or ``overfetch_factor``.

    Returns:
        A ``ChromaDBRetriever`` (local or cloud) or a ``PineconeRetriever``.

    Raises:
        ValueError: If the provided model name is not present in the retriever
            registry, or the vector database configuration is invalid.
    """
    try:
        spec = CHROMADB_RETRIEVER_MODEL_REGISTRY[model_name]
    except KeyError as e:
        available = ", ".join(sorted(CHROMADB_RETRIEVER_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown retriever model '{model_name}'. Available models: {available}"
        ) from e

    if resolve_vector_db_mode(vector_db) == "pinecone":
        return _load_pinecone_retriever(spec, **retriever_kwargs)

    connection, collection_name = spec.chroma_location(vector_db)
    client = connection.client()

    # Encode queries in the precision the database was built with.
    model_kwargs: dict[str, Any] = {}
    if getattr(spec.embedding_function_cls, "is_local", False):
        try:
            metadata = client.get_collection(collection_name).metadata or {}
        except NotFoundError:
            metadata = {}
        dtype = dtype_for_queries(metadata.get(COLLECTION_DTYPE_KEY), get_device())
        if dtype is not None:
            model_kwargs["dtype"] = dtype

    retriever = ChromaDBRetriever(
        client=client,
        collection_name=collection_name,
        embedding_function=spec.embedding_function_cls(
            model=spec.model_id,
            **model_kwargs,  # type: ignore[call-arg]
        ),
        **retriever_kwargs,
    )
    return retriever


def _query_model_kwargs(
    spec: ChromaRetrieverModelSpec, index_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Embedding function arguments to encode queries like the index was built."""
    if not getattr(spec.embedding_function_cls, "is_local", False):
        return {}
    dtype = dtype_for_queries(index_metadata.get(COLLECTION_DTYPE_KEY), get_device())
    return {} if dtype is None else {"dtype": dtype}


def _load_pinecone_retriever(
    spec: ChromaRetrieverModelSpec, **retriever_kwargs: Any
) -> PineconeRetriever:
    """Open a model's Pinecone index and return a retriever for it.

    Raises:
        ValueError: If the index does not exist or Pinecone is not configured.
    """
    from ..vector_stores import PineconeVectorStore, pinecone_index_exists

    connection, index_name = spec.pinecone_location()
    client = connection.client()
    if not pinecone_index_exists(client, index_name):
        raise ValueError(
            f"Pinecone has no index '{index_name}' for '{spec.name}'. Build it with "
            f"`aatm init --vector-db pinecone --embedding-model {spec.name}`, or copy "
            f"a local index with `aatm vector-db upload --to pinecone -e {spec.name}`."
        )
    store = PineconeVectorStore.open(client, index_name)
    model_kwargs = _query_model_kwargs(spec, store.index_metadata())
    return PineconeRetriever(
        store=store,
        embedding_function=spec.embedding_function_cls(
            model=spec.model_id,
            **model_kwargs,  # type: ignore[call-arg]
        ),
        **retriever_kwargs,
    )
