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
    """Configuration for a ChromaDB-backed retriever model."""

    name: str
    model_id: str
    embedding_function_cls: type[chromadb.EmbeddingFunction]
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
    """Instantiate a ChromaDBRetriever from a registered model name."""
    try:
        spec = CHROMADB_RETRIEVER_MODEL_REGISTRY[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(CHROMADB_RETRIEVER_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown retriever model '{model_name}'. Available models: {available}"
        ) from exc

    client = chromadb.PersistentClient(spec.chromadb_path)

    retriever = ChromaDBRetriever(
        client=client,
        collection_name=spec.collection_name,
        embedding_function=spec.embedding_function_cls(model=spec.model_id),
    )
    return retriever
