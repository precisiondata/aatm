from abc import ABC, abstractmethod
from typing import List

from chromadb import EmbeddingFunction
import chromadb

from aatm.data_models import RetrievedExpressionMetadata, RetrieverResults, Translation
from aatm.embedding_functions import (
    GemmaEmbeddingFunction,
    GemmaEmbeddingModels,
    GoogleEmbeddingFunction,
    OpenAIEmbeddingFunction,
    OpenAIEmbeddingModels,
    Qwen3EmbeddingFunction,
    Qwen3Models,
)
from aatm.pipeline import PipelineBaseClass

CHROMADB_RETRIEVER_MODEL_REGISTRY = {
    "qwen3-06B": {
        "model_id": Qwen3Models.QWEN3_06B.value,
        "embedding_function": Qwen3EmbeddingFunction,
        "collection_name": "expressions",
        "chromadb_path": "chroma_vector_dbs/qwen3-06B",
        "output_path": "output/qwen3-06B",
    },
    "qwen3-4B": {
        "model_id": Qwen3Models.QWEN3_4B.value,
        "embedding_function": Qwen3EmbeddingFunction,
        "collection_name": "expressions",
        "chromadb_path": "chroma_vector_dbs/qwen3-4B",
        "output_path": "output/qwen3-4B",
    },
    "gemini-embedding-001": {
        "model_id": "gemini-embedding-001",
        "embedding_function": GoogleEmbeddingFunction,
        "collection_name": "expressions",
        "chromadb_path": "chroma_vector_dbs/gemini-embedding-001",
        "output_path": "output/gemini-embedding-001",
        "rate_limit": 3000,
    },
    "embeddinggemma-300M": {
        "model_id": GemmaEmbeddingModels.EMBEDDING_GEMMA_300M.value,
        "embedding_function": GemmaEmbeddingFunction,
        "collection_name": "expressions",
        "chromadb_path": "chroma_vector_dbs/embeddinggemma-300M",
        "output_path": "output/embeddinggemma-300M",
        "rate_limit": 1000,
    },
    "text-embedding-3-small": {
        "model_id": OpenAIEmbeddingModels.TEXT_EMBEDDING_3_SMALL.value,
        "embedding_function": OpenAIEmbeddingFunction,
        "collection_name": "expressions",
        "chromadb_path": "chroma_vector_dbs/text-embedding-3-small",
        "output_path": "output/text-embedding-3-small",
    },
    "text-embedding-3-large": {
        "model_id": OpenAIEmbeddingModels.TEXT_EMBEDDING_3_LARGE.value,
        "embedding_function": OpenAIEmbeddingFunction,
        "collection_name": "expressions",
        "chromadb_path": "chroma_vector_dbs/text-embedding-3-large",
        "output_path": "output/text-embedding-3-large",
    },
}


def load_chromadb_retriever(model_name: str) -> "ChromaDBRetriever":
    client = chromadb.PersistentClient(
        CHROMADB_RETRIEVER_MODEL_REGISTRY[model_name]["chromadb_path"]
    )
    retriever = ChromaDBRetriever(
        client=client,
        collection_name=CHROMADB_RETRIEVER_MODEL_REGISTRY[model_name][
            "collection_name"
        ],
        embedding_function=CHROMADB_RETRIEVER_MODEL_REGISTRY[model_name][
            "embedding_function"
        ](model=CHROMADB_RETRIEVER_MODEL_REGISTRY[model_name]["model_id"]),
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
        elif isinstance(queries, list) and isinstance(queries[0], Translation):
            queries = [t.text for t in queries]

        assert isinstance(queries, list) and isinstance(queries[0], str), (
            f"queries must be a string, a Translation, a list of strings, or a list of Translation objects. Got {queries}."
        )

        return self.retrieve(queries)


class ChromaDBRetriever(BaseRetriever):
    def __init__(
        self,
        client: chromadb.ClientAPI,
        collection_name: str,
        embedding_function: EmbeddingFunction,
        top_k: int = 10,
        where: str = None,
        *args,
        **kwargs,
    ):
        self.client = client
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.collection = self.client.get_or_create_collection(
            collection_name, embedding_function=embedding_function
        )
        self.top_k = top_k
        self.where = where

    def retrieve(
        self, queries: List[str], where: str = None, top_k: int = None
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
                results["distances"][query_id], results["metadatas"][query_id]
            ):
                query_results.append(
                    RetrievedExpressionMetadata(distance=distance, **metadata)
                )
            processed_results.append(query_results)

        return RetrieverResults(
            results=processed_results,
            queries=queries,
        )
