from abc import ABC, abstractmethod
from typing import Any, List

import chromadb
from chromadb import EmbeddingFunction

from .data_models import RetrievedExpressionMetadata, RetrieverResults, Translation
from .pipeline import PipelineBaseClass


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
