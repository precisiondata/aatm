from abc import ABC, abstractmethod
from typing import List

from chromadb import EmbeddingFunction
import chromadb

from aatm.data_models import RetrievedExpressionMetadata, Translation
from aatm.pipeline import PipelineBaseClass


class BaseRetriever(PipelineBaseClass, ABC):
    @abstractmethod
    def __call__(
        self, text: str | List[str] | List[Translation]
    ) -> List[RetrievedExpressionMetadata]:
        pass


class ChromaDBRetriever(BaseRetriever):
    def __init__(
        self,
        client: chromadb.ClientAPI,
        collection_name: str,
        embedding_function: EmbeddingFunction,
        top_k: int = 10,
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

    def __call__(
        self, text: str | Translation | List[str] | List[Translation]
    ) -> List[List[RetrievedExpressionMetadata]]:
        if isinstance(text, str):
            text = [text]
        elif isinstance(text, Translation):
            text = [text.text]
        elif isinstance(text, list) and isinstance(text[0], Translation):
            text = [t.text for t in text]

        assert isinstance(text, list) and isinstance(text[0], str), (
            f"text must be a string, a Translation, a list of strings, or a list of Translation objects. Got {text}."
        )

        # Query
        print(text)
        results = self.collection.query(
            query_texts=text,
            n_results=self.top_k,
        )
        print(results)
        processed_results = []
        for query_id, _ in enumerate(text):
            query_results = []
            for distance, metadata in zip(
                results["distances"][query_id], results["metadatas"][query_id]
            ):
                query_results.append(
                    RetrievedExpressionMetadata(distance=distance, **metadata)
                )
            processed_results.append(query_results)

        return processed_results
