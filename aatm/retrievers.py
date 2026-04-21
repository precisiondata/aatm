"""
Define retriever abstractions and ChromaDB-based retrieval implementations.

This module provides the base interface for retrieval components used in the
pipeline, along with a concrete retriever backed by ChromaDB. Retrievers are
responsible for accepting one or more queries, fetching candidate expressions,
and returning them in a standardized ``RetrieverResults`` structure.

The module is designed to support pipeline-style composition, allowing
retrievers to be chained with other components such as rerankers and selectors.
"""

from abc import ABC, abstractmethod
from typing import Any, List
import chromadb
from chromadb import EmbeddingFunction

from .data_models import RetrievedExpressionMetadata, RetrieverResults, Translation
from .pipeline import PipelineBaseClass


class BaseRetriever(PipelineBaseClass, ABC):
    """Define the abstract interface for retrieval pipeline components.

    This base class establishes the contract for retrievers that accept one or
    more queries and return structured retrieval results. It also provides a
    flexible ``__call__()`` implementation that normalizes several supported
    query input types before delegating to ``retrieve()``.
    """

    @abstractmethod
    def retrieve(self, queries: List[str]) -> RetrieverResults:
        """Retrieve candidate results for one or more query strings.

        Subclasses must implement this method to perform the actual retrieval
        operation and return results in a ``RetrieverResults`` object.

        Args:
            queries: A list of query strings to retrieve candidates for.

        Returns:
            A ``RetrieverResults`` instance containing the retrieved candidates
                for each query.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        pass

    def __call__(
        self,
        queries: str | Translation | List[str] | List[Translation],
        *args: Any,
        **kwargs: Any,
    ) -> RetrieverResults:
        """Normalize query inputs and perform retrieval.

        This method allows retriever instances to be called directly with a
        single string, a single ``Translation`` object, a list of strings, or a
        list of ``Translation`` objects. Supported inputs are converted into a
        list of query strings before being passed to ``retrieve()``.

        Args:
            queries: Query input to retrieve against. Supported values are a
                single string, a single ``Translation`` object, a list of
                strings, or a list of ``Translation`` objects.

        Returns:
            A ``RetrieverResults`` instance containing the retrieved candidates
                for each normalized query string.

        Raises:
            AssertionError: If the input is not one of the supported query
                formats or does not contain valid string queries.
        """
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

        return self.retrieve(queries, *args, **kwargs)


class ChromaDBRetriever(BaseRetriever):
    """Retrieve candidate expressions from a ChromaDB collection.

    This retriever uses a ChromaDB client and collection to perform vector-based
    retrieval over stored expressions. Retrieved metadata and distances are
    converted into ``RetrievedExpressionMetadata`` objects and returned in a
    standardized ``RetrieverResults`` container.

    The retriever supports default filtering through a ``where`` clause and a
    configurable default number of results per query.
    """

    def __init__(
        self,
        client: chromadb.ClientAPI,
        collection_name: str,
        embedding_function: EmbeddingFunction,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Initialize the ChromaDB retriever.

        This constructor stores the ChromaDB client configuration, creates or
        retrieves the target collection, and sets default retrieval parameters
        such as the number of results to return and any metadata filter to
        apply.

        Args:
            client: ChromaDB client used to access the vector database.
            collection_name: Name of the ChromaDB collection to query.
            embedding_function: Embedding function used by the collection for
                query encoding.
            top_k: Default number of candidate results to return per query.
            where: Optional metadata filter applied to all queries unless
                overridden at retrieval time.
            *args: Additional positional arguments reserved for compatibility.
            **kwargs: Additional keyword arguments reserved for compatibility.

        Returns:
            None.
        """
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
        *args,
        **kwargs,
    ) -> RetrieverResults:
        """Retrieve nearest candidates for the given queries from ChromaDB.

        This method submits the provided query strings to the configured ChromaDB
        collection, optionally overriding the default metadata filter and number
        of returned results. It then converts the raw ChromaDB response into
        ``RetrievedExpressionMetadata`` objects grouped by query and wraps them
        in a ``RetrieverResults`` instance.

        Args:
            queries: List of query strings to search for in the collection.
            where: Optional metadata filter to apply to this retrieval call. If
                not provided, the retriever's default filter is used.
            top_k: Optional number of results to return per query. If not
                provided, the retriever's default ``top_k`` value is used.

        Returns:
            A ``RetrieverResults`` instance containing processed retrieval
                results for each query, including distances and metadata.

        Raises:
            Exception: Propagates errors raised by the underlying ChromaDB
                client or collection query operation.
        """
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
