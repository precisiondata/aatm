"""
Define retriever abstractions and vector-database-backed retrieval implementations.

This module provides the base interface for retrieval components used in the
pipeline, along with retrievers backed by a vector database: `ChromaDBRetriever`
(a local or Chroma Cloud collection) and `PineconeRetriever` (a Pinecone
index). Both share their filtering logic through `VectorStoreRetriever`.
Retrievers are responsible for accepting one or more queries, fetching
candidate expressions, and returning them in a standardized
``RetrieverResults`` structure.

The module is designed to support pipeline-style composition, allowing
retrievers to be chained with other components such as rerankers and selectors.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Literal, Mapping, Sequence, get_args
import chromadb

from .data_models import RetrievedExpressionMetadata, RetrieverResults, Translation
from .pipeline import PipelineBaseClass
from .vector_stores import Hit as _Hit
from .vector_stores import (
    ChromaVectorStore,
    UnsupportedFilterError,
    VectorStore,
    chroma_query,
)

FilterStrategy = Literal["overfetch_fallback", "overfetch", "native", "chroma"]
"""How a `VectorStoreRetriever` applies a `where` metadata filter.

- ``"overfetch_fallback"`` (default for ChromaDB): query without the filter
  for more results than needed, keep the ones that match the filter, and run
  the database's filtered query for the queries where fewer than ``top_k``
  matches survive.
- ``"overfetch"``: the same over-fetch, without the fallback. A query may then
  return fewer than ``top_k`` results.
- ``"native"`` (default for Pinecone): pass the filter to the vector database.
  ChromaDB applies it before the vector search, which is exact but slow on
  large collections when many records match; Pinecone applies it during the
  search, so the over-fetch is not needed there.
- ``"chroma"``: the same as ``"native"``, under its original name.
"""

FILTER_STRATEGIES: tuple[str, ...] = get_args(FilterStrategy)

EmbeddingFunction = Callable[..., Any]
"""Turns a list of texts into one vector each: ``embedding_function(input=texts)``.

Any of `aatm.embedding_functions` works, as does any ``chromadb.EmbeddingFunction``.
An ``embed_query`` method, when present, is used for queries.
"""


def _values_equal(a: Any, b: Any) -> bool:
    # Python treats True == 1; ChromaDB keeps booleans and numbers apart.
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def is_supported_filter(where: Any) -> bool:
    """Whether `matches_filter` can evaluate a ChromaDB `where` filter.

    Supported: field equality (``{"field": value}`` or
    ``{"field": {"$eq": value}}``), ``{"field": {"$in": [...]}}``, and
    ``$and`` / ``$or`` over supported filters. Like ChromaDB, each dict holds
    exactly one key.
    """
    if not isinstance(where, Mapping) or len(where) != 1:
        return False
    ((key, value),) = where.items()
    if key in ("$and", "$or"):
        return (
            isinstance(value, Sequence)
            and not isinstance(value, str)
            and len(value) > 0
            and all(is_supported_filter(clause) for clause in value)
        )
    if key.startswith("$"):
        return False
    if _is_scalar(value):
        return True
    if isinstance(value, Mapping) and len(value) == 1:
        ((op, operand),) = value.items()
        if op == "$eq":
            return _is_scalar(operand)
        if op == "$in":
            return (
                isinstance(operand, Sequence)
                and not isinstance(operand, str)
                and all(_is_scalar(item) for item in operand)
            )
    return False


def matches_filter(
    metadata: Mapping[str, Any] | None, where: Mapping[str, Any]
) -> bool:
    """Evaluate a ChromaDB `where` filter on one record's metadata.

    A record that lacks a field never matches a condition on that field.

    Raises:
        UnsupportedFilterError: If the filter is not one of the shapes listed
            in `is_supported_filter`.
    """
    if not is_supported_filter(where):
        raise UnsupportedFilterError(f"Cannot evaluate filter in Python: {where!r}")
    return _matches(metadata or {}, where)


def _matches(metadata: Mapping[str, Any], where: Mapping[str, Any]) -> bool:
    ((key, value),) = where.items()
    if key == "$and":
        return all(_matches(metadata, clause) for clause in value)
    if key == "$or":
        return any(_matches(metadata, clause) for clause in value)
    if key not in metadata:
        return False
    field_value = metadata[key]
    if isinstance(value, Mapping):
        ((op, operand),) = value.items()
        if op == "$in":
            return any(_values_equal(field_value, item) for item in operand)
        value = operand  # $eq
    return _values_equal(field_value, value)


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

    def __call__(  # type: ignore[override]
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


class VectorStoreRetriever(BaseRetriever):
    """Retrieve candidate expressions from a vector database.

    Queries are embedded with ``embedding_function`` and sent to a
    `aatm.vector_stores.VectorStore`. Retrieved metadata and distances are
    converted into ``RetrievedExpressionMetadata`` objects and returned in a
    standardized ``RetrieverResults`` container. Distances mean the same on
    every backend (see `aatm.vector_stores`), so selectors and thresholds do
    not depend on it.

    The retriever supports default filtering through a ``where`` clause and a
    configurable default number of results per query.

    ``filter_strategy`` chooses how a ``where`` filter is applied (see
    `FilterStrategy`). With ``"overfetch_fallback"``, the retriever first
    queries without the filter for ``max(top_k * overfetch_factor,
    min_overfetch)`` results, then, for the queries still short of ``top_k``
    matches, for ``max_overfetch`` results, and only then falls back to the
    database's filtered query. With an exact search, the ``top_k`` matches that
    survive would be exactly the filtered query's results. Vector indexes are
    approximate, so in practice both methods miss some neighbours, in different
    places: the over-fetch misses matches that sit deep in the unfiltered
    ranking (a filter that fits the query poorly), while ChromaDB's filtered
    search can miss even close matches. Raise ``min_overfetch`` /
    ``max_overfetch`` for deeper filters. Query sizes are capped at the
    store's ``max_top_k``.
    """

    def __init__(
        self,
        store: VectorStore,
        embedding_function: EmbeddingFunction,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        filter_strategy: FilterStrategy | None = None,
        overfetch_factor: int = 20,
        min_overfetch: int = 1000,
        max_overfetch: int = 5000,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the retriever.

        Args:
            store: The vector database index to query.
            embedding_function: Embeds the queries, with the model the index
                was built with.
            top_k: Default number of candidate results to return per query.
            where: Optional metadata filter applied to all queries unless
                overridden at retrieval time.
            filter_strategy: How a ``where`` filter is applied. See
                `FilterStrategy`. Defaults to ``"native"`` for a store that
                filters during the search (Pinecone), and to
                ``"overfetch_fallback"`` otherwise (ChromaDB).
            overfetch_factor: The first unfiltered query fetches
                ``top_k * overfetch_factor`` results (at least
                ``min_overfetch``).
            min_overfetch: Lower bound on the first unfiltered query's size.
            max_overfetch: Size of the second, wider unfiltered query, run for
                the queries the first one left short of ``top_k`` matches.
                Skipped when it is not larger than the first query.
            *args: Additional positional arguments reserved for compatibility.
            **kwargs: Additional keyword arguments reserved for compatibility.
        """
        self.store = store
        self.embedding_function = embedding_function
        if filter_strategy is None:
            filter_strategy = (
                "native" if store.filters_during_search else "overfetch_fallback"
            )
        self._configure(
            top_k,
            where,
            filter_strategy,
            overfetch_factor,
            min_overfetch,
            max_overfetch,
        )

    def _configure(
        self,
        top_k: int,
        where: dict[str, Any] | None,
        filter_strategy: FilterStrategy,
        overfetch_factor: int,
        min_overfetch: int,
        max_overfetch: int,
    ) -> None:
        self.top_k = top_k
        self.where = where
        self.filter_strategy = _check_strategy(filter_strategy)
        if overfetch_factor < 1 or min_overfetch < 1 or max_overfetch < 1:
            raise ValueError(
                "overfetch_factor, min_overfetch and max_overfetch must be >= 1."
            )
        self.overfetch_factor = overfetch_factor
        self.min_overfetch = min_overfetch
        self.max_overfetch = max_overfetch

    @property
    def max_query_size(self) -> int | None:
        """Largest number of results one query may ask for, or ``None``."""
        return self.store.max_top_k

    def content_fingerprint(self) -> dict[str, Any] | None:
        """Identify the index content, for the coding cache (`aatm.omop.cache`)."""
        return self.store.fingerprint()

    def retrieve(
        self,
        queries: List[str],
        where: dict[str, Any] | None = None,
        top_k: int | None = None,
        filter_strategy: FilterStrategy | None = None,
        *args,
        **kwargs,
    ) -> RetrieverResults:
        """Retrieve nearest candidates for the given queries.

        This method submits the provided query strings to the vector database,
        optionally overriding the default metadata filter and number of
        returned results. It then converts the raw response into
        ``RetrievedExpressionMetadata`` objects grouped by query and wraps them
        in a ``RetrieverResults`` instance.

        Args:
            queries: List of query strings to search for.
            where: Optional metadata filter to apply to this retrieval call. If
                not provided, the retriever's default filter is used.
            top_k: Optional number of results to return per query. If not
                provided, the retriever's default ``top_k`` value is used.
            filter_strategy: Optional override of the retriever's
                ``filter_strategy`` for this call. Ignored without a filter.

        Returns:
            A ``RetrieverResults`` instance containing processed retrieval
                results for each query, including distances and metadata.

        Raises:
            UnsupportedFilterError: With ``filter_strategy="overfetch"``, if the
                filter has a shape `is_supported_filter` rejects. The default
                strategy hands such filters to the database instead.
            Exception: Propagates errors raised by the underlying vector
                database client.
        """
        n_results = top_k if top_k is not None else self.top_k
        where = where if where is not None else self.where
        strategy = _check_strategy(
            filter_strategy if filter_strategy is not None else self.filter_strategy
        )

        hits: list[list[_Hit]]
        if not where:
            hits = self._query(query_texts=queries, n_results=n_results)
        elif strategy in ("native", "chroma") or (
            strategy == "overfetch_fallback" and not is_supported_filter(where)
        ):
            hits = self._query(query_texts=queries, n_results=n_results, where=where)
        else:
            if not is_supported_filter(where):
                raise UnsupportedFilterError(
                    f"Cannot evaluate filter in Python: {where!r}. Use "
                    "filter_strategy='native' or 'overfetch_fallback' for it."
                )
            hits = self._overfetch(
                queries,
                where,
                n_results,
                fallback=strategy == "overfetch_fallback",
            )

        processed_results = [
            [self._to_result(distance, metadata) for distance, metadata in query_hits]
            for query_hits in hits
        ]

        return RetrieverResults(
            results=processed_results,
            queries=queries,
        )

    def _to_result(
        self, distance: float, metadata: dict[str, Any]
    ) -> RetrievedExpressionMetadata:
        """Turn one retrieved record into a result."""
        return RetrievedExpressionMetadata(distance=distance, **metadata)  # type: ignore[arg-type]

    def overfetch_sizes(self, top_k: int) -> list[int]:
        """The sizes of the unfiltered queries the over-fetch runs, in order."""
        first = max(top_k * self.overfetch_factor, self.min_overfetch)
        second = self.max_overfetch
        cap = self.max_query_size
        if cap is not None:
            first, second = min(first, cap), min(second, cap)
        return [first, second] if second > first else [first]

    def _overfetch(
        self,
        queries: List[str],
        where: dict[str, Any],
        top_k: int,
        fallback: bool,
    ) -> list[list[_Hit]]:
        """Query without the filter, filter in Python, widen, then fall back."""
        # Embed once: the wider query and the fallback reuse the embeddings.
        embeddings = list(self._embed_queries(queries))
        hits: list[list[_Hit]] = [[] for _ in queries]
        pending = list(range(len(queries)))

        for size in self.overfetch_sizes(top_k):
            batch = self._query(
                query_embeddings=[embeddings[i] for i in pending], n_results=size
            )
            still_short = []
            for i, query_hits in zip(pending, batch):
                matching = [hit for hit in query_hits if _matches(hit[1], where)]
                hits[i] = matching[:top_k]
                # Fewer results than asked for: the whole collection was seen,
                # so no wider query or fallback can find more matches.
                if len(matching) < top_k and len(query_hits) >= size:
                    still_short.append(i)
            pending = still_short
            if not pending:
                return hits

        if fallback:
            batch = self._query(
                query_embeddings=[embeddings[i] for i in pending],
                n_results=top_k,
                where=where,
            )
            for i, query_hits in zip(pending, batch):
                hits[i] = query_hits
        return hits

    def _embed_queries(self, queries: List[str]) -> Any:
        """Embed query texts, with ``embed_query`` when the function has one."""
        embed_query = getattr(self.embedding_function, "embed_query", None)
        if callable(embed_query):
            return embed_query(input=queries)
        return self.embedding_function(input=queries)

    def _query(
        self,
        n_results: int,
        query_texts: List[str] | None = None,
        query_embeddings: Sequence[Any] | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[list[_Hit]]:
        """Run one query per text or embedding and return its (distance, metadata)."""
        if query_embeddings is None:
            assert query_texts is not None, "query_texts or query_embeddings needed"
            query_embeddings = list(self._embed_queries(query_texts))
        if self.max_query_size is not None:
            n_results = min(n_results, self.max_query_size)
        return self.store.query(query_embeddings, top_k=n_results, where=where)


class ChromaDBRetriever(VectorStoreRetriever):
    """Retrieve candidate expressions from a ChromaDB collection.

    This retriever uses a ChromaDB client and collection, local or in Chroma
    Cloud, to perform vector-based retrieval over stored expressions. See
    `VectorStoreRetriever` for the filter strategies.

    A ``where`` filter makes ChromaDB's own query slow on large collections:
    ChromaDB fetches every record that matches the filter before the vector
    search. That is why the default ``filter_strategy`` here is
    ``"overfetch_fallback"``.
    """

    store: ChromaVectorStore

    def __init__(
        self,
        client: chromadb.ClientAPI,
        collection_name: str,
        embedding_function: EmbeddingFunction,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        filter_strategy: FilterStrategy = "overfetch_fallback",
        overfetch_factor: int = 20,
        min_overfetch: int = 1000,
        max_overfetch: int = 5000,
        *args: Any,
        **kwargs: Any,
    ):
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
            filter_strategy: How a ``where`` filter is applied. One of
                ``"overfetch_fallback"`` (default), ``"overfetch"`` or
                ``"native"`` (also ``"chroma"``). See `FilterStrategy`.
            overfetch_factor: The first unfiltered query fetches
                ``top_k * overfetch_factor`` results (at least
                ``min_overfetch``).
            min_overfetch: Lower bound on the first unfiltered query's size.
            max_overfetch: Size of the second, wider unfiltered query, run for
                the queries the first one left short of ``top_k`` matches.
                Skipped when it is not larger than the first query.
            *args: Additional positional arguments reserved for compatibility.
            **kwargs: Additional keyword arguments reserved for compatibility.
        """
        self.client = client
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.collection = self.client.get_or_create_collection(
            collection_name,
            embedding_function=embedding_function,  # type: ignore[arg-type]
        )
        self.store = ChromaVectorStore(self.collection)
        self._configure(
            top_k,
            where,
            filter_strategy,
            overfetch_factor,
            min_overfetch,
            max_overfetch,
        )

    def content_fingerprint(self) -> dict[str, Any] | None:
        """Identify the collection content, for the coding cache."""
        return {
            "collection_id": str(self.collection.id),
            "count": self.collection.count(),
        }

    def _query(self, n_results: int, **kwargs: Any) -> list[list[_Hit]]:  # type: ignore[override]
        """Run one ChromaDB query and return each query's (distance, metadata).

        Many queries are split into several calls; see `chroma_query`.
        """
        return chroma_query(
            self.collection,
            n_results,
            max_query_results=self.store.max_query_results,
            **kwargs,
        )


class PineconeRetriever(VectorStoreRetriever):
    """Retrieve candidate expressions from a Pinecone index.

    Pinecone applies metadata filters during the vector search, so the default
    ``filter_strategy`` is ``"native"``: domain and vocabulary filters go
    straight to Pinecone, without the over-fetch ChromaDB needs. The over-fetch
    strategies still work, with query sizes capped at Pinecone's maximum
    ``top_k`` (10,000).

    Build one with `aatm.registries.retrievers.load_retriever` and
    ``AATM_VECTOR_DB=pinecone``, or from an open index:

    ```python
    from aatm.vector_db import PineconeConnection
    from aatm.vector_stores import PineconeVectorStore

    connection = PineconeConnection.from_env()
    store = PineconeVectorStore.open(connection.client(), "aatm-expressions-qwen3-06b")
    retriever = PineconeRetriever(store, embedding_function)
    ```
    """

    @property
    def index_name(self) -> str:
        """Name of the Pinecone index."""
        return self.store.name

    @property
    def collection_name(self) -> str:
        """Name of the Pinecone index, under the name `ChromaDBRetriever` uses."""
        return self.store.name

    def _to_result(
        self, distance: float, metadata: dict[str, Any]
    ) -> RetrievedExpressionMetadata:
        # Pinecone cannot store nulls, so `None` fields were left out on write.
        # Restore them: several metadata fields are optional but required.
        return RetrievedExpressionMetadata(
            distance=distance,
            **(_NULL_EXPRESSION_FIELDS | metadata),  # type: ignore[arg-type]
        )


_NULL_EXPRESSION_FIELDS: dict[str, Any] = {
    name: None
    for name in RetrievedExpressionMetadata.model_fields
    if name not in ("distance", "rerank_score")
}


def _check_strategy(strategy: str) -> FilterStrategy:
    if strategy not in FILTER_STRATEGIES:
        raise ValueError(
            f"Unknown filter_strategy {strategy!r}. "
            f"Expected one of: {', '.join(FILTER_STRATEGIES)}."
        )
    return strategy  # type: ignore[return-value]
