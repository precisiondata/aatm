"""A small interface over the vector databases AATM can store its index in.

`VectorStore` is what retrievers, the index builder and the upload command need
from a vector database, whichever it is:

- **Read**: `VectorStore.query` returns, for each query embedding, its nearest
  records as ``(distance, metadata)`` pairs, nearest first, optionally
  restricted by a metadata filter.
- **Write**: `VectorStore.existing_ids` and `VectorStore.write`, which adds a
  batch on its first attempt and upserts it on retries, so a retry or a resumed
  build never duplicates rows.
- **Index metadata**: `VectorStore.index_metadata`, e.g. the embedding
  precision the index was built with (see `aatm.precision`).

Implementations: `ChromaVectorStore` (a local or Chroma Cloud collection),
`PineconeVectorStore` (a Pinecone index), and
`aatm.pgvector_store.PgVectorStore` (a PostgreSQL table with pgvector).

**Distances mean the same everywhere.** ChromaDB collections use its default
``l2`` space, whose distance is the *squared* Euclidean distance. Pinecone
indexes are created with the ``euclidean`` metric, whose score is also the
squared Euclidean distance, lower first. So a record has the same distance on
both backends, and ``1 - distance`` (the confidence score), selectors and
thresholds work unchanged.

**Filters** use one backend-neutral format, the ChromaDB ``where`` dicts that
`aatm.omop.coding.retriever_filter` builds: field equality
(``{"field": value}`` or ``{"field": {"$eq": value}}``), comparison and
membership operators (``$ne``, ``$gt``, ``$gte``, ``$lt``, ``$lte``, ``$in``,
``$nin``) and ``$and`` / ``$or``. `to_pinecone_filter` translates them for
Pinecone.

**Pinecone records** have no separate document field and no null metadata
values. `to_pinecone_metadata` drops ``None`` values and keeps the document in
the metadata (it is already there as ``expression`` for AATM's expressions).
Pinecone stores integers as floats; `from_pinecone_metadata` turns whole
floats back into integers.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import numpy as np

from aatm.vector_db import call_with_retries

Hit = tuple[float, dict[str, Any]]
"""One retrieved record: its distance and its metadata."""

T = TypeVar("T")
R = TypeVar("R")


class UnsupportedFilterError(ValueError):
    """Raised when a ``where`` filter cannot be evaluated or translated."""


class VectorStore(ABC):
    """One embedding model's index in a vector database. See the module docstring."""

    name: str
    """Name of the collection or index."""

    max_top_k: int | None = None
    """Largest number of results one query may ask for, or ``None`` for no limit."""

    filters_during_search: bool = False
    """Whether the backend applies a filter during the vector search.

    ChromaDB applies it before the search, which is slow on large collections
    (see `aatm.retrievers.FilterStrategy`); Pinecone applies it during the
    search, so the retrievers pass filters straight to it by default.
    """

    @abstractmethod
    def query(
        self,
        embeddings: Sequence[Sequence[float]],
        top_k: int,
        where: Mapping[str, Any] | None = None,
    ) -> list[list[Hit]]:
        """Return each embedding's ``top_k`` nearest records, nearest first.

        Args:
            embeddings: One query vector per query.
            top_k: Results per query. Capped at `max_top_k`.
            where: Optional metadata filter in the neutral format.

        Returns:
            For each query, ``(distance, metadata)`` pairs.
        """

    @abstractmethod
    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        """Return which of ``ids`` are stored."""

    @abstractmethod
    def write(
        self,
        ids: Sequence[str],
        embeddings: Any,
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
        attempt: int = 0,
    ) -> None:
        """Write a batch of records.

        Args:
            ids: Record IDs.
            embeddings: One vector per record, e.g. a float32 numpy array.
            documents: Each record's text.
            metadatas: Each record's metadata.
            attempt: 0 on the first try. Retries upsert instead of adding, so a
                batch that reached the server before an error is not duplicated.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored records."""

    @abstractmethod
    def index_metadata(self) -> dict[str, Any]:
        """Return the metadata recorded on the index itself, e.g. its precision."""

    def fingerprint(self) -> dict[str, Any]:
        """Identify the index content, for `aatm.omop.cache`."""
        return {"index": self.name, "count": self.count()}


# --- ChromaDB --------------------------------------------------------------------


CHROMA_MAX_QUERY_RESULTS = 30_000
"""Default cap on the records one ChromaDB query call returns, over all its queries.

A local ChromaDB fetches the metadata of every returned record in one SQLite
statement with one variable per record, so a call returning more than SQLite's
32,766-variable limit fails with ``too many SQL variables`` (e.g. 33 queries of
1,000 results, or 328 of 100). The filter does not matter, only
``len(embeddings) * top_k``. 30,000 leaves some headroom.
"""


class ChromaVectorStore(VectorStore):
    """A ChromaDB collection, local or in Chroma Cloud."""

    def __init__(
        self, collection: Any, max_query_results: int = CHROMA_MAX_QUERY_RESULTS
    ):
        """Wrap an open ``chromadb`` collection.

        Args:
            collection: The collection.
            max_query_results: Most records one ``collection.query`` call may
                return. `query` splits its embeddings into calls of at most
                ``max_query_results // top_k`` queries (at least one). See
                `CHROMA_MAX_QUERY_RESULTS`.
        """
        if max_query_results < 1:
            raise ValueError(f"max_query_results must be positive: {max_query_results}")
        self.collection = collection
        self.name = collection.name
        self.max_query_results = max_query_results

    def query(
        self,
        embeddings: Sequence[Sequence[float]],
        top_k: int,
        where: Mapping[str, Any] | None = None,
    ) -> list[list[Hit]]:
        return chroma_query(
            self.collection,
            top_k,
            query_embeddings=embeddings,
            where=where,
            max_query_results=self.max_query_results,
        )

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        return set(self.collection.get(ids=list(ids), include=[])["ids"])

    def write(
        self,
        ids: Sequence[str],
        embeddings: Any,
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
        attempt: int = 0,
    ) -> None:
        batch: dict[str, Any] = {
            "ids": list(ids),
            "embeddings": embeddings,
            "documents": list(documents),
            "metadatas": [dict(m) for m in metadatas],
        }
        # ChromaDB ignores an ``add`` for an existing ID; upsert replaces it.
        if attempt == 0:
            self.collection.add(**batch)
        else:
            self.collection.upsert(**batch)

    def count(self) -> int:
        return int(self.collection.count())

    def index_metadata(self) -> dict[str, Any]:
        return dict(self.collection.metadata or {})

    def fingerprint(self) -> dict[str, Any]:
        return {"collection_id": str(self.collection.id), "count": self.count()}


def chroma_query(
    collection: Any,
    n_results: int,
    query_texts: Sequence[str] | None = None,
    query_embeddings: Sequence[Sequence[float]] | None = None,
    where: Mapping[str, Any] | None = None,
    max_query_results: int = CHROMA_MAX_QUERY_RESULTS,
) -> list[list[Hit]]:
    """Query a ChromaDB collection, split into calls ChromaDB can answer.

    Each call holds at most ``max_query_results // n_results`` queries (at
    least one), so it returns at most about ``max_query_results`` records; see
    `CHROMA_MAX_QUERY_RESULTS`. Results keep the order of the queries.

    Args:
        collection: The collection.
        n_results: Results per query.
        query_texts: Query texts, embedded by the collection's embedding
            function. Give these or ``query_embeddings``.
        query_embeddings: Query vectors.
        where: Optional metadata filter.
        max_query_results: Most records one call may return.

    Returns:
        For each query, ``(distance, metadata)`` pairs, nearest first.
    """
    if (query_texts is None) == (query_embeddings is None):
        raise ValueError("Give exactly one of query_texts and query_embeddings")
    queries: list[Any]
    if query_texts is not None:
        key, queries = "query_texts", list(query_texts)
    else:
        assert query_embeddings is not None
        key, queries = "query_embeddings", list(query_embeddings)
    kwargs: dict[str, Any] = {"where": dict(where)} if where else {}
    batch_size = max(1, max_query_results // max(n_results, 1))
    hits: list[list[Hit]] = []
    for batch in _chunks(queries, batch_size):
        results = collection.query(
            n_results=n_results,
            include=["metadatas", "distances"],
            **{key: list(batch)},
            **kwargs,
        )
        hits.extend(chroma_hits(results))
    return hits


def chroma_hits(results: Mapping[str, Any]) -> list[list[Hit]]:
    """Convert a ChromaDB query result to ``(distance, metadata)`` pairs per query."""
    assert results["distances"] is not None, "distances must not be None"
    assert results["metadatas"] is not None, "metadatas must not be None"
    return [
        [
            (distance, dict(metadata or {}))
            for distance, metadata in zip(distances, metadatas)
        ]
        for distances, metadatas in zip(results["distances"], results["metadatas"])
    ]


# --- Pinecone: filters and metadata ----------------------------------------------

PINECONE_METRIC = "euclidean"
"""Metric of AATM's Pinecone indexes: the squared Euclidean distance, as ChromaDB's ``l2``."""

PINECONE_MAX_TOP_K = 10_000
"""Largest ``top_k`` Pinecone accepts in one query."""

PINECONE_MAX_UPSERT_RECORDS = 1_000
"""Largest number of records in one Pinecone upsert."""

PINECONE_MAX_UPSERT_BYTES = 2 * 1024 * 1024
"""Largest Pinecone upsert request, in bytes."""

PINECONE_MAX_FETCH_IDS = 1_000
"""Largest number of IDs in one Pinecone fetch."""

PINECONE_MAX_METADATA_BYTES = 40 * 1024
"""Largest metadata of one Pinecone record, in bytes."""

PINECONE_MAX_IN_VALUES = 10_000
"""Largest number of values in one ``$in`` or ``$nin`` filter."""

DOCUMENT_METADATA_KEY = "aatm_document"
"""Metadata key holding a record's text, when its ``expression`` is not that text."""

_COMPARISON_OPERATORS = ("$eq", "$ne", "$gt", "$gte", "$lt", "$lte")
_LIST_OPERATORS = ("$in", "$nin")

# JSON size of one float, as the client serializes it, at most.
_BYTES_PER_VALUE = 24
# Field names, braces and separators around each record.
_BYTES_PER_RECORD = 64
# Headroom under the request limit.
_UPSERT_SAFETY = 0.85


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def to_pinecone_filter(where: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a neutral (ChromaDB-style) ``where`` filter to Pinecone's syntax.

    Field equality becomes ``$eq``; the other operators and ``$and`` / ``$or``
    keep their names, which Pinecone shares with ChromaDB. A dict with several
    fields becomes their ``$and``.

    Raises:
        UnsupportedFilterError: For operators Pinecone does not support, e.g.
            ChromaDB's document filters, or more than 10,000 ``$in`` values.
    """
    if not isinstance(where, Mapping) or not where:
        raise UnsupportedFilterError(f"Not a filter: {where!r}")
    clauses = [_translate_clause(key, value) for key, value in where.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _translate_clause(key: str, value: Any) -> dict[str, Any]:
    if key in ("$and", "$or"):
        if isinstance(value, str) or not isinstance(value, Sequence) or not value:
            raise UnsupportedFilterError(f"{key} needs a non-empty list: {value!r}")
        return {key: [to_pinecone_filter(clause) for clause in value]}
    if key.startswith("$"):
        raise UnsupportedFilterError(f"Pinecone filters do not support {key!r}.")
    if _is_scalar(value):
        return {key: {"$eq": value}}
    if isinstance(value, Mapping) and value:
        condition: dict[str, Any] = {}
        for op, operand in value.items():
            if op in _COMPARISON_OPERATORS and _is_scalar(operand):
                condition[op] = operand
            elif op in _LIST_OPERATORS and _is_scalar_list(operand):
                if len(operand) > PINECONE_MAX_IN_VALUES:
                    raise UnsupportedFilterError(
                        f"Pinecone accepts at most {PINECONE_MAX_IN_VALUES} values "
                        f"in {op}; got {len(operand)} for {key!r}."
                    )
                condition[op] = list(operand)
            else:
                raise UnsupportedFilterError(
                    f"Pinecone filters do not support {op!r} on {key!r}: {operand!r}"
                )
        return {key: condition}
    raise UnsupportedFilterError(f"Unsupported condition on {key!r}: {value!r}")


def _is_scalar_list(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and all(_is_scalar(item) for item in value)
    )


def to_pinecone_metadata(
    metadata: Mapping[str, Any], document: str | None = None
) -> dict[str, Any]:
    """Convert a record's metadata and document to Pinecone record metadata.

    ``None`` values are dropped (Pinecone has no null). Strings, numbers,
    booleans and lists of strings are kept; other values become strings. The
    document is stored under `DOCUMENT_METADATA_KEY`, unless the metadata's
    ``expression`` already holds it.

    Raises:
        ValueError: If the metadata exceeds Pinecone's 40 KB per record.
    """
    converted: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if _is_scalar(value):
            converted[key] = value
        elif isinstance(value, Sequence) and not isinstance(value, str):
            converted[key] = [str(item) for item in value if item is not None]
        else:
            converted[key] = str(getattr(value, "value", value))
    if document is not None and converted.get("expression") != document:
        converted[DOCUMENT_METADATA_KEY] = document

    size = len(json.dumps(converted, ensure_ascii=False).encode("utf-8"))
    if size > PINECONE_MAX_METADATA_BYTES:
        raise ValueError(
            f"Record metadata is {size} bytes; Pinecone accepts at most "
            f"{PINECONE_MAX_METADATA_BYTES} bytes per record."
        )
    return converted


def from_pinecone_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Convert Pinecone record metadata back to AATM metadata.

    Whole floats become integers again, since Pinecone stores every number as a
    float. The stored document key is removed.
    """
    result: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if key == DOCUMENT_METADATA_KEY:
            continue
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        result[key] = value
    return result


def pinecone_document(metadata: Mapping[str, Any] | None) -> str | None:
    """Return the document stored in a Pinecone record's metadata."""
    metadata = metadata or {}
    document = metadata.get(DOCUMENT_METADATA_KEY, metadata.get("expression"))
    return None if document is None else str(document)


def plan_upsert_batches(
    records: Sequence[Mapping[str, Any]],
    max_records: int = PINECONE_MAX_UPSERT_RECORDS,
    max_bytes: int = PINECONE_MAX_UPSERT_BYTES,
) -> list[list[Mapping[str, Any]]]:
    """Split records into upsert requests within Pinecone's count and size limits.

    The size of each record is estimated from its dimension and its metadata's
    JSON size, with headroom under ``max_bytes``.

    Raises:
        ValueError: If one record alone is too large for a request.
    """
    budget = int(max_bytes * _UPSERT_SAFETY)
    batches: list[list[Mapping[str, Any]]] = []
    batch: list[Mapping[str, Any]] = []
    batch_bytes = 0
    for record in records:
        size = (
            _BYTES_PER_RECORD
            + len(str(record["id"]))
            + _BYTES_PER_VALUE * len(record["values"])
            + len(json.dumps(record.get("metadata") or {}, ensure_ascii=False))
        )
        if size > budget:
            raise ValueError(
                f"Record {record['id']!r} is about {size} bytes, too large for one "
                f"Pinecone upsert ({max_bytes} bytes)."
            )
        if batch and (len(batch) >= max_records or batch_bytes + size > budget):
            batches.append(batch)
            batch, batch_bytes = [], 0
        batch.append(record)
        batch_bytes += size
    if batch:
        batches.append(batch)
    return batches


def _chunks(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _parallel_map(func: Callable[[T], R], items: Sequence[T], workers: int) -> list[R]:
    """Map ``func`` over ``items`` in threads, keeping the order."""
    if workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(func, items))


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field of a Pinecone response object or of a plain dict."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


# --- Pinecone: the store ---------------------------------------------------------


class PineconeVectorStore(VectorStore):
    """A Pinecone index. Every call is retried on rate limits and transient errors.

    Pinecone answers one query vector per request, so several queries run in
    parallel threads (``query_concurrency``). Writes are split by
    `plan_upsert_batches` and use ``upsert``, which is idempotent, so retries and
    resumed builds never duplicate records.
    """

    max_top_k = PINECONE_MAX_TOP_K
    filters_during_search = True

    def __init__(
        self,
        index: Any,
        name: str,
        tags: Mapping[str, str] | None = None,
        dimension: int | None = None,
        host: str | None = None,
        query_concurrency: int = 8,
        write_concurrency: int = 4,
        fetch_batch_size: int = 200,
        retry: Callable[[Callable[[int], Any]], Any] = call_with_retries,
    ):
        """Wrap an open Pinecone index. Use `open` to open one by name.

        Args:
            index: A ``pinecone`` data-plane ``Index``.
            name: The index name.
            tags: The index's tags, which hold its AATM metadata.
            dimension: The index's vector dimension.
            host: The index host, for the fingerprint.
            query_concurrency: Queries sent at the same time.
            write_concurrency: Upsert requests sent at the same time.
            fetch_batch_size: IDs per fetch request when checking which records
                exist (at most 1,000; fetch also returns the vectors, so smaller
                requests stay light).
            retry: Retry wrapper, replaceable in tests.
        """
        if not 1 <= fetch_batch_size <= PINECONE_MAX_FETCH_IDS:
            raise ValueError(
                f"fetch_batch_size must be between 1 and {PINECONE_MAX_FETCH_IDS}."
            )
        self.index = index
        self.name = name
        self.tags = dict(tags or {})
        self.dimension = dimension
        self.host = host
        self.query_concurrency = query_concurrency
        self.write_concurrency = write_concurrency
        self.fetch_batch_size = fetch_batch_size
        self._retry = retry

    @classmethod
    def open(cls, client: Any, name: str, **kwargs: Any) -> PineconeVectorStore:
        """Open an existing index by name.

        Args:
            client: A ``pinecone.Pinecone`` client.
            name: The index name.
            **kwargs: Passed on to the constructor.
        """
        description = call_with_retries(lambda _: client.describe_index(name))
        host = _get(description, "host")
        index = client.index(name=name, host=host) if host else client.index(name)
        return cls(
            index,
            name,
            tags=_get(description, "tags") or {},
            dimension=_get(description, "dimension"),
            host=host,
            **kwargs,
        )

    def query(
        self,
        embeddings: Sequence[Sequence[float]],
        top_k: int,
        where: Mapping[str, Any] | None = None,
    ) -> list[list[Hit]]:
        top_k = min(top_k, PINECONE_MAX_TOP_K)
        pinecone_filter = to_pinecone_filter(where) if where else None
        vectors = [_as_floats(embedding) for embedding in embeddings]

        def one(vector: list[float]) -> list[Hit]:
            response = self._retry(
                lambda _: self.index.query(
                    vector=vector,
                    top_k=top_k,
                    filter=pinecone_filter,
                    include_metadata=True,
                )
            )
            return [
                (
                    float(_get(match, "score")),
                    from_pinecone_metadata(_get(match, "metadata")),
                )
                for match in _get(response, "matches") or []
            ]

        return _parallel_map(one, vectors, self.query_concurrency)

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        def one(batch: Sequence[str]) -> set[str]:
            response = self._retry(lambda _: self.index.fetch(ids=list(batch)))
            return set(_get(response, "vectors") or {})

        batches = list(_chunks(list(ids), self.fetch_batch_size))
        found: set[str] = set()
        for part in _parallel_map(one, batches, self.query_concurrency):
            found |= part
        return found

    def write(
        self,
        ids: Sequence[str],
        embeddings: Any,
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
        attempt: int = 0,
    ) -> None:
        # Upsert is idempotent, so the first attempt and retries are the same.
        vectors = np.asarray(embeddings, dtype=np.float32)
        records = [
            {
                "id": identifier,
                "values": vector.tolist(),
                "metadata": to_pinecone_metadata(metadata, document),
            }
            for identifier, vector, document, metadata in zip(
                ids, vectors, documents, metadatas
            )
        ]

        def one(batch: list[Mapping[str, Any]]) -> None:
            self._retry(lambda _: self.index.upsert(vectors=batch, show_progress=False))

        _parallel_map(one, plan_upsert_batches(records), self.write_concurrency)

    def count(self) -> int:
        stats = self._retry(lambda _: self.index.describe_index_stats())
        return int(_get(stats, "total_vector_count") or 0)

    def index_metadata(self) -> dict[str, Any]:
        return dict(self.tags)

    def fingerprint(self) -> dict[str, Any]:
        return {
            "pinecone_index": self.name,
            "host": self.host,
            "dimension": self.dimension,
            "count": self.count(),
        }


def _as_floats(embedding: Any) -> list[float]:
    return np.asarray(embedding, dtype=np.float32).astype(float).tolist()


def pinecone_index_exists(client: Any, name: str) -> bool:
    """Return whether a Pinecone index called ``name`` exists."""
    return bool(call_with_retries(lambda _: client.has_index(name)))


def create_pinecone_index(
    client: Any,
    name: str,
    dimension: int,
    cloud: str,
    region: str,
    tags: Mapping[str, str] | None = None,
) -> None:
    """Create a serverless Pinecone index for AATM and wait until it is ready.

    The index uses the `PINECONE_METRIC` (squared Euclidean distance), so its
    distances match ChromaDB's. ``tags`` record AATM's index metadata, e.g. the
    embedding precision.
    """
    from aatm.vector_db import import_pinecone

    pinecone = import_pinecone()
    client.create_index(
        name=name,
        dimension=dimension,
        metric=PINECONE_METRIC,
        spec=pinecone.ServerlessSpec(cloud=cloud, region=region),
        tags={key: str(value) for key, value in (tags or {}).items()} or None,
    )


def delete_pinecone_index(client: Any, name: str) -> None:
    """Delete a Pinecone index."""
    call_with_retries(lambda _: client.delete_index(name))
