"""Tests for `aatm.retrievers`: the Python filter and the filter strategies."""

import hashlib

import chromadb
import numpy as np
import pytest
from chromadb import EmbeddingFunction

from aatm.omop.coding import retriever_filter
from aatm.retrievers import (
    ChromaDBRetriever,
    UnsupportedFilterError,
    is_supported_filter,
    matches_filter,
)

DIM = 8
COLLECTION = "expressions"


# --- Python filter evaluation -------------------------------------------------

META = {
    "std_domain_id": "Measurement",
    "std_vocabulary_id": "LOINC",
    "flag": True,
    "n": 1,
}


@pytest.mark.parametrize(
    "where, expected",
    [
        ({"std_domain_id": "Measurement"}, True),
        ({"std_domain_id": "Condition"}, False),
        ({"std_domain_id": {"$eq": "Measurement"}}, True),
        ({"std_domain_id": {"$eq": "Condition"}}, False),
        ({"std_vocabulary_id": {"$in": ["SNOMED", "LOINC"]}}, True),
        ({"std_vocabulary_id": {"$in": ["SNOMED"]}}, False),
        ({"std_vocabulary_id": {"$in": []}}, False),
        (
            {
                "$and": [
                    {"std_domain_id": "Measurement"},
                    {"std_vocabulary_id": "LOINC"},
                ]
            },
            True,
        ),
        (
            {
                "$and": [
                    {"std_domain_id": "Measurement"},
                    {"std_vocabulary_id": "SNOMED"},
                ]
            },
            False,
        ),
        (
            {"$or": [{"std_domain_id": "Condition"}, {"std_vocabulary_id": "LOINC"}]},
            True,
        ),
        (
            {"$or": [{"std_domain_id": "Condition"}, {"std_vocabulary_id": "SNOMED"}]},
            False,
        ),
        ({"missing_field": "x"}, False),
        ({"missing_field": {"$in": ["x"]}}, False),
        # Booleans and numbers are different values, as in ChromaDB.
        ({"flag": True}, True),
        ({"flag": 1}, False),
        ({"n": True}, False),
        ({"n": 1}, True),
    ],
)
def test_matches_filter(where, expected):
    assert is_supported_filter(where)
    assert matches_filter(META, where) is expected


@pytest.mark.parametrize(
    "domain, vocabularies",
    [("Measurement", None), (None, ["LOINC"]), ("Measurement", ["LOINC", "SNOMED"])],
)
def test_retriever_filter_shapes_are_supported(domain, vocabularies):
    assert is_supported_filter(retriever_filter(domain, vocabularies))


@pytest.mark.parametrize(
    "where",
    [
        {},
        {"a": "x", "b": "y"},  # ChromaDB wants one key per dict
        {"a": {"$ne": "x"}},
        {"a": {"$gt": 1}},
        {"a": {"$nin": ["x"]}},
        {"a": {"$eq": "x", "$in": ["x"]}},
        {"a": {"$in": "x"}},
        {"a": ["x"]},
        {"$and": []},
        {"$and": [{"a": {"$ne": "x"}}]},
        {"$not": [{"a": "x"}]},
        "a",
    ],
)
def test_unsupported_filters_raise(where):
    assert not is_supported_filter(where)
    with pytest.raises(UnsupportedFilterError):
        matches_filter(META, where)


# --- Strategies against a real ChromaDB collection ----------------------------


def text_vector(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
    return np.random.default_rng(seed).normal(size=DIM).astype(np.float32)


class HashEmbeddingFunction(EmbeddingFunction):
    """Deterministic embeddings, so tests do not load a model."""

    def __init__(self) -> None:
        pass

    def __call__(self, input):
        return [text_vector(text) for text in input]


# 300 records: mostly Condition, some Measurement, three Device.
DOMAINS = ["Condition"] * 267 + ["Measurement"] * 30 + ["Device"] * 3


def record_metadata(i: int, domain: str) -> dict:
    return {
        "expression": f"expression {i}",
        "expression_concept_id": str(i),
        "expression_origin": "standard_concept",
        "std_concept_id": str(i),
        "std_concept_name": f"concept {i}",
        "std_vocabulary_id": "LOINC" if domain == "Measurement" and i % 2 else "SNOMED",
        "std_vocabulary_code": str(i),
        "std_domain_id": domain,
    }


class QuerySpy:
    """Wraps a collection and records every `query` call's arguments."""

    def __init__(self, collection):
        self._collection = collection
        self.calls: list[dict] = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return self._collection.query(**kwargs)

    def __getattr__(self, name):
        return getattr(self._collection, name)


@pytest.fixture
def client(tmp_path):
    client = chromadb.PersistentClient(str(tmp_path / "chroma"))
    collection = client.create_collection(
        COLLECTION, embedding_function=HashEmbeddingFunction()
    )
    rng = np.random.default_rng(0)
    collection.add(
        ids=[str(i) for i in range(len(DOMAINS))],
        embeddings=rng.normal(size=(len(DOMAINS), DIM)).astype(np.float32),
        metadatas=[record_metadata(i, d) for i, d in enumerate(DOMAINS)],
        documents=[f"expression {i}" for i in range(len(DOMAINS))],
    )
    return client


def make_retriever(client, **kwargs) -> ChromaDBRetriever:
    # Factor 1: the tests set the first over-fetch size through min_overfetch.
    kwargs.setdefault("overfetch_factor", 1)
    retriever = ChromaDBRetriever(
        client=client,
        collection_name=COLLECTION,
        embedding_function=HashEmbeddingFunction(),
        top_k=5,
        **kwargs,
    )
    retriever.collection = QuerySpy(retriever.collection)
    return retriever


def concept_ids(results) -> list[list[str]]:
    return [[r.std_concept_id for r in query] for query in results.results]


QUERIES = [f"query {i}" for i in range(8)]


@pytest.mark.parametrize(
    "where",
    [
        {"std_domain_id": "Condition"},
        {"std_domain_id": "Measurement"},
        retriever_filter("Measurement", ["LOINC"]),
    ],
)
def test_overfetch_matches_chroma_filtered_query(client, where):
    chroma = make_retriever(client, filter_strategy="chroma")
    overfetch = make_retriever(client, min_overfetch=20, max_overfetch=300)

    expected = chroma.retrieve(QUERIES, where=where)
    got = overfetch.retrieve(QUERIES, where=where)

    assert concept_ids(got) == concept_ids(expected)
    for got_query, expected_query in zip(got.results, expected.results):
        assert [r.distance for r in got_query] == pytest.approx(
            [r.distance for r in expected_query]
        )
    assert all(len(query) == 5 for query in got.results)


def test_enough_matches_skip_the_filtered_query(client):
    retriever = make_retriever(client, min_overfetch=100, max_overfetch=100)

    retriever.retrieve(QUERIES, where={"std_domain_id": "Condition"})

    assert len(retriever.collection.calls) == 1
    (call,) = retriever.collection.calls
    assert "where" not in call
    assert call["n_results"] == 100
    assert len(call["query_embeddings"]) == len(QUERIES)


def test_rare_filter_falls_back_to_the_filtered_query(client):
    retriever = make_retriever(client, min_overfetch=20, max_overfetch=40)
    where = {"std_domain_id": "Device"}

    results = retriever.retrieve(["query 0"], where=where)

    sizes = [call["n_results"] for call in retriever.collection.calls]
    assert sizes == [20, 40, 5]
    assert retriever.collection.calls[-1]["where"] == where
    # Only three Device records exist; the fallback finds all of them.
    assert sorted(concept_ids(results)[0]) == ["297", "298", "299"]


def test_fallback_runs_only_for_the_queries_left_short(client):
    retriever = make_retriever(client, min_overfetch=20, max_overfetch=20)
    where = {"std_domain_id": "Measurement"}
    # Find one query whose top 20 holds at least 5 Measurement records and one
    # whose top 20 does not.
    chroma = make_retriever(client, filter_strategy="chroma")
    unfiltered = chroma.retrieve([f"q{i}" for i in range(200)], top_k=20)
    counts = {
        q: sum(r.std_domain_id == "Measurement" for r in rs)
        for q, rs in zip(unfiltered.queries, unfiltered.results)
    }
    rich = next(q for q, c in counts.items() if c >= 5)
    poor = next(q for q, c in counts.items() if c < 5)

    results = retriever.retrieve([rich, poor], where=where)

    fallback_calls = [c for c in retriever.collection.calls if "where" in c]
    assert len(fallback_calls) == 1
    assert len(fallback_calls[0]["query_embeddings"]) == 1
    expected = chroma.retrieve([rich, poor], where=where)
    assert concept_ids(results) == concept_ids(expected)


def test_overfetch_without_fallback_returns_what_it_found(client):
    retriever = make_retriever(
        client, filter_strategy="overfetch", min_overfetch=20, max_overfetch=40
    )

    results = retriever.retrieve(["query 0"], where={"std_domain_id": "Device"})

    assert all("where" not in call for call in retriever.collection.calls)
    assert len(results.results[0]) < 5
    assert all(r.std_domain_id == "Device" for r in results.results[0])


def test_no_fallback_once_the_whole_collection_was_seen(client):
    retriever = make_retriever(client, min_overfetch=20, max_overfetch=1000)

    results = retriever.retrieve(["query 0"], where={"std_domain_id": "Device"})

    assert [call["n_results"] for call in retriever.collection.calls] == [20, 1000]
    assert sorted(concept_ids(results)[0]) == ["297", "298", "299"]


def test_chroma_strategy_passes_the_filter_to_chromadb(client):
    retriever = make_retriever(client, filter_strategy="chroma")
    where = {"std_domain_id": "Measurement"}

    retriever.retrieve(["query 0"], where=where)

    (call,) = retriever.collection.calls
    assert call["where"] == where
    assert call["n_results"] == 5


def test_strategy_can_be_overridden_per_call(client):
    retriever = make_retriever(client, filter_strategy="chroma")

    retriever.retrieve(
        ["query 0"], where={"std_domain_id": "Condition"}, filter_strategy="overfetch"
    )

    assert all("where" not in call for call in retriever.collection.calls)


def test_default_where_is_used(client):
    retriever = make_retriever(client, where={"std_domain_id": "Measurement"})

    results = retriever.retrieve(QUERIES)

    assert all(r.std_domain_id == "Measurement" for q in results.results for r in q)


@pytest.mark.parametrize("where", [None, {}])
def test_no_filter_is_a_single_unfiltered_query(client, where):
    retriever = make_retriever(client)

    results = retriever.retrieve(["query 0"], where=where)

    (call,) = retriever.collection.calls
    assert "where" not in call
    assert call["n_results"] == 5
    assert len(results.results[0]) == 5


def test_unsupported_filter_goes_to_chromadb_by_default(client):
    retriever = make_retriever(client)
    where = {"std_domain_id": {"$ne": "Condition"}}

    results = retriever.retrieve(["query 0"], where=where)

    (call,) = retriever.collection.calls
    assert call["where"] == where
    assert all(r.std_domain_id != "Condition" for r in results.results[0])


def test_unsupported_filter_raises_without_fallback(client):
    retriever = make_retriever(client, filter_strategy="overfetch")

    with pytest.raises(UnsupportedFilterError):
        retriever.retrieve(["query 0"], where={"std_domain_id": {"$ne": "Condition"}})


def test_unknown_strategy_raises(client):
    with pytest.raises(ValueError, match="filter_strategy"):
        make_retriever(client, filter_strategy="post")
    retriever = make_retriever(client)
    with pytest.raises(ValueError, match="filter_strategy"):
        retriever.retrieve(["q"], where={"a": "b"}, filter_strategy="post")


def test_overfetch_sizes(client):
    retriever = make_retriever(
        client, overfetch_factor=20, min_overfetch=200, max_overfetch=2000
    )
    assert retriever.overfetch_sizes(5) == [200, 2000]
    assert retriever.overfetch_sizes(50) == [1000, 2000]
    assert retriever.overfetch_sizes(100) == [2000]
    assert retriever.overfetch_sizes(500) == [10000]


def test_default_overfetch_sizes(client):
    retriever = ChromaDBRetriever(
        client=client,
        collection_name=COLLECTION,
        embedding_function=HashEmbeddingFunction(),
    )
    assert retriever.filter_strategy == "overfetch_fallback"
    assert retriever.overfetch_sizes(10) == [1000, 5000]
    assert retriever.overfetch_sizes(100) == [2000, 5000]
    assert retriever.overfetch_sizes(300) == [6000]


# --- Many queries in one call ----------------------------------------------------


def test_filtered_query_returning_more_records_than_sqlite_allows(client):
    """A local ChromaDB fails past ~32,766 returned records in one query call.

    200 queries of every Condition record (267) return 53,400 records, which
    used to raise ``too many SQL variables``.
    """
    retriever = make_retriever(client, filter_strategy="native")
    where = {"std_domain_id": "Condition"}
    queries = [f"lab test {i}" for i in range(200)]

    results = retriever.retrieve(queries, where=where, top_k=300)

    assert len(results.results) == len(queries)
    assert all(len(query) == 267 for query in results.results)
    assert all(r.std_domain_id == "Condition" for q in results.results for r in q)


def test_overfetch_fallback_with_many_queries(client):
    # The default strategy's fallback is a filtered query of every pending text.
    retriever = make_retriever(client, min_overfetch=300, max_overfetch=300)
    queries = [f"lab test {i}" for i in range(200)]

    results = retriever.retrieve(queries, where={"std_domain_id": "Device"}, top_k=5)

    assert all(len(query) == 3 for query in results.results)


def test_queries_are_split_into_calls_and_keep_their_order(client):
    retriever = make_retriever(client, filter_strategy="native")
    retriever.store.max_query_results = 12  # 12 // top_k 5 = 2 queries per call
    where = {"std_domain_id": "Measurement"}
    queries = [f"query {i}" for i in range(7)]

    got = retriever.retrieve(queries, where=where)

    calls = retriever.collection.calls
    assert [len(call["query_texts"]) for call in calls] == [2, 2, 2, 1]
    assert all(call["where"] == where for call in calls)
    expected = [concept_ids(retriever.retrieve([q], where=where))[0] for q in queries]
    assert concept_ids(got) == expected


def test_max_query_results_must_be_positive(client):
    from aatm.vector_stores import ChromaVectorStore

    with pytest.raises(ValueError, match="max_query_results"):
        ChromaVectorStore(client.get_collection(COLLECTION), max_query_results=0)
