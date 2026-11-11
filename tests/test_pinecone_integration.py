"""Against a real Pinecone project. Runs only when ``PINECONE_API_KEY`` is set.

It creates a small serverless index named ``aatm-test-<random>`` (in
``PINECONE_CLOUD`` / ``PINECONE_REGION``, default ``aws`` / ``us-east-1``),
writes, queries and deletes it.
"""

import os
import time
import uuid

import numpy as np
import pytest

from aatm.omop.coding import retriever_filter
from aatm.retrievers import PineconeRetriever
from aatm.vector_db import PineconeConnection
from aatm.vector_stores import (
    PineconeVectorStore,
    create_pinecone_index,
    delete_pinecone_index,
)

# Read at import: the autouse fixture in conftest.py clears it for other tests.
ENV = {
    name: os.environ[name]
    for name in ("PINECONE_API_KEY", "PINECONE_CLOUD", "PINECONE_REGION")
    if os.environ.get(name)
}

pytestmark = pytest.mark.skipif(
    "PINECONE_API_KEY" not in ENV, reason="PINECONE_API_KEY is not set"
)

DIM = 8
DOMAINS = ["Condition"] * 40 + ["Measurement"] * 20


class FixedEmbeddings:
    def __init__(self, vectors):
        self.vectors = vectors

    def __call__(self, input):
        return [self.vectors[int(text)] for text in input]


def wait_for(condition, timeout=120.0):
    # Pinecone is eventually consistent: new records take a moment to show up.
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("Pinecone did not catch up in time")
        time.sleep(2)


@pytest.fixture(scope="module")
def store():
    connection = PineconeConnection.from_env(ENV)
    client = connection.client()
    name = f"aatm-test-{uuid.uuid4().hex[:12]}"
    create_pinecone_index(
        client,
        name,
        DIM,
        connection.cloud,
        connection.region,
        tags={"aatm_embedding_dtype": "float32"},
    )
    try:
        yield PineconeVectorStore.open(client, name)
    finally:
        delete_pinecone_index(client, name)


@pytest.fixture(scope="module")
def vectors(store):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(len(DOMAINS), DIM)).astype(np.float32)
    ids = [f"id-{i}" for i in range(len(DOMAINS))]
    metadatas = [
        {
            "expression": f"expression {i}",
            "std_concept_id": str(i),
            "std_domain_id": domain,
            "std_vocabulary_id": "LOINC" if i % 2 else "SNOMED",
            "target_vocabulary_id": None,
        }
        for i, domain in enumerate(DOMAINS)
    ]
    store.write(ids, vectors, [m["expression"] for m in metadatas], metadatas)
    wait_for(lambda: store.count() == len(DOMAINS))
    return vectors


def test_records_are_found_and_counted(store, vectors):
    wait_for(lambda: len(store.existing_ids(["id-0", "id-59", "missing"])) == 2)
    assert store.existing_ids(["id-0", "id-59", "missing"]) == {"id-0", "id-59"}
    assert store.index_metadata() == {"aatm_embedding_dtype": "float32"}


def test_distances_are_squared_euclidean(store, vectors):
    retriever = PineconeRetriever(store, FixedEmbeddings(vectors), top_k=5)

    (results,) = retriever.retrieve(["3"]).results

    expected = sorted(float(np.sum((v - vectors[3]) ** 2)) for v in vectors)[:5]
    assert [r.distance for r in results] == pytest.approx(expected, abs=1e-3)
    assert results[0].std_concept_id == "3"


def test_filters_are_applied_during_the_search(store, vectors):
    retriever = PineconeRetriever(store, FixedEmbeddings(vectors), top_k=5)
    where = retriever_filter("Measurement", ["LOINC"])

    (results,) = retriever.retrieve(["3"], where=where).results

    matching = [i for i, d in enumerate(DOMAINS) if d == "Measurement" and i % 2 == 1]
    expected = sorted(
        matching, key=lambda i: float(np.sum((vectors[i] - vectors[3]) ** 2))
    )[:5]
    assert [r.std_concept_id for r in results] == [str(i) for i in expected]
    assert all(r.std_domain_id == "Measurement" for r in results)
