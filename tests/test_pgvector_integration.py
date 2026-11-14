"""Against a real PostgreSQL server with pgvector. Runs only when ``AATM_TEST_PG_DSN`` is set.

Start a throwaway server with Docker, then point the tests at it:

```bash
docker run -d --name aatm-pgvector-test -e POSTGRES_HOST_AUTH_METHOD=trust \\
    -e POSTGRES_DB=aatm -p 127.0.0.1:54329:5432 pgvector/pgvector:pg17
AATM_TEST_PG_DSN=postgresql://postgres@127.0.0.1:54329/aatm uv run pytest tests/test_pgvector_integration.py
```

Every test works in tables with a random prefix and drops them afterwards.
"""

# ruff: noqa: F811 - the imported fixtures are used as test arguments

import os
import uuid

import dotenv
import numpy as np
import pytest

from aatm import local_database_utils as ldu
from aatm.data_models import RetrievedExpressionMetadata
from aatm.omop.cache import data_fingerprint
from aatm.omop.coding import retriever_filter
from aatm.registries import retrievers as registry
from aatm.retrievers import ChromaDBRetriever, PgVectorRetriever, matches_filter
from aatm.vector_db import PgVectorConnection, VectorDatabaseConfigError
from test_local_database_utils import (  # noqa: F401 - fixtures
    FakeEmbeddingFunction,
    answer_prompt,
    fake_project,
)
from test_retrievers import (  # noqa: F401 - fixture
    COLLECTION,
    DIM,
    DOMAINS,
    QUERIES,
    HashEmbeddingFunction,
    client,
    record_metadata,
)

# Read at import, like the Pinecone integration test.
DSN = (
    os.environ.get("AATM_TEST_PG_DSN")
    or dotenv.dotenv_values(".env").get("AATM_TEST_PG_DSN")
    or ""
)

pytestmark = pytest.mark.skipif(not DSN, reason="AATM_TEST_PG_DSN is not set")

if DSN:
    from aatm.pgvector_store import (
        PgVectorStore,
        create_pg_indexes,
        create_pg_table,
        drop_pg_table,
        pg_table_count,
        pg_table_exists,
    )


@pytest.fixture
def connection():
    """A connection whose tables have a random prefix, dropped after the test."""
    prefix = f"aatm_test_{uuid.uuid4().hex[:8]}"
    connection = PgVectorConnection(dsn=DSN, table_prefix=prefix)
    yield connection
    with connection.client() as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT tablename FROM pg_tables WHERE tablename LIKE %s",
                (f"{prefix}%",),
            )
        ]
    for table in tables:
        drop_pg_table(DSN, table)


def make_store(connection, name="index", dimension=DIM, metadata=None, **kwargs):
    table = connection.table_name(name)
    create_pg_table(DSN, table, dimension, metadata)
    return PgVectorStore(DSN, table, **kwargs)


@pytest.fixture
def store(connection):
    """The records and vectors of the ChromaDB `client` fixture, with search indexes."""
    store = make_store(connection, metadata={"k": "v"})
    rng = np.random.default_rng(0)
    write_records(store, rng.normal(size=(len(DOMAINS), DIM)).astype(np.float32))
    create_pg_indexes(DSN, store.name)
    return store


def write_records(store, embeddings, attempt=0):
    ids = [str(i) for i in range(len(DOMAINS))]
    store.write(
        ids,
        embeddings,
        [f"expression {i}" for i in ids],
        [record_metadata(i, d) for i, d in enumerate(DOMAINS)],
        attempt=attempt,
    )


def make_retriever(store, **kwargs) -> PgVectorRetriever:
    return PgVectorRetriever(store, HashEmbeddingFunction(), top_k=5, **kwargs)


# --- The store -------------------------------------------------------------------


def test_store_counts_and_finds_records(store):
    assert store.count() == len(DOMAINS)
    assert store.existing_ids(["0", "5", "missing"]) == {"0", "5"}
    assert store.existing_ids([]) == set()
    assert store.index_metadata() == {"k": "v"}
    assert store.version is not None and store.version >= (0, 8, 0)


def test_table_status(connection, store):
    assert pg_table_exists(DSN, store.name)
    assert pg_table_count(DSN, store.name) == len(DOMAINS)
    assert pg_table_count(DSN, connection.table_name("missing")) is None
    empty = make_store(connection, "empty")
    assert pg_table_count(DSN, empty.name) == 0


def test_distances_are_squared_euclidean(connection):
    store = make_store(connection)
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(50, DIM)).astype(np.float32)
    store.write(
        [str(i) for i in range(50)],
        vectors,
        [""] * 50,
        [{"i": i} for i in range(50)],
    )

    (hits,) = store.query(vectors[:1], top_k=10)

    exact = np.sort(((vectors - vectors[0]) ** 2).sum(axis=1))[:10]
    assert [d for d, _ in hits] == pytest.approx(exact, rel=1e-5, abs=1e-5)
    assert hits[0][1] == {"i": 0}


def test_first_writes_skip_existing_ids_and_retries_replace_them(store):
    zeros = np.zeros((len(DOMAINS), DIM), dtype=np.float32)

    write_records(store, zeros)  # attempt 0: like ChromaDB's add, keeps the rows
    assert store.count() == len(DOMAINS)
    (hits,) = store.query(zeros[:1], top_k=1)
    assert hits[0][0] > 0

    write_records(store, zeros, attempt=1)  # a retry upserts
    assert store.count() == len(DOMAINS)
    (hits,) = store.query(zeros[:1], top_k=1)
    assert hits[0][0] == pytest.approx(0)


def test_a_batch_that_repeats_an_id_is_written_once(connection):
    store = make_store(connection)
    vectors = np.ones((2, DIM), dtype=np.float32)

    store.write(["a", "a"], vectors, ["x", "x"], [{}, {}])

    assert store.count() == 1


@pytest.mark.parametrize(
    "where",
    [
        {"std_domain_id": "Measurement"},
        retriever_filter("Measurement", ["LOINC"]),
        {"std_vocabulary_id": {"$in": ["LOINC", "SNOMED"]}},
        {"$or": [{"std_domain_id": "Device"}, {"std_vocabulary_id": "LOINC"}]},
        {"std_domain_id": {"$ne": "Condition"}},
        {"std_domain_id": {"$nin": ["Condition", "Device"]}},
        {"missing_field": {"$ne": "x"}},
    ],
)
def test_filters_match_what_python_evaluates(store, where):
    rng = np.random.default_rng(0)
    query = rng.normal(size=(1, DIM)).astype(np.float32)

    (hits,) = store.query(query, top_k=len(DOMAINS), where=where)

    expected = {
        str(i)
        for i, domain in enumerate(DOMAINS)
        if _matches(record_metadata(i, domain), where)
    }
    assert {m["expression_concept_id"] for _, m in hits} == expected


def _matches(metadata, where):
    ((key, value),) = where.items()
    if isinstance(value, dict) and set(value) & {"$ne", "$nin"}:
        if key not in metadata:
            return False
        ((op, operand),) = value.items()
        return metadata[key] != operand if op == "$ne" else metadata[key] not in operand
    return matches_filter(metadata, where)


def test_numeric_comparisons_ignore_other_types(connection):
    store = make_store(connection)
    values = [1, 2.5, 3, "4", True, None]
    store.write(
        [str(i) for i in range(len(values))],
        np.ones((len(values), DIM), dtype=np.float32),
        [""] * len(values),
        [{"n": v} for v in values],
    )

    (hits,) = store.query(np.ones((1, DIM)), top_k=10, where={"n": {"$gte": 2}})

    assert sorted(m["n"] for _, m in hits) == [2.5, 3]


def test_a_filtered_search_finds_top_k_rare_matches(connection):
    """Iterative scans keep searching past the HNSW candidates for rare matches."""
    store = make_store(connection, ef_search=10)
    rng = np.random.default_rng(2)
    n = 3000
    vectors = rng.normal(size=(n, DIM)).astype(np.float32)
    store.write(
        [str(i) for i in range(n)],
        vectors,
        [""] * n,
        [{"rare": i % 300 == 0} for i in range(n)],
    )
    create_pg_indexes(DSN, store.name)

    (hits,) = store.query(vectors[1:2], top_k=10, where={"rare": True})

    assert len(hits) == 10
    assert all(m["rare"] for _, m in hits)


def test_large_embeddings_are_indexed_as_halfvec(connection):
    dimension = 2500
    store = make_store(connection, dimension=dimension)
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(40, dimension)).astype(np.float32)
    store.write(
        [str(i) for i in range(40)], vectors, [""] * 40, [{"i": i} for i in range(40)]
    )
    create_pg_indexes(DSN, store.name)
    store = PgVectorStore(DSN, store.name)

    (hits,) = store.query(vectors[5:6], top_k=3)

    assert store.index_kind == "halfvec"
    assert hits[0][1] == {"i": 5}
    # Distances come from the full-precision vectors.
    exact = np.sort(((vectors - vectors[5]) ** 2).sum(axis=1))[:3]
    assert [d for d, _ in hits] == pytest.approx(exact, rel=1e-4)


def test_exact_search_matches_brute_force(store):
    exact_store = PgVectorStore(DSN, store.name, exact=True)
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(len(DOMAINS), DIM)).astype(np.float32)

    (hits,) = exact_store.query(
        vectors[:1], top_k=20, where={"std_domain_id": "Condition"}
    )

    condition = [i for i, d in enumerate(DOMAINS) if d == "Condition"]
    distances = ((vectors[condition] - vectors[0]) ** 2).sum(axis=1)
    want = [str(condition[i]) for i in np.argsort(distances)[:20]]
    assert [m["expression_concept_id"] for _, m in hits] == want


def test_lookups_and_writes_use_one_connection_per_thread(store):
    import threading

    connections = {}

    def lookup():
        store.existing_ids(["0"])
        connections["thread"] = store._conn()

    thread = threading.Thread(target=lookup)
    thread.start()
    thread.join()

    assert connections["thread"] is not store._conn()


def test_a_missing_table_is_explained(connection):
    with pytest.raises(VectorDatabaseConfigError, match="No AATM index table"):
        PgVectorStore(DSN, connection.table_name("missing"))


def test_an_unreachable_server_is_explained():
    with pytest.raises(VectorDatabaseConfigError, match="Could not connect"):
        PgVectorStore(
            "postgresql://postgres:pw@127.0.0.1:1/aatm?connect_timeout=2", "x"
        )


# --- The retriever ---------------------------------------------------------------


def make_chroma_retriever(client) -> ChromaDBRetriever:
    return ChromaDBRetriever(
        client=client,
        collection_name=COLLECTION,
        embedding_function=HashEmbeddingFunction(),
        top_k=5,
        filter_strategy="chroma",
    )


@pytest.mark.parametrize(
    "where",
    [
        None,
        {"std_domain_id": "Condition"},
        {"std_domain_id": "Measurement"},
        retriever_filter("Measurement", ["LOINC"]),
        retriever_filter("Device", None),
    ],
)
@pytest.mark.parametrize("strategy", ["native", "overfetch_fallback", "overfetch"])
def test_pgvector_returns_what_chromadb_returns(client, store, where, strategy):
    expected = make_chroma_retriever(client).retrieve(QUERIES, where=where)
    retriever = make_retriever(
        store, filter_strategy=strategy, min_overfetch=50, max_overfetch=400
    )

    got = retriever.retrieve(QUERIES, where=where)

    for got_query, expected_query in zip(got.results, expected.results):
        assert [r.model_dump(exclude={"distance"}) for r in got_query] == [
            r.model_dump(exclude={"distance"}) for r in expected_query
        ]
        assert [r.distance for r in got_query] == pytest.approx(
            [r.distance for r in expected_query], rel=1e-5, abs=1e-5
        )


def test_the_default_strategy_filters_in_the_database(store):
    assert make_retriever(store).filter_strategy == "native"


def test_null_metadata_fields_come_back_as_none(connection):
    store = make_store(connection)
    metadata = record_metadata(0, "Condition") | {"std_vocabulary_code": None}
    del metadata["expression_concept_id"]  # absent, as ChromaDB leaves out nulls
    store.write(
        ["0"], np.zeros((1, DIM), dtype=np.float32), ["expression 0"], [metadata]
    )

    (results,) = make_retriever(store).retrieve(["q"]).results

    expected = RetrievedExpressionMetadata(
        distance=results[0].distance, expression_concept_id=None, **metadata
    )
    assert results[0] == expected


def test_coding_cache_fingerprints_the_table(store):
    store.location = "db:5432/aatm"

    assert data_fingerprint(make_retriever(store)) == {
        "pg_table": store.name,
        "location": "db:5432/aatm",
        "dimension": DIM,
        "count": len(DOMAINS),
    }


# --- Registry, build and upload --------------------------------------------------


@pytest.fixture
def pg_project(fake_project, connection, monkeypatch):
    def pgvector_location(self):
        return connection, connection.table_name(self.name)

    monkeypatch.setattr(
        registry.ChromaRetrieverModelSpec, "pgvector_location", pgvector_location
    )
    return connection


def embedded_texts():
    return sum(len(call) for call in FakeEmbeddingFunction.calls)


def table_store(connection, model="fake-model"):
    return connection.open_store(connection.table_name(model))


def test_build_writes_every_row_once_and_indexes_the_table(pg_project, fake_project):
    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="pgvector")

    store = table_store(pg_project)
    assert store.count() == 280
    assert store.dimension == 4  # learned from one probe embedding
    assert embedded_texts() == 280 + 1
    assert not fake_project.exists()  # nothing built locally
    with pg_project.client() as conn:
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT indexdef FROM pg_indexes WHERE tablename = %s", (store.name,)
            )
        }
    assert any("USING hnsw" in index for index in indexes)
    assert any("USING gin" in index for index in indexes)


def test_repair_only_embeds_missing_rows(pg_project, fake_project, monkeypatch):
    from test_local_database_utils import write_dataset

    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="pgvector")
    write_dataset(
        fake_project.parent.parent / "datasets/standard_concept_synonym.csv",
        n_rows=40,
    )  # 10 new rows
    FakeEmbeddingFunction.calls = []
    answer_prompt(monkeypatch, "Repair")

    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="pgvector")

    assert embedded_texts() == 10
    assert table_store(pg_project).count() == 290


def test_overwrite_recreates_the_table(pg_project, monkeypatch):
    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="pgvector")
    FakeEmbeddingFunction.calls = []
    answer_prompt(monkeypatch, "Overwrite")

    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="pgvector")

    assert embedded_texts() == 280 + 1
    assert table_store(pg_project).count() == 280


def test_skip_adds_the_indexes_an_interrupted_build_missed(pg_project, monkeypatch):
    table = pg_project.table_name("fake-model")
    store = make_store(pg_project, "fake-model", dimension=4)
    store.write(["x"], np.ones((1, 4), dtype=np.float32), ["x"], [{}])
    answer_prompt(monkeypatch, "Skip")

    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="pgvector")

    with pg_project.client() as conn:
        indexes = [
            row[0]
            for row in conn.execute(
                "SELECT indexdef FROM pg_indexes WHERE tablename = %s", (table,)
            )
        ]
    assert any("USING hnsw" in index for index in indexes)


def test_build_rejects_a_local_directory(pg_project, tmp_path):
    with pytest.raises(ValueError, match="local vector database"):
        ldu.build_local_vector_database(
            "fake-model", vector_db_dir=tmp_path, vector_db="pgvector"
        )


def test_load_retriever_opens_the_models_table(pg_project, monkeypatch):
    monkeypatch.setenv("AATM_VECTOR_DB", "pgvector")
    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="pgvector")

    retriever = registry.load_retriever("fake-model", top_k=3)

    assert isinstance(retriever, PgVectorRetriever)
    assert retriever.table_name == pg_project.table_name("fake-model")
    (results,) = retriever.retrieve(["expression 7"]).results
    assert len(results) == 3


def test_load_retriever_explains_a_missing_table(pg_project, monkeypatch):
    monkeypatch.setenv("AATM_VECTOR_DB", "pgvector")

    with pytest.raises(ValueError, match="--vector-db pgvector") as error:
        registry.load_retriever("fake-model")
    assert "PostgreSQL has no table" in str(error.value)
    assert "aatm vector-db upload --to pgvector -e fake-model" in str(error.value)


def test_upload_copies_the_local_index_and_resumes(fake_project, connection):
    ldu.build_local_vector_database("fake-model", chunk_size=100)
    FakeEmbeddingFunction.calls = []

    summary = ldu.upload_vector_database_to_cloud(
        "fake-model", cloud_connection=connection, write_batch_size=64
    )

    assert summary.collection_name == connection.table_name("fake-model")
    assert (summary.total, summary.uploaded, summary.skipped) == (280, 280, 0)
    assert FakeEmbeddingFunction.calls == []
    assert table_store(connection).count() == 280

    again = ldu.upload_vector_database_to_cloud(
        "fake-model", cloud_connection=connection
    )
    assert (again.uploaded, again.skipped) == (0, 280)


def test_upload_to_pgvector_is_chosen_with_to(fake_project, connection, monkeypatch):
    ldu.build_local_vector_database("fake-model", chunk_size=100)
    monkeypatch.setenv("AATM_PG_DSN", DSN)
    monkeypatch.setenv("AATM_PG_TABLE_PREFIX", connection.table_prefix)

    summary = ldu.upload_vector_database_to_cloud("fake-model", to="pgvector")

    assert summary.uploaded == 280


def test_upload_refuses_to_mix_precisions(fake_project, connection):
    ldu.build_local_vector_database("fake-model", chunk_size=100)
    create_pg_table(
        DSN,
        connection.table_name("fake-model"),
        4,
        {"aatm_embedding_dtype": "bfloat16"},
    )

    with pytest.raises(ValueError, match="mix precisions"):
        ldu.upload_vector_database_to_cloud("fake-model", cloud_connection=connection)


def test_vector_db_check_lists_tables(pg_project, monkeypatch):
    from typer.testing import CliRunner

    from aatm.main import app

    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="pgvector")
    monkeypatch.setattr(PgVectorConnection, "from_env", lambda env=None: pg_project)

    result = CliRunner().invoke(
        app, ["vector-db", "check", "--vector-db", "pgvector"], env={"COLUMNS": "200"}
    )

    assert result.exit_code == 0, result.output
    assert "Connected" in result.output
    assert "pgvector 0." in result.output
    table = pg_project.table_name("fake-model")
    assert f"fake-model: table '{table}' (built, about 280 rows)" in result.output
