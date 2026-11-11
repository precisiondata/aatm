"""Pinecone as the vector database, against a fake Pinecone client (see fake_pinecone)."""

# ruff: noqa: F811 - the imported fixtures are used as test arguments

import numpy as np
import pytest

from aatm import local_database_utils as ldu
from aatm.omop.cache import data_fingerprint
from aatm.omop.coding import retriever_filter
from aatm.registries import retrievers as registry
from aatm.retrievers import ChromaDBRetriever, PineconeRetriever
from aatm.vector_db import (
    ChromaConnection,
    PineconeConnection,
    VectorDatabaseConfigError,
    call_with_retries,
    connection_from_env,
    is_transient_error,
    pinecone_index_name,
)
from aatm.vector_stores import (
    DOCUMENT_METADATA_KEY,
    PINECONE_MAX_TOP_K,
    PineconeVectorStore,
    UnsupportedFilterError,
    create_pinecone_index,
    from_pinecone_metadata,
    pinecone_document,
    plan_upsert_batches,
    to_pinecone_filter,
    to_pinecone_metadata,
)
from fake_pinecone import FakePineconeConnection, fake_pinecone
from test_local_database_utils import (  # noqa: F401 - fixtures
    FakeEmbeddingFunction,
    FakeLocalModel,
    answer_prompt,
    fake_local_project,
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

PINECONE_ENV = {"AATM_VECTOR_DB": "pinecone", "PINECONE_API_KEY": "secret-key"}


# --- Configuration ---------------------------------------------------------------


def test_pinecone_needs_an_api_key():
    with pytest.raises(VectorDatabaseConfigError, match="PINECONE_API_KEY"):
        connection_from_env("x", env={"AATM_VECTOR_DB": "pinecone"})


def test_pinecone_settings_come_from_the_environment():
    env = PINECONE_ENV | {
        "PINECONE_INDEX_PREFIX": "team",
        "PINECONE_CLOUD": "gcp",
        "PINECONE_REGION": "europe-west4",
    }
    connection = connection_from_env("x", env=env)

    assert isinstance(connection, PineconeConnection)
    assert (connection.index_prefix, connection.cloud, connection.region) == (
        "team",
        "gcp",
        "europe-west4",
    )
    assert connection.index_name("qwen3-06B") == "team-qwen3-06b"


def test_pinecone_defaults():
    connection = PineconeConnection.from_env(PINECONE_ENV)

    assert (connection.cloud, connection.region) == ("aws", "us-east-1")
    assert connection.index_name("qwen3-06B") == "aatm-expressions-qwen3-06b"
    assert connection.is_cloud and connection.mode == "pinecone"


def test_pinecone_api_key_never_appears_in_repr_or_description():
    connection = PineconeConnection.from_env(PINECONE_ENV)
    assert "secret-key" not in repr(connection)
    assert "secret-key" not in connection.describe()


def test_chroma_connection_refuses_pinecone_mode():
    with pytest.raises(VectorDatabaseConfigError, match="connection_from_env"):
        ChromaConnection.from_env("x", env=PINECONE_ENV)


def test_connection_from_env_keeps_chroma_modes():
    assert connection_from_env("db", env={}).path == "db"
    cloud = {"AATM_VECTOR_DB": "cloud", "CHROMA_API_KEY": "k"}
    assert connection_from_env("db", env=cloud).is_cloud


@pytest.mark.parametrize(
    ("prefix", "model", "expected"),
    [
        ("aatm-expressions", "qwen3-06B", "aatm-expressions-qwen3-06b"),
        (
            "aatm-expressions",
            "text-embedding-3-small",
            "aatm-expressions-text-embedding-3-small",
        ),
        ("My_Team", "gemini.embedding/001", "my-team-gemini-embedding-001"),
        ("", "-model-", "model"),
    ],
)
def test_index_names_are_valid_for_pinecone(prefix, model, expected):
    assert pinecone_index_name(prefix, model) == expected


def test_long_index_names_are_shortened_without_collisions():
    a = pinecone_index_name("aatm-expressions", "a-very-long-embedding-model-name-v1")
    b = pinecone_index_name("aatm-expressions", "a-very-long-embedding-model-name-v2")

    assert len(a) <= 45 and len(b) <= 45
    assert a != b
    assert not a.endswith("-") and not a.startswith("-")


def test_index_name_needs_valid_characters():
    with pytest.raises(VectorDatabaseConfigError):
        pinecone_index_name("", "___")


def test_pinecone_rate_limits_and_server_errors_are_transient():
    from pinecone.errors import exceptions as errors

    assert is_transient_error(errors.RateLimitError())
    assert is_transient_error(errors.ServiceError())
    assert is_transient_error(errors.PineconeConnectionError("reset"))
    assert not is_transient_error(errors.UnauthorizedError())
    assert not is_transient_error(errors.NotFoundError())


def test_pinecone_rate_limits_are_retried():
    from pinecone.errors import exceptions as errors

    attempts = []

    def flaky(attempt):
        attempts.append(attempt)
        if attempt < 2:
            raise errors.RateLimitError()
        return "ok"

    assert call_with_retries(flaky, sleep=lambda _: None) == "ok"
    assert attempts == [0, 1, 2]


# --- Filters, metadata and batches ----------------------------------------------


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        ({"std_domain_id": "Measurement"}, {"std_domain_id": {"$eq": "Measurement"}}),
        ({"a": {"$eq": 1}}, {"a": {"$eq": 1}}),
        ({"a": {"$in": ["x", "y"]}}, {"a": {"$in": ["x", "y"]}}),
        ({"a": {"$nin": ["x"]}}, {"a": {"$nin": ["x"]}}),
        ({"a": {"$gte": 2, "$lt": 5}}, {"a": {"$gte": 2, "$lt": 5}}),
        ({"a": True}, {"a": {"$eq": True}}),
        (
            retriever_filter("Measurement", ["LOINC", "SNOMED"]),
            {
                "$and": [
                    {"std_domain_id": {"$eq": "Measurement"}},
                    {"std_vocabulary_id": {"$in": ["LOINC", "SNOMED"]}},
                ]
            },
        ),
        (
            {"$or": [{"a": "x"}, {"b": {"$ne": "y"}}]},
            {"$or": [{"a": {"$eq": "x"}}, {"b": {"$ne": "y"}}]},
        ),
        ({"a": "x", "b": "y"}, {"$and": [{"a": {"$eq": "x"}}, {"b": {"$eq": "y"}}]}),
    ],
)
def test_filters_translate_to_pinecone(where, expected):
    assert to_pinecone_filter(where) == expected


@pytest.mark.parametrize(
    "where",
    [
        {},
        {"$contains": "x"},
        {"a": {"$contains": "x"}},
        {"a": {"$in": "x"}},
        {"a": ["x"]},
        {"$and": []},
        {"a": {"$in": [str(i) for i in range(10_001)]}},
    ],
)
def test_unsupported_filters_are_rejected(where):
    with pytest.raises(UnsupportedFilterError):
        to_pinecone_filter(where)


def test_metadata_drops_nulls_and_keeps_the_document_once():
    metadata = {"expression": "creatinine", "target_vocabulary_id": None, "n": 3}

    converted = to_pinecone_metadata(metadata, document="creatinine")

    assert converted == {"expression": "creatinine", "n": 3}
    assert pinecone_document(converted) == "creatinine"


def test_a_document_other_than_the_expression_is_stored():
    converted = to_pinecone_metadata({"expression": "a"}, document="b")

    assert converted[DOCUMENT_METADATA_KEY] == "b"
    assert pinecone_document(converted) == "b"
    assert from_pinecone_metadata(converted) == {"expression": "a"}


def test_whole_floats_become_integers_again():
    assert from_pinecone_metadata({"n": 3.0, "x": 0.5, "flag": True}) == {
        "n": 3,
        "x": 0.5,
        "flag": True,
    }


def test_metadata_over_40_kb_is_rejected():
    with pytest.raises(ValueError, match="40960"):
        to_pinecone_metadata({"expression": "x" * 41_000})


def record(i: int, dim: int, metadata_size: int = 10) -> dict:
    return {"id": str(i), "values": [0.0] * dim, "metadata": {"m": "x" * metadata_size}}


def test_upsert_batches_respect_the_record_limit():
    batches = plan_upsert_batches([record(i, 4) for i in range(2500)])
    assert [len(b) for b in batches] == [1000, 1000, 500]


def test_upsert_batches_respect_the_request_size():
    records = [record(i, 3072) for i in range(100)]  # ~74 KB each in JSON

    batches = plan_upsert_batches(records, max_bytes=1_000_000)

    assert sum(len(b) for b in batches) == 100
    assert all(len(b) <= 11 for b in batches)
    assert [r["id"] for b in batches for r in b] == [str(i) for i in range(100)]


def test_a_record_too_large_for_one_request_is_rejected():
    with pytest.raises(ValueError, match="too large"):
        plan_upsert_batches([record(0, 100_000)])


# --- The store and the retriever ------------------------------------------------


def write_records(store, embeddings):
    ids = [str(i) for i in range(len(DOMAINS))]
    store.write(
        ids,
        embeddings,
        [f"expression {i}" for i in ids],
        [record_metadata(i, d) for i, d in enumerate(DOMAINS)],
    )


@pytest.fixture
def pinecone_store(tmp_path):
    connection = fake_pinecone(tmp_path / "pinecone.db")
    pc = connection.client()
    create_pinecone_index(pc, "test-index", DIM, "aws", "us-east-1", tags={"k": "v"})
    store = PineconeVectorStore.open(pc, "test-index", query_concurrency=1)
    # The same records and vectors as the ChromaDB `client` fixture.
    rng = np.random.default_rng(0)
    write_records(store, rng.normal(size=(len(DOMAINS), DIM)).astype(np.float32))
    return store


def make_pinecone_retriever(store, **kwargs) -> PineconeRetriever:
    kwargs.setdefault("overfetch_factor", 1)
    return PineconeRetriever(store, HashEmbeddingFunction(), top_k=5, **kwargs)


def make_chroma_retriever(client, **kwargs) -> ChromaDBRetriever:
    return ChromaDBRetriever(
        client=client,
        collection_name=COLLECTION,
        embedding_function=HashEmbeddingFunction(),
        top_k=5,
        filter_strategy="chroma",
        **kwargs,
    )


def test_store_counts_and_finds_records(pinecone_store):
    assert pinecone_store.count() == len(DOMAINS)
    assert pinecone_store.existing_ids(["0", "5", "missing"]) == {"0", "5"}
    assert pinecone_store.index_metadata() == {"k": "v"}


def test_existing_ids_are_fetched_in_batches(pinecone_store):
    pinecone_store.fetch_batch_size = 7
    ids = [str(i) for i in range(len(DOMAINS))] + ["missing"]

    assert pinecone_store.existing_ids(ids) == set(ids) - {"missing"}


def test_rewriting_records_does_not_duplicate_them(pinecone_store):
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(len(DOMAINS), DIM)).astype(np.float32)

    write_records(pinecone_store, embeddings)

    assert pinecone_store.count() == len(DOMAINS)


def test_the_default_strategy_passes_the_filter_to_pinecone(pinecone_store):
    retriever = make_pinecone_retriever(pinecone_store)
    assert retriever.filter_strategy == "native"

    retriever.retrieve(QUERIES[:2], where=retriever_filter("Measurement", ["LOINC"]))

    calls = pinecone_store.index.queries
    assert len(calls) == 2  # one request per query
    assert all(call["top_k"] == 5 for call in calls)
    assert calls[0]["filter"] == {
        "$and": [
            {"std_domain_id": {"$eq": "Measurement"}},
            {"std_vocabulary_id": {"$in": ["LOINC"]}},
        ]
    }


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
def test_pinecone_returns_what_chromadb_returns(
    client, pinecone_store, where, strategy
):
    expected = make_chroma_retriever(client).retrieve(QUERIES, where=where)
    retriever = make_pinecone_retriever(
        pinecone_store, filter_strategy=strategy, min_overfetch=50, max_overfetch=400
    )

    got = retriever.retrieve(QUERIES, where=where)

    for got_query, expected_query in zip(got.results, expected.results):
        assert [r.model_dump(exclude={"distance"}) for r in got_query] == [
            r.model_dump(exclude={"distance"}) for r in expected_query
        ]
        # Same distance, so the same confidence score (1 - distance).
        assert [r.distance for r in got_query] == pytest.approx(
            [r.distance for r in expected_query], rel=1e-5, abs=1e-5
        )


def test_over_fetch_sizes_are_capped_at_pinecones_top_k(pinecone_store):
    retriever = PineconeRetriever(
        pinecone_store,
        HashEmbeddingFunction(),
        overfetch_factor=1000,
        max_overfetch=50_000,
    )

    assert retriever.overfetch_sizes(20) == [PINECONE_MAX_TOP_K]


def test_large_top_k_is_capped(pinecone_store):
    retriever = make_pinecone_retriever(pinecone_store)

    results = retriever.retrieve(QUERIES[:1], top_k=20_000)

    assert len(results.results[0]) == len(DOMAINS)
    assert pinecone_store.index.queries[-1]["top_k"] == PINECONE_MAX_TOP_K


def test_coding_cache_fingerprints_the_index(pinecone_store):
    retriever = make_pinecone_retriever(pinecone_store)

    assert data_fingerprint(retriever) == {
        "pinecone_index": "test-index",
        "host": "test-index.fake.pinecone.io",
        "dimension": DIM,
        "count": len(DOMAINS),
    }


# --- load_retriever --------------------------------------------------------------


@pytest.fixture
def pinecone_env(tmp_path, monkeypatch):
    for name, value in PINECONE_ENV.items():
        monkeypatch.setenv(name, value)
    path = tmp_path / "pinecone.db"
    monkeypatch.setattr(
        PineconeConnection, "client", lambda self: fake_pinecone(path).client()
    )
    return fake_pinecone(path)


def register(monkeypatch, name, embedding_function_cls):
    spec = registry.ChromaRetrieverModelSpec(
        name=name, model_id=name, embedding_function_cls=embedding_function_cls
    )
    monkeypatch.setitem(registry.CHROMADB_RETRIEVER_MODEL_REGISTRY, name, spec)
    return spec


def test_location_is_one_index_per_model(pinecone_env, monkeypatch):
    spec = register(monkeypatch, "fake-model", FakeEmbeddingFunction)

    connection, name = spec.vector_db_location()

    assert isinstance(connection, PineconeConnection)
    assert name == "aatm-expressions-fake-model"


def test_load_retriever_opens_the_models_index(pinecone_env, monkeypatch):
    register(monkeypatch, "fake-model", FakeEmbeddingFunction)
    create_pinecone_index(
        pinecone_env.client(), "aatm-expressions-fake-model", 4, "aws", "us-east-1"
    )

    retriever = registry.load_retriever("fake-model", top_k=3)

    assert isinstance(retriever, PineconeRetriever)
    assert retriever.index_name == "aatm-expressions-fake-model"
    assert retriever.collection_name == "aatm-expressions-fake-model"
    assert retriever.top_k == 3


def test_load_retriever_explains_a_missing_index(pinecone_env, monkeypatch):
    register(monkeypatch, "fake-model", FakeEmbeddingFunction)

    with pytest.raises(ValueError, match="aatm init --vector-db pinecone"):
        registry.load_retriever("fake-model")


def test_queries_are_encoded_in_the_indexs_precision(pinecone_env, monkeypatch):
    from aatm import precision

    monkeypatch.setattr(registry, "get_device", lambda: "cuda")
    monkeypatch.setattr(
        precision.torch.cuda, "is_bf16_supported", lambda including_emulation=True: True
    )
    FakeLocalModel.loaded_dtypes = []
    register(monkeypatch, "fake-local", FakeLocalModel)
    create_pinecone_index(
        pinecone_env.client(),
        "aatm-expressions-fake-local",
        4,
        "aws",
        "us-east-1",
        tags={"aatm_embedding_dtype": "bfloat16"},
    )

    registry.load_retriever("fake-local")

    assert FakeLocalModel.loaded_dtypes == ["bfloat16"]


# --- Building the index ----------------------------------------------------------


@pytest.fixture
def pinecone_project(fake_project, tmp_path, monkeypatch):
    connection = fake_pinecone(tmp_path / "pinecone.db")

    def pinecone_location(self):
        return connection, connection.index_name(self.name)

    monkeypatch.setattr(
        registry.ChromaRetrieverModelSpec, "pinecone_location", pinecone_location
    )
    return connection


def index_store(connection, name="aatm-expressions-fake-model"):
    return PineconeVectorStore.open(connection.client(), name)


def embedded_texts():
    return sum(len(call) for call in FakeEmbeddingFunction.calls)


def test_pinecone_build_writes_every_row_once(pinecone_project, fake_project):
    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="pinecone")

    store = index_store(pinecone_project)
    assert store.count() == 280
    assert store.dimension == 4  # learned from one probe embedding
    assert embedded_texts() == 280 + 1
    assert not fake_project.exists()  # nothing built locally


def test_pinecone_repair_only_embeds_missing_rows(
    pinecone_project, fake_project, monkeypatch
):
    from test_local_database_utils import write_dataset

    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="pinecone")
    write_dataset(
        fake_project.parent.parent / "datasets/standard_concept_synonym.csv",
        n_rows=40,
    )  # 10 new rows
    FakeEmbeddingFunction.calls = []
    answer_prompt(monkeypatch, "Repair")

    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="pinecone")

    assert embedded_texts() == 10
    assert index_store(pinecone_project).count() == 290


def test_pinecone_overwrite_recreates_the_index(pinecone_project, monkeypatch):
    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="pinecone")
    FakeEmbeddingFunction.calls = []
    answer_prompt(monkeypatch, "Overwrite")

    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="pinecone")

    assert embedded_texts() == 280 + 1
    assert index_store(pinecone_project).count() == 280


def test_pinecone_skip_leaves_the_index_alone(pinecone_project, monkeypatch):
    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="pinecone")
    FakeEmbeddingFunction.calls = []
    answer_prompt(monkeypatch, "Skip")

    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="pinecone")

    assert embedded_texts() == 0


def test_pinecone_build_rejects_a_local_directory(pinecone_project, tmp_path):
    with pytest.raises(ValueError, match="local vector database"):
        ldu.build_local_vector_database(
            "fake-model", vector_db_dir=tmp_path, vector_db="pinecone"
        )


def test_pinecone_index_records_the_precision(
    fake_local_project, tmp_path, monkeypatch
):
    connection = fake_pinecone(tmp_path / "pinecone.db")
    monkeypatch.setattr(
        registry.ChromaRetrieverModelSpec,
        "pinecone_location",
        lambda self: (connection, connection.index_name(self.name)),
    )

    ldu.build_local_vector_database(
        "fake-local", encode_batch_size=8, vector_db="pinecone", embedding_dtype="bf16"
    )

    store = index_store(connection, "aatm-expressions-fake-local")
    assert store.index_metadata() == {"aatm_embedding_dtype": "bfloat16"}
    assert store.count() == 20


# --- Copying a local index to Pinecone --------------------------------------------


def test_upload_copies_the_local_index_without_re_embedding(fake_project, tmp_path):
    import chromadb

    ldu.build_local_vector_database("fake-model", chunk_size=100)
    FakeEmbeddingFunction.calls = []
    connection = fake_pinecone(tmp_path / "pinecone.db")

    summary = ldu.upload_vector_database_to_cloud(
        "fake-model",
        cloud_connection=connection,
        read_page_size=64,
        write_batch_size=50,
    )

    assert summary.collection_name == "aatm-expressions-fake-model"
    assert (summary.total, summary.uploaded, summary.skipped) == (280, 280, 0)
    assert FakeEmbeddingFunction.calls == []

    local = chromadb.PersistentClient(path=str(fake_project)).get_collection(
        "expressions"
    )
    try:
        expected = local.get(include=["embeddings", "metadatas"])
    finally:
        del local
        ldu.release_chromadb_clients()
    embeddings = np.asarray(expected["embeddings"], dtype=np.float64)
    query = embeddings[:3]

    store = index_store(connection)
    hits = store.query(query, top_k=10)
    # The fake embeddings ([len(text)] * 4) have many exact ties, so compare
    # distances, not the order of records at equal distance. Compare with the exact
    # nearest distances, not ChromaDB's: its HNSW search is approximate and, with
    # so many duplicates, sometimes misses records at distance 0.
    for got, q in zip(hits, query):
        want_distances = np.sort(((embeddings - q) ** 2).sum(axis=1))[:10]
        assert [d for d, _ in got] == pytest.approx(want_distances, abs=1e-4)

    by_id = {m["expression_id"]: m for query_hits in hits for _, m in query_hits}
    for identifier, metadata in zip(expected["ids"], expected["metadatas"]):
        if identifier in by_id:
            assert by_id[identifier] == {
                k: v for k, v in metadata.items() if v is not None
            }


def test_upload_to_pinecone_resumes(fake_project, tmp_path):
    ldu.build_local_vector_database("fake-model", chunk_size=100)
    connection = fake_pinecone(tmp_path / "pinecone.db")
    ldu.upload_vector_database_to_cloud("fake-model", cloud_connection=connection)

    summary = ldu.upload_vector_database_to_cloud(
        "fake-model", cloud_connection=connection
    )

    assert (summary.uploaded, summary.skipped) == (0, 280)
    assert index_store(connection).count() == 280


def test_upload_to_pinecone_is_chosen_with_to(fake_project, tmp_path, monkeypatch):
    ldu.build_local_vector_database("fake-model", chunk_size=100)
    monkeypatch.setenv("PINECONE_API_KEY", "secret-key")
    path = tmp_path / "pinecone.db"
    monkeypatch.setattr(
        PineconeConnection, "client", lambda self: fake_pinecone(path).client()
    )

    summary = ldu.upload_vector_database_to_cloud("fake-model", to="pinecone")

    assert summary.uploaded == 280


def test_upload_refuses_to_mix_precisions(fake_project, tmp_path):
    ldu.build_local_vector_database("fake-model", chunk_size=100)
    connection = fake_pinecone(tmp_path / "pinecone.db")
    create_pinecone_index(
        connection.client(),
        "aatm-expressions-fake-model",
        4,
        "aws",
        "us-east-1",
        tags={"aatm_embedding_dtype": "bfloat16"},
    )

    with pytest.raises(ValueError, match="mix precisions"):
        ldu.upload_vector_database_to_cloud("fake-model", cloud_connection=connection)


def test_upload_rejects_an_unknown_destination(fake_project):
    with pytest.raises(ValueError, match="Unknown upload destination"):
        ldu.upload_vector_database_to_cloud("fake-model", to="weaviate")


def test_fake_connection_is_a_pinecone_connection():
    # The build and upload code dispatch on the connection's type.
    assert isinstance(fake_pinecone("x"), PineconeConnection)
    assert isinstance(fake_pinecone("x"), FakePineconeConnection)


# --- CLI -------------------------------------------------------------------------


def test_vector_db_check_lists_pinecone_indexes(pinecone_env, monkeypatch):
    from typer.testing import CliRunner

    from aatm.main import app

    register(monkeypatch, "fake-model", FakeEmbeddingFunction)
    create_pinecone_index(
        pinecone_env.client(), "aatm-expressions-fake-model", 4, "aws", "us-east-1"
    )

    result = CliRunner().invoke(app, ["vector-db", "check"], env={"COLUMNS": "200"})

    assert result.exit_code == 0, result.output
    assert "Connected." in result.output
    assert "index 'aatm-expressions-fake-model' (built)" in result.output
    assert "secret-key" not in result.output


def test_vector_db_check_needs_the_pinecone_key(monkeypatch):
    from typer.testing import CliRunner

    from aatm.main import app

    result = CliRunner().invoke(app, ["vector-db", "check", "--vector-db", "pinecone"])

    assert result.exit_code == 1
    assert "PINECONE_API_KEY" in result.output


def test_null_metadata_fields_come_back_as_none(tmp_path):
    # Pinecone cannot store nulls; several metadata fields are optional but required.
    pc = fake_pinecone(tmp_path / "pinecone.db").client()
    create_pinecone_index(pc, "nulls", DIM, "aws", "us-east-1")
    store = PineconeVectorStore.open(pc, "nulls")
    metadata = record_metadata(0, "Condition") | {
        "std_vocabulary_code": None,
        "expression_concept_id": None,
    }
    store.write(
        ["0"], np.zeros((1, DIM), dtype=np.float32), ["expression 0"], [metadata]
    )

    (results,) = make_pinecone_retriever(store).retrieve(["q"]).results

    # The same as a ChromaDB record that stores the nulls.
    from aatm.data_models import RetrievedExpressionMetadata

    expected = RetrievedExpressionMetadata(distance=results[0].distance, **metadata)
    assert results[0] == expected


def test_upload_copies_only_the_given_ids(fake_project, tmp_path):
    ldu.build_local_vector_database("fake-model", chunk_size=100)
    local = stored_ids_of(fake_project)
    subset = local[:37] + ["not-in-the-index"]
    connection = fake_pinecone(tmp_path / "pinecone.db")

    summary = ldu.upload_vector_database_to_cloud(
        "fake-model", cloud_connection=connection, ids=subset, read_page_size=10
    )

    assert (summary.total, summary.uploaded, summary.skipped) == (38, 37, 0)
    store = index_store(connection)
    assert store.count() == 37
    assert store.existing_ids(local) == set(local[:37])


def stored_ids_of(db_path):
    from test_local_database_utils import stored_ids

    return sorted(stored_ids(db_path))
