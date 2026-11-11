import chromadb
import httpx
import numpy as np
import pytest
from chromadb.errors import NotFoundError, RateLimitError

from aatm import vector_db
from aatm.local_database_utils import release_chromadb_clients
from aatm.registries import retrievers
from aatm.vector_db import (
    ChromaConnection,
    VectorDatabaseConfigError,
    call_with_retries,
    resolve_vector_db_mode,
)
from aatm.vector_db_writer import ChromaWriterProcess
from fake_cloud import fake_cloud

CLOUD_ENV = {"AATM_VECTOR_DB": "cloud", "CHROMA_API_KEY": "secret-key"}


# Mode and connection settings


def test_mode_defaults_to_local():
    assert resolve_vector_db_mode(env={}) == "local"


def test_mode_comes_from_the_environment_unless_given():
    assert resolve_vector_db_mode(env={"AATM_VECTOR_DB": " Cloud "}) == "cloud"
    assert resolve_vector_db_mode("local", env={"AATM_VECTOR_DB": "cloud"}) == "local"


def test_unknown_mode_is_rejected():
    with pytest.raises(VectorDatabaseConfigError, match="Unknown vector database"):
        resolve_vector_db_mode("remote", env={})


def test_local_connection_uses_the_given_path():
    connection = ChromaConnection.from_env(local_path="some/dir", env={})
    assert connection == ChromaConnection(mode="local", path="some/dir")
    assert not connection.is_cloud


def test_cloud_needs_an_api_key():
    with pytest.raises(VectorDatabaseConfigError, match="CHROMA_API_KEY"):
        ChromaConnection.from_env(local_path="x", env={"AATM_VECTOR_DB": "cloud"})


def test_cloud_settings_come_from_the_environment():
    env = CLOUD_ENV | {"CHROMA_TENANT": "t", "CHROMA_DATABASE": "aatm"}
    connection = ChromaConnection.from_env(local_path="x", env=env)

    assert connection.is_cloud
    assert (connection.tenant, connection.database) == ("t", "aatm")
    assert connection.cloud_host == "api.trychroma.com"
    assert connection.path is None


def test_api_key_never_appears_in_repr_or_description():
    connection = ChromaConnection.from_env(local_path="x", env=CLOUD_ENV)
    assert "secret-key" not in repr(connection)
    assert "secret-key" not in connection.describe()


def test_cloud_client_gets_the_settings(monkeypatch):
    calls = []
    monkeypatch.setattr(chromadb, "CloudClient", lambda **kw: calls.append(kw))
    env = CLOUD_ENV | {"CHROMA_DATABASE": "aatm", "CHROMA_CLOUD_HOST": "example.test"}

    ChromaConnection.from_env(local_path="x", env=env).client()

    assert calls == [
        {
            "tenant": None,
            "database": "aatm",
            "api_key": "secret-key",
            "cloud_host": "example.test",
        }
    ]


def test_local_client_is_persistent(tmp_path):
    client = ChromaConnection.local(str(tmp_path)).client()
    client.create_collection("expressions", embedding_function=None)
    del client
    release_chromadb_clients()
    assert any(tmp_path.iterdir())


# Retries


def test_transient_errors_are_retried_and_attempts_are_counted():
    attempts = []

    def flaky(attempt):
        attempts.append(attempt)
        if attempt < 2:
            raise RateLimitError("slow down")
        return "ok"

    assert call_with_retries(flaky, sleep=lambda _: None) == "ok"
    assert attempts == [0, 1, 2]


def test_network_errors_are_transient():
    assert vector_db.is_transient_error(httpx.ConnectTimeout("timeout"))
    assert not vector_db.is_transient_error(NotFoundError("missing"))
    assert not vector_db.is_transient_error(ValueError("bad input"))


def test_permanent_errors_are_not_retried():
    attempts = []

    def broken(attempt):
        attempts.append(attempt)
        raise NotFoundError("no such collection")

    with pytest.raises(NotFoundError):
        call_with_retries(broken, sleep=lambda _: None)
    assert attempts == [0]


def test_retries_stop_after_the_last_attempt():
    delays = []

    def always_limited(attempt):
        raise RateLimitError("slow down")

    with pytest.raises(RateLimitError):
        call_with_retries(always_limited, attempts=3, sleep=delays.append)
    assert len(delays) == 2


# Registry


def spec(name="fake-model"):
    return retrievers.ChromaRetrieverModelSpec(
        name=name,
        model_id=name,
        embedding_function_cls=None,  # type: ignore[arg-type]
    )


def test_local_location_uses_one_directory_per_model():
    connection, collection = spec().chroma_location("local")
    assert connection.path == spec().chromadb_path
    assert collection == "expressions"


def test_cloud_location_uses_one_collection_per_model(monkeypatch):
    for name, value in CLOUD_ENV.items():
        monkeypatch.setenv(name, value)

    connection, collection = spec().chroma_location()

    assert connection.is_cloud
    assert collection == "expressions-fake-model"


class FakeQueryEmbedding(chromadb.EmbeddingFunction):
    def __init__(self, model, *args, **kwargs):
        pass

    def __call__(self, input):
        return [np.ones(3, dtype=np.float32) for _ in input]


def test_load_retriever_queries_the_cloud_collection(tmp_path, monkeypatch):
    model = retrievers.ChromaRetrieverModelSpec(
        name="fake-model",
        model_id="fake-model",
        embedding_function_cls=FakeQueryEmbedding,  # type: ignore[arg-type]
    )
    monkeypatch.setitem(
        retrievers.CHROMADB_RETRIEVER_MODEL_REGISTRY, "fake-model", model
    )
    for name, value in CLOUD_ENV.items():
        monkeypatch.setenv(name, value)
    cloud_dir = tmp_path / "cloud"
    monkeypatch.setattr(
        ChromaConnection, "client", lambda self: fake_cloud(cloud_dir).client()
    )

    retriever = retrievers.load_retriever("fake-model", top_k=3)

    assert retriever.collection_name == "expressions-fake-model"
    assert retriever.collection.name == "expressions-fake-model"


# Writer process against a (fake) cloud connection


def test_writer_process_accepts_a_connection(tmp_path):
    connection = fake_cloud(tmp_path)
    client = connection.client()
    client.create_collection("expressions", embedding_function=None)
    del client
    release_chromadb_clients()

    with ChromaWriterProcess(connection, "expressions") as writer:
        writer.add(
            1,
            ["a", "b"],
            np.ones((2, 3), dtype=np.float32),
            ["x", "y"],
            [{"k": 1}, {"k": 2}],
        )
        assert {ack.seq for ack in writer.close()} == {1}

    collection = connection.client().get_collection("expressions")
    assert sorted(collection.get(include=[])["ids"]) == ["a", "b"]
    del collection
    release_chromadb_clients()
