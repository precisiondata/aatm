"""The retriever defaults to the vector database aatm init built, and refuses an unbuilt one."""

import logging
from pathlib import Path

import chromadb
import numpy as np
import pytest

from aatm import built_stores
from aatm.built_stores import (
    NoVectorDatabaseError,
    VectorDatabaseNotBuiltError,
    built_vector_databases,
    default_retriever_name,
    record_vector_database,
    recorded_vector_databases,
)
from aatm.local_database_utils import release_chromadb_clients
from aatm.omop.coding import AgenticCoder
from aatm.registries import retrievers as registry
from aatm.retrievers import ChromaDBRetriever
from aatm.terminology_mapper import TerminologyMapper
from test_concept_search import ScriptedLLM, vocab_db  # noqa: F401 - fixture


class FakeQueryEmbedding(chromadb.EmbeddingFunction):
    def __init__(self, model=None, *args, **kwargs):
        pass

    def __call__(self, input):
        return [np.ones(3, dtype=np.float32) for _ in input]


MODELS = ("fake-small", "fake-large")


@pytest.fixture
def project(tmp_path, monkeypatch):
    """An empty project that knows only the fake models."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        registry,
        "CHROMADB_RETRIEVER_MODEL_REGISTRY",
        {
            name: registry.ChromaRetrieverModelSpec(
                name=name,
                model_id=name,
                embedding_function_cls=FakeQueryEmbedding,  # type: ignore[arg-type]
            )
            for name in MODELS
        },
    )
    yield tmp_path
    release_chromadb_clients()


def build(name: str, n: int = 2) -> Path:
    """Build a local store of ``n`` expressions for a fake model."""
    path = Path(".aatm/chroma_vector_dbs") / name
    client = chromadb.PersistentClient(path=str(path))
    collection = client.create_collection("expressions", embedding_function=None)
    if n:
        collection.add(
            ids=[str(i) for i in range(n)],
            embeddings=np.ones((n, 3), dtype=np.float32),
            documents=[f"expression {i}" for i in range(n)],
        )
    del collection, client
    release_chromadb_clients()
    return path


# --- the record ------------------------------------------------------------------


def test_records_are_read_newest_first(project):
    record_vector_database("fake-small", "local", Path(".aatm/chroma_vector_dbs/x"))
    record_vector_database("fake-large", "cloud", "expressions-fake-large")
    record_vector_database("fake-small", "local", ".aatm/chroma_vector_dbs/fake-small")

    stores = recorded_vector_databases()
    assert [(s.embedding_model, s.vector_db) for s in stores] == [
        ("fake-small", "local"),
        ("fake-large", "cloud"),
    ]
    assert stores[0].location == ".aatm/chroma_vector_dbs/fake-small"  # replaced
    assert [s.embedding_model for s in recorded_vector_databases("cloud")] == [
        "fake-large"
    ]
    assert "vector_databases:" in built_stores.CONFIG_PATH.read_text()


def test_a_broken_config_is_ignored(project):
    built_stores.CONFIG_PATH.write_text("vector_databases: [unclosed")
    assert recorded_vector_databases() == []


# --- the default -----------------------------------------------------------------


def test_the_default_is_the_recorded_model(project, caplog):
    build("fake-small")
    build("fake-large", n=3)
    record_vector_database("fake-small", "local", ".aatm/chroma_vector_dbs/fake-small")
    record_vector_database("fake-large", "local", ".aatm/chroma_vector_dbs/fake-large")

    with caplog.at_level(logging.INFO, logger="aatm.built_stores"):
        assert default_retriever_name() == "fake-large"
    assert "most recently built" in caplog.text
    assert "fake-small" in caplog.text

    retriever = registry.load_retriever()
    assert retriever.collection.count() == 3


def test_a_record_of_another_vector_database_is_not_the_default(project):
    build("fake-small")
    record_vector_database("fake-large", "cloud", "expressions-fake-large")

    # Nothing recorded locally: the only local store.
    assert default_retriever_name("local") == "fake-small"


def test_without_a_record_the_only_built_store_is_the_default(project):
    build("fake-small")
    assert default_retriever_name() == "fake-small"


def test_without_a_record_several_built_stores_are_a_choice(project):
    build("fake-small")
    build("fake-large")

    with pytest.raises(VectorDatabaseNotBuiltError, match="Several") as error:
        default_retriever_name()
    assert not isinstance(error.value, NoVectorDatabaseError)
    assert "fake-small  local" in str(error.value)
    assert "fake-large  local" in str(error.value)


def test_with_nothing_built_there_is_no_default(project):
    with pytest.raises(NoVectorDatabaseError, match="aatm init"):
        default_retriever_name()


# --- refusing an unbuilt store ---------------------------------------------------


def test_an_unbuilt_model_lists_the_built_stores(project):
    build("fake-small", n=2512)

    with pytest.raises(VectorDatabaseNotBuiltError) as error:
        registry.load_retriever("fake-large")

    assert str(error.value) == (
        "No vector database for 'fake-large' "
        "(.aatm/chroma_vector_dbs/fake-large not found).\n"
        "Built and ready:\n"
        "  - fake-small  local  .aatm/chroma_vector_dbs/fake-small  "
        "(2,512 expressions)\n"
        'Use retriever="fake-small", or build it with: '
        "aatm init --embedding-model fake-large"
    )
    assert not Path(".aatm/chroma_vector_dbs/fake-large").exists()


def test_an_empty_store_is_not_built(project):
    build("fake-small", n=0)

    with pytest.raises(VectorDatabaseNotBuiltError, match="empty collection"):
        registry.load_retriever("fake-small")
    assert built_vector_databases() == []


def test_with_nothing_built_the_error_says_how_to_build(project):
    with pytest.raises(
        VectorDatabaseNotBuiltError, match="No vector database is built"
    ):
        registry.load_retriever("fake-small")


def test_the_retriever_does_not_create_a_collection(project):
    client = chromadb.PersistentClient(path=str(project / "db"))

    with pytest.raises(
        VectorDatabaseNotBuiltError, match="no collection 'expressions'"
    ):
        ChromaDBRetriever(client, "expressions", FakeQueryEmbedding())
    assert client.list_collections() == []
    del client


# --- the mapper and the coder ----------------------------------------------------


def test_the_mapper_defaults_to_the_built_store(project):
    build("fake-small")

    mapper = TerminologyMapper(output_dir=project / "out")

    assert isinstance(mapper.retriever, ChromaDBRetriever)
    assert mapper.retriever.collection.count() == 2


def test_the_mapper_needs_a_built_store(project):
    with pytest.raises(NoVectorDatabaseError):
        TerminologyMapper(output_dir=project / "out")


def test_the_coder_defaults_to_the_built_store(project, vocab_db):  # noqa: F811
    build("fake-small")

    coder = AgenticCoder(ScriptedLLM([]), vocabulary=vocab_db)

    assert isinstance(coder.agent.tools.retriever, ChromaDBRetriever)


def test_without_a_store_the_coder_searches_the_vocabulary(
    project,
    vocab_db,  # noqa: F811
    caplog,
):
    with caplog.at_level(logging.WARNING, logger="aatm.omop.coding"):
        coder = AgenticCoder(ScriptedLLM([]), vocabulary=vocab_db)
    assert coder.agent.tools.retriever is None
    assert "vocabulary tools only" in caplog.text

    build("fake-small")
    coder = AgenticCoder(ScriptedLLM([]), retriever=False, vocabulary=vocab_db)
    assert coder.agent.tools.retriever is None


def test_a_store_sqlite_cannot_count_is_opened_with_chromadb(project, monkeypatch):
    build("fake-small", n=2)
    monkeypatch.setattr(built_stores, "_local_count", lambda path, name: None)

    (store,) = built_vector_databases()
    assert store.count == 2


def test_a_store_that_cannot_be_opened_is_not_built(project, monkeypatch):
    build("fake-small")
    build("fake-large")

    def broken(path, name):
        if path.endswith("fake-large"):
            raise RuntimeError("Error loading hnsw index")
        return 2

    monkeypatch.setattr(built_stores, "_local_count", broken)

    assert [s.embedding_model for s in built_vector_databases()] == ["fake-small"]
    with pytest.raises(VectorDatabaseNotBuiltError, match="could not be opened"):
        registry.load_retriever("fake-large")
