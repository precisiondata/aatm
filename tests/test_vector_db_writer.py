import chromadb
import numpy as np
import pytest

from aatm.local_database_utils import release_chromadb_clients
from aatm.vector_db_writer import ChromaWriterProcess, VectorDatabaseWriterError

COLLECTION = "expressions"


@pytest.fixture
def db_path(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection(COLLECTION, embedding_function=None)
    collection.add(ids=["existing"], embeddings=np.ones((1, 3), dtype=np.float32))
    del collection, client
    release_chromadb_clients()
    return str(tmp_path)


def read_back(path):
    collection = chromadb.PersistentClient(path=path).get_collection(COLLECTION)
    try:
        return collection.get(include=["embeddings"])
    finally:
        del collection
        release_chromadb_clients()


def vectors(n, value=1.0):
    return np.full((n, 3), value, dtype=np.float32)


def test_writes_batches_and_acknowledges_them(db_path):
    with ChromaWriterProcess(db_path, COLLECTION) as writer:
        assert writer.existing_ids(["existing", "new-1"]) == {"existing"}
        writer.add(1, ["new-1", "new-2"], vectors(2), ["a", "b"], [{"k": 1}, {"k": 2}])
        writer.add(2, ["new-3"], vectors(1), ["c"], [{"k": 3}])
        acks = writer.poll() + writer.close()

    assert sorted((ack.seq, ack.n_added) for ack in acks) == [(1, 2), (2, 1)]
    assert sorted(read_back(db_path)["ids"]) == ["existing", "new-1", "new-2", "new-3"]


def test_lookups_see_earlier_writes(db_path):
    with ChromaWriterProcess(db_path, COLLECTION) as writer:
        writer.add(1, ["new-1"], vectors(1), ["a"], [{"k": 1}])
        writer.close()

    with ChromaWriterProcess(db_path, COLLECTION) as writer:
        assert writer.existing_ids(["new-1", "new-2"]) == {"new-1"}


def test_duplicate_ids_keep_the_first_version(db_path):
    # The pipeline relies on this: a row encoded twice must not change the database.
    with ChromaWriterProcess(db_path, COLLECTION) as writer:
        writer.add(1, ["existing"], vectors(1, value=0.0), ["x"], [{"k": 0}])
        writer.close()

    result = read_back(db_path)
    stored = dict(zip(result["ids"], result["embeddings"]))
    assert len(result["ids"]) == 1
    np.testing.assert_allclose(stored["existing"], [1.0, 1.0, 1.0])


def test_write_errors_are_raised_in_the_parent(db_path):
    with pytest.raises(VectorDatabaseWriterError, match="dimension"):
        with ChromaWriterProcess(db_path, COLLECTION) as writer:
            wrong_dimension = np.ones((1, 5), dtype=np.float32)
            writer.add(1, ["bad"], wrong_dimension, ["x"], [{"k": 0}])
            writer.close()


def test_missing_collection_is_raised_in_the_parent(tmp_path):
    with pytest.raises(VectorDatabaseWriterError):
        with ChromaWriterProcess(str(tmp_path), "does-not-exist") as writer:
            writer.existing_ids(["a"])
