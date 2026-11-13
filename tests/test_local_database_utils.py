from datetime import datetime, timezone

import numpy as np
from chromadb import EmbeddingFunction
import pytest

from aatm import local_database_utils as ldu
from aatm.precision import EmbeddingPrecisionError


def test_collect_calibration_texts_ranks_longest_by_tokens(tmp_path):
    import pandas as pd

    path = tmp_path / "standard_concept.csv"
    expressions = ["a" * 50, "b" * 10, "word " * 3, "c" * 30, "d"]
    pd.DataFrame({"expression": expressions + [None, "d"]}).to_csv(path, index=False)

    # Pretend each space-separated word is a token: "word word word " is the
    # longest in tokens although not in characters.
    def token_lengths(texts):
        return [len(text.split()) * 100 + len(text) for text in texts]

    longest, typical = ldu.collect_calibration_texts(
        [path], n_longest=2, n_typical=10, token_lengths=token_lengths
    )

    assert longest == ["word " * 3, "a" * 50]
    assert sorted(typical) == sorted(expressions)  # deduplicated, no missing values


@pytest.mark.parametrize(
    ("requested", "is_local", "expected"),
    [
        (None, True, ldu.LOCAL_MODEL_CHUNK_SIZE),
        (None, False, ldu.API_MODEL_CHUNK_SIZE),
        (500, True, 500),
        (500, False, 500),
        (100_000, True, 5461),  # capped at the ChromaDB limit
    ],
)
def test_resolve_chunk_size(requested, is_local, expected):
    assert (
        ldu.resolve_chunk_size(
            requested, is_local=is_local, max_chromadb_batch_size=5461
        )
        == expected
    )


def test_resolve_chunk_size_rejects_non_positive():
    with pytest.raises(ValueError):
        ldu.resolve_chunk_size(0, is_local=True, max_chromadb_batch_size=5461)


DATASET_COLUMNS = [
    "expression",
    "expression_concept_id",
    "std_concept_id",
    "std_concept_name",
    "std_vocabulary_id",
    "std_vocabulary_code",
    "std_domain_id",
]


def write_dataset(path, n_rows, start=0):
    import pandas as pd

    rows = [
        [
            f"expression {i}",
            1000 + i,
            2000 + i,
            f"concept {i}",
            "SNOMED",
            str(i),
            "Condition",
        ]
        for i in range(start, start + n_rows)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=DATASET_COLUMNS).to_csv(path, index=False)


def test_prepare_chunks_skips_existing_ids_and_reports_every_chunk(tmp_path):
    dataset = tmp_path / "standard_concept.csv"
    write_dataset(dataset, n_rows=5)
    seen_lookups = []

    def existing_ids(ids):
        seen_lookups.append(ids)
        return set(ids[:2])  # pretend the first two rows are stored already

    chunks = list(
        ldu.prepare_chunks(
            [dataset],
            chunk_size=3,
            existing_ids=existing_ids,
            profiler=ldu.TimingProfiler(enabled=False),
        )
    )

    assert [(c.row_start, c.n_rows) for c in chunks] == [(0, 3), (3, 2)]
    assert all(c.dataset_rows == 5 for c in chunks)
    assert chunks[0].documents == ["expression 2"]
    assert chunks[1].ids == []  # both rows "exist"; still yielded for progress
    assert len(chunks[0].ids) == len(chunks[0].metadatas) == 1


class FakeEmbeddingFunction(EmbeddingFunction):
    """CPU-only stand-in that records every text it embeds."""

    calls: list[list[str]] = []

    def __init__(self, model, *args, **kwargs):
        pass

    def __call__(self, input):
        FakeEmbeddingFunction.calls.append(list(input))
        return [np.full(4, float(len(text)), dtype=np.float32) for text in input]


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    from aatm.registries import retrievers

    monkeypatch.chdir(tmp_path)
    FakeEmbeddingFunction.calls = []
    spec = retrievers.ChromaRetrieverModelSpec(
        name="fake-model",
        model_id="fake-model",
        embedding_function_cls=FakeEmbeddingFunction,  # type: ignore[arg-type]
    )
    monkeypatch.setitem(
        retrievers.CHROMADB_RETRIEVER_MODEL_REGISTRY, "fake-model", spec
    )
    write_dataset(tmp_path / ".aatm/datasets/standard_concept.csv", n_rows=250)
    write_dataset(tmp_path / ".aatm/datasets/standard_concept_synonym.csv", n_rows=30)
    return tmp_path / spec.chromadb_path


def stored_ids(db_path):
    import chromadb

    collection = chromadb.PersistentClient(path=str(db_path)).get_collection(
        "expressions"
    )
    try:
        return collection.get(include=[])["ids"]
    finally:
        del collection
        ldu.release_chromadb_clients()


def answer_prompt(monkeypatch, answer):
    class Prompt:
        def ask(self):
            return answer

    monkeypatch.setattr(ldu.questionary, "select", lambda *a, **k: Prompt())


def test_build_writes_every_row_once(fake_project):
    ldu.build_local_vector_database("fake-model", chunk_size=100)

    ids = stored_ids(fake_project)
    assert len(ids) == len(set(ids)) == 280
    assert sum(len(call) for call in FakeEmbeddingFunction.calls) == 280


def test_build_records_the_database(fake_project, monkeypatch):
    from aatm.built_stores import recorded_vector_databases

    ldu.build_local_vector_database("fake-model", chunk_size=100)

    (store,) = recorded_vector_databases()
    assert store.embedding_model == "fake-model"
    assert store.vector_db == "local"
    assert store.location == ".aatm/chroma_vector_dbs/fake-model"
    assert store.built_at

    # Keeping the existing database keeps its build date.
    later = datetime(2100, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "aatm.built_stores.datetime",
        type("Later", (), {"now": staticmethod(lambda tz: later)}),
    )
    answer_prompt(monkeypatch, "Skip")
    ldu.build_local_vector_database("fake-model", chunk_size=100)
    assert recorded_vector_databases() == [store]


def test_repair_only_embeds_missing_rows(fake_project, monkeypatch):
    ldu.build_local_vector_database("fake-model", chunk_size=100)
    write_dataset(
        fake_project.parent.parent / "datasets/standard_concept_synonym.csv",
        n_rows=40,
    )  # 10 new rows
    FakeEmbeddingFunction.calls = []
    answer_prompt(monkeypatch, "Repair")

    ldu.build_local_vector_database("fake-model", chunk_size=100)

    assert sum(len(call) for call in FakeEmbeddingFunction.calls) == 10
    assert len(stored_ids(fake_project)) == 290


def test_errors_while_encoding_stop_the_build(fake_project, monkeypatch):
    def broken(self, input):
        raise RuntimeError("GPU fell over")

    monkeypatch.setattr(FakeEmbeddingFunction, "__call__", broken)

    with pytest.raises(RuntimeError, match="GPU fell over"):
        ldu.build_local_vector_database("fake-model", chunk_size=100)


class FakeLocalModel(FakeEmbeddingFunction):
    """Local-model stand-in that records the precision it was loaded in."""

    is_local = True
    loaded_dtypes: list[str] = []

    def __init__(self, model, dtype="float32", *args, **kwargs):
        FakeLocalModel.loaded_dtypes.append(dtype)
        FakeLocalModel.last = self
        self.batch_size = 8
        self.max_batch_tokens = None


@pytest.fixture
def fake_local_project(tmp_path, monkeypatch):
    from aatm import precision
    from aatm.registries import retrievers

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AATM_EMBEDDING_DTYPE", raising=False)
    # Pretend to be a bfloat16-capable GPU so half precision passes validation.
    monkeypatch.setattr(ldu, "get_device", lambda: "cuda")
    monkeypatch.setattr(
        precision.torch.cuda,
        "is_bf16_supported",
        lambda including_emulation=True: True,
    )
    # The fake checkpoint is stored in float32, like EmbeddingGemma.
    monkeypatch.setattr(precision, "checkpoint_dtype", lambda model_id: "float32")
    FakeLocalModel.loaded_dtypes = []
    spec = retrievers.ChromaRetrieverModelSpec(
        name="fake-local",
        model_id="fake-local",
        embedding_function_cls=FakeLocalModel,  # type: ignore[arg-type]
    )
    monkeypatch.setitem(
        retrievers.CHROMADB_RETRIEVER_MODEL_REGISTRY, "fake-local", spec
    )
    write_dataset(tmp_path / ".aatm/datasets/standard_concept.csv", n_rows=20)
    return tmp_path / spec.chromadb_path


def stored_dtype(db_path):
    import chromadb

    collection = chromadb.PersistentClient(path=str(db_path)).get_collection(
        "expressions"
    )
    try:
        return (collection.metadata or {}).get("aatm_embedding_dtype")
    finally:
        del collection
        ldu.release_chromadb_clients()


def build_local(**kwargs):
    ldu.build_local_vector_database("fake-local", encode_batch_size=8, **kwargs)


def test_new_database_defaults_to_checkpoint_precision(fake_local_project, monkeypatch):
    from aatm import precision

    monkeypatch.setattr(precision, "checkpoint_dtype", lambda model_id: "bfloat16")

    build_local()

    assert FakeLocalModel.loaded_dtypes == ["bfloat16"]
    assert stored_dtype(fake_local_project) == "bfloat16"


def test_float32_checkpoint_builds_in_float32(fake_local_project):
    build_local()
    assert FakeLocalModel.loaded_dtypes == ["float32"]
    assert stored_dtype(fake_local_project) == "float32"


def test_new_database_records_requested_half_precision(fake_local_project):
    build_local(embedding_dtype="bf16")
    assert FakeLocalModel.loaded_dtypes == ["bfloat16"]
    assert stored_dtype(fake_local_project) == "bfloat16"


def test_repair_with_different_precision_fails_before_loading_model(
    fake_local_project, monkeypatch
):
    build_local()
    answer_prompt(monkeypatch, "Repair")
    FakeLocalModel.loaded_dtypes = []

    with pytest.raises(EmbeddingPrecisionError, match="built with float32"):
        build_local(embedding_dtype="bfloat16")

    assert FakeLocalModel.loaded_dtypes == []
    assert stored_dtype(fake_local_project) == "float32"


def test_repair_with_env_var_mismatch_fails(fake_local_project, monkeypatch):
    build_local()
    answer_prompt(monkeypatch, "Repair")
    monkeypatch.setenv("AATM_EMBEDDING_DTYPE", "float16")

    with pytest.raises(EmbeddingPrecisionError):
        build_local()


def test_repair_without_request_uses_stored_precision(fake_local_project, monkeypatch):
    build_local(embedding_dtype="bfloat16")
    answer_prompt(monkeypatch, "Repair")
    FakeLocalModel.loaded_dtypes = []

    build_local()

    assert FakeLocalModel.loaded_dtypes == ["bfloat16"]


def test_overwrite_may_change_precision(fake_local_project, monkeypatch):
    build_local()
    answer_prompt(monkeypatch, "Overwrite")

    build_local(embedding_dtype="bfloat16")

    assert stored_dtype(fake_local_project) == "bfloat16"


def test_repair_of_untracked_database_fails(fake_local_project, monkeypatch):
    import chromadb

    client = chromadb.PersistentClient(path=str(fake_local_project))
    client.create_collection("expressions", embedding_function=None)  # no metadata
    del client
    ldu.release_chromadb_clients()
    answer_prompt(monkeypatch, "Repair")

    with pytest.raises(EmbeddingPrecisionError, match="before AATM recorded"):
        build_local()


def test_embedding_dtype_is_rejected_for_api_models(fake_project):
    with pytest.raises(ValueError, match="only applies to local models"):
        ldu.build_local_vector_database("fake-model", embedding_dtype="bfloat16")


def test_token_budget_is_passed_to_local_model(fake_local_project):
    ldu.build_local_vector_database("fake-local", encode_max_batch_tokens=4096)
    assert FakeLocalModel.last.max_batch_tokens == 4096


def test_fixed_batch_size_turns_off_token_budget(fake_local_project):
    build_local()
    assert FakeLocalModel.last.batch_size == 8
    assert FakeLocalModel.last.max_batch_tokens is None


def test_batch_size_and_token_budget_are_exclusive(fake_local_project):
    with pytest.raises(ValueError, match="not both"):
        ldu.build_local_vector_database(
            "fake-local", encode_batch_size=8, encode_max_batch_tokens=4096
        )


# Chroma Cloud (simulated by a local database behind a cloud-mode connection)


@pytest.fixture
def cloud_project(fake_project, monkeypatch):
    from aatm.registries import retrievers
    from fake_cloud import fake_cloud

    cloud_dir = fake_project.parent / "fake-cloud"
    cloud_dir.mkdir(parents=True)

    def chroma_location(self, vector_db=None, local_path=None):
        assert vector_db == "cloud"
        return fake_cloud(cloud_dir), self.cloud_collection_name

    monkeypatch.setattr(
        retrievers.ChromaRetrieverModelSpec, "chroma_location", chroma_location
    )
    return cloud_dir


def cloud_ids(cloud_dir):
    import chromadb

    collection = chromadb.PersistentClient(path=str(cloud_dir)).get_collection(
        "expressions-fake-model"
    )
    try:
        return collection.get(include=[])["ids"]
    finally:
        del collection
        ldu.release_chromadb_clients()


def test_cloud_build_writes_to_the_model_collection(cloud_project, fake_project):
    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="cloud")

    ids = cloud_ids(cloud_project)
    assert len(ids) == len(set(ids)) == 280
    assert not fake_project.exists()  # nothing built locally

    from aatm.built_stores import recorded_vector_databases

    (store,) = recorded_vector_databases("cloud")
    assert (store.embedding_model, store.location) == (
        "fake-model",
        "expressions-fake-model",
    )


def test_cloud_overwrite_deletes_the_collection(cloud_project, monkeypatch):
    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="cloud")
    FakeEmbeddingFunction.calls = []
    answer_prompt(monkeypatch, "Overwrite")

    ldu.build_local_vector_database("fake-model", chunk_size=100, vector_db="cloud")

    assert sum(len(call) for call in FakeEmbeddingFunction.calls) == 280
    assert len(cloud_ids(cloud_project)) == 280


def test_cloud_build_rejects_a_local_directory(cloud_project, tmp_path):
    with pytest.raises(ValueError, match="local vector database"):
        ldu.build_local_vector_database(
            "fake-model", vector_db_dir=tmp_path, vector_db="cloud"
        )


def test_upload_copies_the_local_index_without_re_embedding(fake_project, tmp_path):
    import chromadb

    from fake_cloud import fake_cloud

    ldu.build_local_vector_database("fake-model", chunk_size=100)
    FakeEmbeddingFunction.calls = []
    cloud_dir = tmp_path / "fake-cloud"

    summary = ldu.upload_vector_database_to_cloud(
        "fake-model",
        cloud_connection=fake_cloud(cloud_dir),
        read_page_size=64,
        write_batch_size=50,
    )

    assert (summary.total, summary.uploaded, summary.skipped) == (280, 280, 0)
    assert FakeEmbeddingFunction.calls == []

    local = chromadb.PersistentClient(path=str(fake_project)).get_collection(
        "expressions"
    )
    cloud = chromadb.PersistentClient(path=str(cloud_dir)).get_collection(
        "expressions-fake-model"
    )
    try:
        expected = local.get(include=["embeddings", "documents", "metadatas"])
        actual = cloud.get(
            ids=expected["ids"], include=["embeddings", "documents", "metadatas"]
        )
        by_id = {
            identifier: (list(vector), document, metadata)
            for identifier, vector, document, metadata in zip(
                actual["ids"],
                actual["embeddings"],
                actual["documents"],
                actual["metadatas"],
            )
        }
        for identifier, vector, document, metadata in zip(
            expected["ids"],
            expected["embeddings"],
            expected["documents"],
            expected["metadatas"],
        ):
            assert by_id[identifier] == (list(vector), document, metadata)
        assert cloud.metadata == local.metadata
    finally:
        del local, cloud
        ldu.release_chromadb_clients()


def test_upload_resumes_by_skipping_rows_already_in_the_cloud(fake_project, tmp_path):
    from fake_cloud import fake_cloud

    ldu.build_local_vector_database("fake-model", chunk_size=100)
    cloud = fake_cloud(tmp_path / "fake-cloud")
    ldu.upload_vector_database_to_cloud("fake-model", cloud_connection=cloud)

    summary = ldu.upload_vector_database_to_cloud("fake-model", cloud_connection=cloud)

    assert (summary.uploaded, summary.skipped) == (0, 280)


def test_upload_needs_a_local_database(fake_project, tmp_path):
    from fake_cloud import fake_cloud

    with pytest.raises(ValueError, match="No local vector database"):
        ldu.upload_vector_database_to_cloud(
            "fake-model", cloud_connection=fake_cloud(tmp_path / "c")
        )
