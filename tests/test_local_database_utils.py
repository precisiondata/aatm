import numpy as np
from chromadb import EmbeddingFunction
import pytest

from aatm import local_database_utils as ldu


class FakeLocalEmbeddingFunction:
    is_local = True

    def __init__(self) -> None:
        self.model = object()


@pytest.fixture
def fake_gpu(monkeypatch):
    """Simulate a GPU with 10 GiB free where each batch item costs 0.1 GiB."""
    gib = 1024**3
    monkeypatch.setattr(ldu.torch.cuda, "mem_get_info", lambda: (10 * gib, 24 * gib))
    monkeypatch.setattr(ldu, "gpu_memory_cleanup", lambda **_: None)

    measured_models = []

    def fake_measure(model, batch_size, seq_len, device="cuda"):
        measured_models.append(model)
        return {"peak_reserved_gib": 1.0 + 0.1 * batch_size}

    monkeypatch.setattr(ldu, "measure_embedding_vram", fake_measure)
    return measured_models


def test_estimate_batch_size_reuses_the_loaded_model(fake_gpu):
    function = FakeLocalEmbeddingFunction()

    batch_size = ldu.estimate_batch_size(function)

    # (10 GiB free - 1.5 GiB overhead) / 0.1 GiB per item
    assert batch_size == 85
    assert fake_gpu and all(model is function.model for model in fake_gpu)


def test_estimate_batch_size_rejects_non_local_models(fake_gpu):
    class ApiFunction:
        is_local = False

    with pytest.raises(ValueError, match="not a local embedding model"):
        ldu.estimate_batch_size(ApiFunction())


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
