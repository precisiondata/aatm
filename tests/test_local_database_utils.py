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
