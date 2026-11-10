from typing import Any

import numpy as np
import pytest

from aatm import embedding_functions as ef
from aatm import precision


class FakeSentenceTransformer:
    """Stand-in for SentenceTransformer that records how it was used."""

    instances: list["FakeSentenceTransformer"] = []

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.init_kwargs = kwargs
        self.encode_calls: list[dict[str, Any]] = []
        FakeSentenceTransformer.instances.append(self)

    def eval(self) -> "FakeSentenceTransformer":
        return self

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.encode_calls.append(kwargs)
        return np.ones((len(texts), 3), dtype=np.float32)


@pytest.fixture
def fake_st(monkeypatch):
    FakeSentenceTransformer.instances = []
    monkeypatch.setattr(ef, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(ef, "get_device", lambda: "cpu")
    monkeypatch.delenv("AATM_EMBEDDING_DTYPE", raising=False)
    # Checkpoints are "stored" in bfloat16 unless a test says otherwise.
    monkeypatch.setattr(precision, "checkpoint_dtype", lambda model_id: "bfloat16")
    monkeypatch.setattr(
        precision.torch.cuda,
        "is_bf16_supported",
        lambda including_emulation=True: True,
    )
    return FakeSentenceTransformer


@pytest.mark.parametrize("cls", [ef.Qwen3EmbeddingFunction, ef.GemmaEmbeddingFunction])
def test_local_models_are_marked_local_before_instantiation(cls):
    # The vector database builder checks this on the class, before loading weights.
    assert cls.is_local is True


@pytest.mark.parametrize(
    ("cls", "model_id"),
    [
        (ef.Qwen3EmbeddingFunction, ef.Qwen3EmbeddingModels.QWEN3_06B.value),
        (ef.GemmaEmbeddingFunction, ef.GemmaEmbeddingModels.EMBEDDING_GEMMA_300M.value),
    ],
)
def test_local_models_use_configured_batch_size(fake_st, cls, model_id):
    function = cls(model=model_id, batch_size=7)
    function.batch_size = 128  # the builder overwrites it after estimating

    embeddings = function(["a", "b"])

    assert len(embeddings) == 2
    assert fake_st.instances[0].encode_calls[0]["batch_size"] == 128


def test_qwen3_encodes_with_query_prompt(fake_st):
    function = ef.Qwen3EmbeddingFunction(model=ef.Qwen3EmbeddingModels.QWEN3_06B.value)
    function(["a"])
    assert fake_st.instances[0].encode_calls[0]["prompt_name"] == "query"


def test_gemma_encodes_without_prompt(fake_st):
    function = ef.GemmaEmbeddingFunction(
        model=ef.GemmaEmbeddingModels.EMBEDDING_GEMMA_300M.value
    )
    function(["a"])
    assert "prompt_name" not in fake_st.instances[0].encode_calls[0]


@pytest.mark.parametrize(
    ("cls", "model_id"),
    [
        (ef.Qwen3EmbeddingFunction, ef.Qwen3EmbeddingModels.QWEN3_06B.value),
        (ef.GemmaEmbeddingFunction, ef.GemmaEmbeddingModels.EMBEDDING_GEMMA_300M.value),
    ],
)
def test_local_models_load_without_key_value_cache(fake_st, cls, model_id):
    # The cache only helps generation; for embeddings it wastes GPU memory.
    cls(model=model_id)
    assert fake_st.instances[0].init_kwargs["config_kwargs"] == {"use_cache": False}


def test_unknown_model_id_is_rejected(fake_st):
    with pytest.raises(ValueError):
        ef.Qwen3EmbeddingFunction(model="not-a-model")


def test_models_load_in_checkpoint_precision_by_default(fake_st, monkeypatch):
    monkeypatch.setattr(ef, "get_device", lambda: "cuda")
    function = ef.Qwen3EmbeddingFunction(model=ef.Qwen3EmbeddingModels.QWEN3_06B.value)
    assert function.dtype == "bfloat16"
    assert (
        fake_st.instances[0].init_kwargs["model_kwargs"]["dtype"] is ef.torch.bfloat16
    )


def test_half_precision_checkpoint_loads_in_float32_on_cpu(fake_st):
    function = ef.Qwen3EmbeddingFunction(model=ef.Qwen3EmbeddingModels.QWEN3_06B.value)
    assert function.dtype == "float32"
    assert fake_st.instances[0].init_kwargs["model_kwargs"]["dtype"] is ef.torch.float32


def test_models_load_in_requested_half_precision(fake_st, monkeypatch):
    monkeypatch.setattr(ef, "get_device", lambda: "cuda")
    function = ef.Qwen3EmbeddingFunction(
        model=ef.Qwen3EmbeddingModels.QWEN3_06B.value, dtype="float16"
    )
    assert function.dtype == "float16"
    assert fake_st.instances[0].init_kwargs["model_kwargs"]["dtype"] is ef.torch.float16


def test_gemma_rejects_float16(fake_st, monkeypatch):
    monkeypatch.setattr(ef, "get_device", lambda: "cuda")
    with pytest.raises(ef.EmbeddingPrecisionError, match="does not support float16"):
        ef.GemmaEmbeddingFunction(
            model=ef.GemmaEmbeddingModels.EMBEDDING_GEMMA_300M.value, dtype="float16"
        )
    assert fake_st.instances == []  # rejected before loading weights
