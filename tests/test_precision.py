import pytest

from aatm import precision
from aatm.precision import (
    EMBEDDING_DTYPE_ENV_VAR,
    EmbeddingPrecisionError,
    dtype_for_existing_collection,
    dtype_for_queries,
    resolve_embedding_dtype,
)


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    monkeypatch.delenv(EMBEDDING_DTYPE_ENV_VAR, raising=False)


@pytest.fixture
def bf16_gpu(monkeypatch):
    monkeypatch.setattr(
        precision.torch.cuda, "is_bf16_supported", lambda including_emulation=True: True
    )


def test_default_is_full_precision():
    assert resolve_embedding_dtype(None, device="cuda") == "float32"
    assert resolve_embedding_dtype(None, device="cpu") == "float32"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("bf16", "bfloat16"),
        ("BFloat16", "bfloat16"),
        ("fp16", "float16"),
        ("fp32", "float32"),
    ],
)
def test_aliases_are_normalized(bf16_gpu, value, expected):
    assert resolve_embedding_dtype(value, device="cuda") == expected


def test_environment_variable_is_used_when_no_argument(bf16_gpu, monkeypatch):
    monkeypatch.setenv(EMBEDDING_DTYPE_ENV_VAR, "bfloat16")
    assert resolve_embedding_dtype(None, device="cuda") == "bfloat16"
    assert resolve_embedding_dtype("float32", device="cuda") == "float32"


def test_unknown_precision_is_rejected():
    with pytest.raises(EmbeddingPrecisionError, match="Unsupported"):
        resolve_embedding_dtype("int8", device="cuda")


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_half_precision_requires_cuda(device):
    with pytest.raises(EmbeddingPrecisionError, match="NVIDIA GPU"):
        resolve_embedding_dtype("float16", device=device)


def test_bfloat16_requires_native_support(monkeypatch):
    monkeypatch.setattr(
        precision.torch.cuda,
        "is_bf16_supported",
        lambda including_emulation=True: False,
    )
    with pytest.raises(EmbeddingPrecisionError, match="bfloat16"):
        resolve_embedding_dtype("bfloat16", device="cuda")


def test_repair_uses_stored_precision_when_none_requested():
    assert dtype_for_existing_collection("bfloat16", None) == "bfloat16"


def test_repair_with_same_precision_is_allowed():
    assert dtype_for_existing_collection("float32", "float32") == "float32"


def test_repair_with_different_precision_is_rejected():
    with pytest.raises(EmbeddingPrecisionError, match="built with float32"):
        dtype_for_existing_collection("float32", "bfloat16")


def test_repair_of_untracked_database_is_rejected():
    with pytest.raises(EmbeddingPrecisionError, match="Overwrite"):
        dtype_for_existing_collection(None, None)


def test_queries_use_stored_precision(bf16_gpu):
    assert dtype_for_queries("bfloat16", device="cuda") == "bfloat16"
    assert dtype_for_queries(None, device="cuda") is None


def test_queries_fall_back_to_float32_off_gpu():
    assert dtype_for_queries("bfloat16", device="cpu") == "float32"
