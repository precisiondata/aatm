"""
Choose the numeric precision local embedding models run in.

By default, local embedding models run in the precision their checkpoint is
stored in, read from the model's ``config.json``: ``bfloat16`` for Qwen3
Embedding, ``float32`` for EmbeddingGemma. If this machine cannot run the
checkpoint's precision (half precision off an NVIDIA GPU, or ``bfloat16`` on a
GPU without native support), the model runs in ``float32`` instead.

You can choose another precision. Half precision roughly halves memory use and
encodes faster on an NVIDIA GPU:

- ``bfloat16``: needs an Ampere or newer GPU (RTX 30xx, A100 and later).
- ``float16``: works on older GPUs too, but some models produce unstable
  embeddings in it (EmbeddingGemma does not support it).

The precision is chosen, in order of priority, by:

1. the ``embedding_dtype`` argument of `build_local_vector_database` (or
   ``aatm init --embedding-dtype``);
2. the ``AATM_EMBEDDING_DTYPE`` environment variable;
3. the checkpoint's precision, as described above.

Half precision changes the embeddings slightly, so one vector database must not
mix precisions. The builder records the precision in the collection's metadata
and refuses to repair a database with a different one. When AATM loads a
retriever, it encodes queries in the precision the database was built with.

```python
from aatm.precision import resolve_embedding_dtype

resolve_embedding_dtype(None, device="cuda", model_id="Qwen/Qwen3-Embedding-0.6B")
# "bfloat16", the checkpoint's precision, unless AATM_EMBEDDING_DTYPE is set
resolve_embedding_dtype("fp32", device="cuda")  # "float32"
```
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal, cast

import torch

from aatm.logs import get_logger

logger = get_logger(__name__)

EMBEDDING_DTYPE_ENV_VAR = "AATM_EMBEDDING_DTYPE"
"""Environment variable that selects the precision of local embedding models."""

EmbeddingDtype = Literal["float32", "bfloat16", "float16"]
SUPPORTED_EMBEDDING_DTYPES: tuple[EmbeddingDtype, ...] = (
    "float32",
    "bfloat16",
    "float16",
)
FALLBACK_EMBEDDING_DTYPE: EmbeddingDtype = "float32"
"""Precision used when the checkpoint's own cannot be read or run on this machine."""

COLLECTION_DTYPE_KEY = "aatm_embedding_dtype"
"""ChromaDB collection metadata key that records the precision of its embeddings."""

_ALIASES: dict[str, EmbeddingDtype] = {
    "fp32": "float32",
    "bf16": "bfloat16",
    "fp16": "float16",
}

_TORCH_DTYPES: dict[EmbeddingDtype, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class EmbeddingPrecisionError(ValueError):
    """Raised when a precision cannot be used, or would mix precisions in one database."""


def parse_embedding_dtype(value: str) -> EmbeddingDtype:
    """Normalize a precision name such as ``"bf16"`` to its canonical form.

    Args:
        value: Precision name or alias (``fp32``, ``bf16``, ``fp16``).

    Returns:
        The canonical precision name.

    Raises:
        EmbeddingPrecisionError: If the name is not supported.
    """
    name = value.strip().lower()
    name = _ALIASES.get(name, name)
    if name not in SUPPORTED_EMBEDDING_DTYPES:
        raise EmbeddingPrecisionError(
            f"Unsupported embedding precision {value!r}; "
            f"use one of {', '.join(SUPPORTED_EMBEDDING_DTYPES)}."
        )
    return cast(EmbeddingDtype, name)


def requested_embedding_dtype(value: str | None = None) -> EmbeddingDtype | None:
    """Return the precision the user asked for, or ``None`` if they did not choose.

    Args:
        value: Precision passed explicitly. Takes priority over
            ``AATM_EMBEDDING_DTYPE``.

    Returns:
        The canonical precision name, or ``None`` when neither the argument nor
        the environment variable is set.

    Raises:
        EmbeddingPrecisionError: If the value is not supported.
    """
    if value is None:
        value = os.environ.get(EMBEDDING_DTYPE_ENV_VAR, "").strip() or None
    if value is None:
        return None
    return parse_embedding_dtype(value)


def check_dtype_supported(dtype: EmbeddingDtype, device: str) -> None:
    """Check that ``device`` can run a model in ``dtype``.

    Args:
        dtype: Precision to check.
        device: Device the model will run on (``cuda``, ``mps`` or ``cpu``).

    Raises:
        EmbeddingPrecisionError: If half precision is requested off CUDA, or
            ``bfloat16`` on a GPU without native support for it.
    """
    if dtype == "float32":
        return
    if device != "cuda":
        raise EmbeddingPrecisionError(
            f"Embedding precision {dtype!r} is only supported on an NVIDIA GPU "
            f"(CUDA), but local models run on {device!r}. Use 'float32'."
        )
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported(
        including_emulation=False
    ):
        raise EmbeddingPrecisionError(
            "This GPU has no native bfloat16 support. Use 'float16' or 'float32'."
        )


@lru_cache(maxsize=None)
def checkpoint_dtype(model_id: str) -> EmbeddingDtype | None:
    """Return the precision a model's checkpoint is stored in.

    Reads ``dtype`` (``torch_dtype`` in older transformers versions) from the
    model's ``config.json``, using the Hugging Face cache when available.

    Args:
        model_id: Hugging Face model identifier.

    Returns:
        The checkpoint's precision, or ``None`` if the config cannot be read or
        names a precision AATM does not support.
    """
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception as error:  # noqa: BLE001 - any failure means "unknown"
        logger.warning(f"Could not read the precision of {model_id}: {error}")
        return None

    stored = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if stored is None:
        return None

    name = str(stored).removeprefix("torch.")
    try:
        return parse_embedding_dtype(name)
    except EmbeddingPrecisionError:
        logger.warning(f"{model_id} is stored in unsupported precision {name!r}.")
        return None


def default_embedding_dtype(model_id: str | None, device: str) -> EmbeddingDtype:
    """Return the precision to use when the user did not choose one.

    Args:
        model_id: Hugging Face model identifier, or ``None`` if unknown.
        device: Device the model will run on.

    Returns:
        The checkpoint's precision if ``device`` can run it, otherwise
        `FALLBACK_EMBEDDING_DTYPE`.
    """
    dtype = checkpoint_dtype(model_id) if model_id else None
    if dtype is None:
        return FALLBACK_EMBEDDING_DTYPE

    try:
        check_dtype_supported(dtype, device)
    except EmbeddingPrecisionError:
        logger.info(
            f"{model_id} is stored in {dtype}, which {device!r} cannot run; "
            f"using {FALLBACK_EMBEDDING_DTYPE}."
        )
        return FALLBACK_EMBEDDING_DTYPE
    return dtype


def resolve_embedding_dtype(
    value: str | None, device: str, model_id: str | None = None
) -> EmbeddingDtype:
    """Return the precision to load a local model in.

    Args:
        value: Precision passed explicitly, or ``None`` to use
            ``AATM_EMBEDDING_DTYPE`` or the default.
        device: Device the model will run on.
        model_id: Hugging Face model identifier, used to find the default
            precision in the checkpoint.

    Returns:
        The canonical precision name.

    Raises:
        EmbeddingPrecisionError: If a requested precision is unsupported, or the
            device cannot run it.
    """
    requested = requested_embedding_dtype(value)
    if requested is None:
        return default_embedding_dtype(model_id, device)
    check_dtype_supported(requested, device)
    return requested


def to_torch_dtype(dtype: EmbeddingDtype) -> torch.dtype:
    """Return the `torch.dtype` for a precision name."""
    return _TORCH_DTYPES[dtype]


def dtype_for_existing_collection(
    stored: str | None, requested: EmbeddingDtype | None
) -> EmbeddingDtype:
    """Choose the precision for adding rows to an existing collection.

    Args:
        stored: Precision recorded in the collection's metadata, or ``None`` if
            the collection predates precision tracking.
        requested: Precision the user asked for, or ``None`` if they did not
            choose. In that case the collection's own precision is used.

    Returns:
        The precision to build with, always equal to the stored one.

    Raises:
        EmbeddingPrecisionError: If the collection's precision is unknown, or
            differs from the requested one.
    """
    if stored is None:
        raise EmbeddingPrecisionError(
            "This vector database was built before AATM recorded embedding "
            "precision, so adding rows to it could mix precisions. Choose "
            "'Overwrite' to rebuild it."
        )

    stored_dtype = parse_embedding_dtype(stored)
    if requested is not None and requested != stored_dtype:
        raise EmbeddingPrecisionError(
            f"This vector database was built with {stored_dtype} embeddings, but "
            f"{requested} was requested. Repairing it would mix precisions in one "
            f"database. Repair with {stored_dtype}, or choose 'Overwrite' to "
            f"rebuild it with {requested}."
        )
    return stored_dtype


def dtype_for_queries(stored: str | None, device: str) -> EmbeddingDtype | None:
    """Choose the precision for encoding queries against a collection.

    Queries should be encoded in the precision the database was built with. If
    this machine cannot run it, fall back to ``float32``, which only shifts
    scores slightly.

    Args:
        stored: Precision recorded in the collection's metadata, or ``None``.
        device: Device local models run on.

    Returns:
        The precision to use, or ``None`` to apply the usual selection rules
        when the collection has no recorded precision.
    """
    if stored is None:
        return None

    dtype = parse_embedding_dtype(stored)
    try:
        check_dtype_supported(dtype, device)
    except EmbeddingPrecisionError as error:
        logger.warning(
            f"The vector database was built with {dtype} embeddings, but this "
            f"machine cannot run that precision: {error} Encoding queries in "
            "float32 instead; scores may differ slightly."
        )
        return "float32"
    return dtype
