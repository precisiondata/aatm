"""
Choose the numeric precision local embedding models run in.

Local embedding models run in full precision (``float32``) by default. On an
NVIDIA GPU you can opt in to half precision, which roughly halves memory use and
encodes faster:

- ``bfloat16``: needs an Ampere or newer GPU (RTX 30xx, A100 and later).
- ``float16``: works on older GPUs too, but some models produce unstable
  embeddings in it (EmbeddingGemma does not support it).

Choose the precision with, in order of priority:

1. the ``embedding_dtype`` argument of `build_local_vector_database` (or
   ``aatm init --embedding-dtype``);
2. the ``AATM_EMBEDDING_DTYPE`` environment variable;
3. the default, ``float32``.

Half precision changes the embeddings slightly, so one vector database must not
mix precisions. The builder records the precision in the collection's metadata
and refuses to repair a database with a different one. When AATM loads a
retriever, it encodes queries in the precision the database was built with.

```python
from aatm.precision import resolve_embedding_dtype

resolve_embedding_dtype(None, device="cuda")  # "float32" unless AATM_EMBEDDING_DTYPE is set
resolve_embedding_dtype("bf16", device="cuda")  # "bfloat16"
```
"""

from __future__ import annotations

import os
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
DEFAULT_EMBEDDING_DTYPE: EmbeddingDtype = "float32"

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


def resolve_embedding_dtype(value: str | None, device: str) -> EmbeddingDtype:
    """Return the precision to load a local model in.

    Args:
        value: Precision passed explicitly, or ``None`` to use
            ``AATM_EMBEDDING_DTYPE`` or the default.
        device: Device the model will run on.

    Returns:
        The canonical precision name.

    Raises:
        EmbeddingPrecisionError: If the precision is unsupported, or the device
            cannot run it.
    """
    dtype = requested_embedding_dtype(value) or DEFAULT_EMBEDDING_DTYPE
    check_dtype_supported(dtype, device)
    return dtype


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
