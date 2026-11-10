"""Keep local embedding models within GPU memory while building the vector database.

Running out of GPU memory is worse than a crash on some systems. On Windows, the
NVIDIA driver can move the overflow to system RAM instead of raising an error,
and the GPU then runs dozens of times slower without saying why. This module
prevents overflows in three layers:

1. `GpuMemoryLimit` caps how much memory PyTorch may use, below what is free on
   the GPU. PyTorch then raises `torch.OutOfMemoryError` before the driver would
   start spilling to system RAM.
2. `calibrate_max_batch_tokens` picks the token budget per batch by encoding
   real texts: the longest ones, to measure the worst-case memory, and typical
   ones, to measure speed. It returns the fastest budget whose worst case fits
   comfortably under the cap. `calibrate_batch_size` does the same for a fixed
   number of texts per batch.
3. `encode_with_oom_backoff` halves the budget (or batch size) and retries if
   an out-of-memory error happens anyway, for example because another program
   started using the GPU.

The fastest setting usually needs far less memory than the GPU has. On an RTX
4090, EmbeddingGemma-300M and Qwen3-Embedding-0.6B are fastest at 4-8k tokens
per batch, about 2 GiB; larger batches no longer fit in the GPU's on-chip cache
and get slower. Free memory is not idle compute here, so calibration does not
try to fill it.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

import torch

from aatm.logs import get_logger

logger = get_logger(__name__)

GIB = 1024**3

CANDIDATE_BATCH_SIZES: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024)
"""Batch sizes `calibrate_batch_size` tries, smallest first."""

CANDIDATE_BATCH_TOKENS: tuple[int, ...] = (
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
)
"""Token budgets `calibrate_max_batch_tokens` tries, smallest first."""

CALIBRATION_MEMORY_MARGIN = 0.85
"""Share of the memory limit a worst-case batch may use during calibration."""

MIN_SPEEDUP_FOR_LARGER_BATCH = 1.05
"""Speed ratio over the best smaller batch size a larger one must reach to be chosen."""


class GpuMemoryError(RuntimeError):
    """Raised when a model cannot run within the available GPU memory."""


def free_cached_gpu_memory() -> None:
    """Release unused memory cached by PyTorch, for example after an OOM error."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class GpuMemoryLimit:
    """Cap PyTorch's GPU memory use for the duration of a ``with`` block.

    The cap is the memory PyTorch already holds (for example the model's
    weights) plus what is free on the GPU, minus a headroom left for other
    programs and the display. The previous cap is restored on exit.

    Attributes:
        enabled: When ``False``, the block runs without a cap.
        headroom_fraction: Share of total GPU memory left unused.
        min_headroom_bytes: Minimum memory left unused, in bytes.
        limit_bytes: The cap in bytes, set on entry. ``0`` when the cap is
            disabled or CUDA is unavailable.
    """

    enabled: bool = True
    headroom_fraction: float = 0.10
    min_headroom_bytes: int = GIB
    limit_bytes: int = 0
    _device: int = field(default=0, repr=False)
    _previous_fraction: float | None = field(default=None, repr=False)

    def __enter__(self) -> GpuMemoryLimit:
        if not self.enabled or not torch.cuda.is_available():
            return self

        self._device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(self._device)
        headroom = max(self.min_headroom_bytes, int(total * self.headroom_fraction))
        held = torch.cuda.memory_reserved(self._device)
        self.limit_bytes = held + max(0, free - headroom)

        get_fraction = getattr(torch.cuda, "get_per_process_memory_fraction", None)
        self._previous_fraction = (
            get_fraction(self._device) if get_fraction is not None else 1.0
        )
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, self.limit_bytes / total), self._device
        )
        logger.info(
            f"Limiting GPU memory to {self.limit_bytes / GIB:.1f} GiB of "
            f"{total / GIB:.1f} GiB ({free / GIB:.1f} GiB free at start)."
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._previous_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(
                self._previous_fraction, self._device
            )
            self._previous_fraction = None


class CudaMemoryProbe:
    """Read PyTorch's peak GPU memory. Replaced by a fake in tests."""

    def reset_peak(self) -> None:
        free_cached_gpu_memory()
        torch.cuda.reset_peak_memory_stats()

    def peak_bytes(self) -> int:
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_reserved())

    def synchronize(self) -> None:
        torch.cuda.synchronize()


def calibrate_batch_size(
    encode: Callable[[Sequence[str], int], Any],
    longest_texts: Sequence[str],
    typical_texts: Sequence[str],
    limit_bytes: int,
    candidates: Sequence[int] = CANDIDATE_BATCH_SIZES,
    memory_margin: float = CALIBRATION_MEMORY_MARGIN,
    probe: CudaMemoryProbe | None = None,
    timer: Callable[[], float] = time.perf_counter,
    worst_case_rows: Callable[[int], int] | None = None,
    unit: str = "batch size",
) -> int:
    """Pick the fastest batch size whose worst case fits in GPU memory.

    For each candidate, smallest first:

    1. Encode a batch of the longest texts: ``worst_case_rows(size)`` of them,
       by default ``size``. Padding makes every row as long as the longest, so
       this is the most memory any batch of that size can need. Stop if it uses more than ``memory_margin`` of
       ``limit_bytes`` or raises an out-of-memory error.
    2. Encode ``typical_texts`` and measure texts per second. Stop once a size
       is not at least `MIN_SPEEDUP_FOR_LARGER_BATCH` times faster than the best
       so far, because the extra memory is not worth it.

    Args:
        encode: Encodes texts with the given batch size.
        longest_texts: Longest texts in the data, longest first. Should hold at
            least as many texts as the largest candidate.
        typical_texts: A random sample of the data, for measuring speed.
        limit_bytes: GPU memory PyTorch may use, from `GpuMemoryLimit`.
        candidates: Batch sizes to try, in increasing order.
        memory_margin: Share of ``limit_bytes`` a worst-case batch may use.
        probe: Reads peak GPU memory.
        timer: Clock used to measure speed.
        worst_case_rows: Number of longest texts in the worst-case batch for a
            candidate. Defaults to the candidate itself.
        unit: What a candidate is, for log messages.

    Returns:
        The chosen batch size.

    Raises:
        GpuMemoryError: If even the smallest candidate does not fit.
    """
    if not longest_texts or not typical_texts:
        return candidates[0]

    probe = probe or CudaMemoryProbe()
    budget = memory_margin * limit_bytes
    best_size: int | None = None
    best_rate = 0.0

    # Warm up kernels so the first candidate's speed is not skewed.
    encode(typical_texts[: candidates[0]], candidates[0])

    for size in candidates:
        # Repeat the longest texts if there are fewer than `size`: no real batch
        # can hold longer ones, so this is still the worst case.
        rows = worst_case_rows(size) if worst_case_rows else size
        repeats = -(-rows // len(longest_texts))
        worst_case = (list(longest_texts) * repeats)[:rows]

        probe.reset_peak()
        out_of_memory = False
        try:
            encode(worst_case, size)
            peak = probe.peak_bytes()
        except torch.OutOfMemoryError:
            out_of_memory = True
        if out_of_memory:
            free_cached_gpu_memory()
            logger.debug(
                f"{unit.capitalize()} {size}: out of memory on the longest texts."
            )
            break

        if peak > budget:
            logger.debug(
                f"{unit.capitalize()} {size}: worst case needs {peak / GIB:.2f} GiB, "
                f"over the {budget / GIB:.2f} GiB budget."
            )
            break

        started = timer()
        encode(typical_texts, size)
        probe.synchronize()
        rate = len(typical_texts) / max(timer() - started, 1e-9)
        logger.debug(
            f"{unit.capitalize()} {size}: worst case {peak / GIB:.2f} GiB, "
            f"{rate:.0f} texts/s."
        )

        # A larger batch needs more memory, so it must be clearly faster to win.
        if rate >= best_rate * MIN_SPEEDUP_FOR_LARGER_BATCH:
            best_size, best_rate = size, rate
        else:
            break

    free_cached_gpu_memory()

    if best_size is None:
        raise GpuMemoryError(
            f"The model does not fit in GPU memory even with {unit} "
            f"{candidates[0]}. Try half precision (embedding_dtype='bfloat16'), "
            "a smaller model, or closing other programs that use the GPU."
        )

    logger.info(f"Encoding with {unit} {best_size} ({best_rate:.0f} texts/s).")
    return best_size


def calibrate_max_batch_tokens(
    encode: Callable[[Sequence[str], int], Any],
    longest_texts: Sequence[str],
    longest_tokens: int,
    typical_texts: Sequence[str],
    limit_bytes: int,
    candidates: Sequence[int] = CANDIDATE_BATCH_TOKENS,
    memory_margin: float = CALIBRATION_MEMORY_MARGIN,
    probe: CudaMemoryProbe | None = None,
    timer: Callable[[], float] = time.perf_counter,
) -> int:
    """Pick the fastest token budget per batch whose worst case fits in memory.

    Works like `calibrate_batch_size`, with token budgets as candidates. The
    worst case for a budget is a batch of the longest texts, as many as the
    budget allows: it has the most tokens per row, so attention needs the most
    memory.

    Args:
        encode: Encodes texts with the given token budget.
        longest_texts: Longest texts in the data, longest first.
        longest_tokens: Token count of the longest text.
        typical_texts: A random sample of the data, for measuring speed.
        limit_bytes: GPU memory PyTorch may use, from `GpuMemoryLimit`.
        candidates: Token budgets to try, in increasing order.
        memory_margin: Share of ``limit_bytes`` a worst-case batch may use.
        probe: Reads peak GPU memory.
        timer: Clock used to measure speed.

    Returns:
        The chosen token budget.

    Raises:
        GpuMemoryError: If even the smallest candidate does not fit.
    """
    return calibrate_batch_size(
        encode,
        longest_texts,
        typical_texts,
        limit_bytes=limit_bytes,
        candidates=candidates,
        memory_margin=memory_margin,
        probe=probe,
        timer=timer,
        worst_case_rows=lambda budget: max(1, budget // max(1, longest_tokens)),
        unit="token budget",
    )


def encode_with_oom_backoff(embedding_function: Any, texts: Sequence[str]) -> Any:
    """Encode texts, halving the batch size after each out-of-memory error.

    Halves ``max_batch_tokens`` when the embedding function batches by token
    budget, and ``batch_size`` otherwise. The smaller value is kept for later
    calls.

    Args:
        embedding_function: Local embedding function with a ``batch_size``
            attribute and, optionally, a ``max_batch_tokens`` one.
        texts: Texts to encode.

    Returns:
        The embeddings.

    Raises:
        GpuMemoryError: If encoding runs out of memory even with batch size 1.
    """
    while True:
        out_of_memory = False
        try:
            return embedding_function(texts)
        except torch.OutOfMemoryError:
            # Leave the except block before freeing memory: the exception's
            # traceback keeps the failed batch's tensors alive until then.
            out_of_memory = True

        if out_of_memory:
            free_cached_gpu_memory()
            attribute = (
                "max_batch_tokens"
                if getattr(embedding_function, "max_batch_tokens", None) is not None
                else "batch_size"
            )
            size = getattr(embedding_function, attribute)
            if size <= 1:
                raise GpuMemoryError(
                    "Ran out of GPU memory even with batch size 1. Try half "
                    "precision (embedding_dtype='bfloat16'), a smaller model, or "
                    "closing other programs that use the GPU."
                )
            setattr(embedding_function, attribute, size // 2)
            label = "token budget" if attribute == "max_batch_tokens" else "batch size"
            logger.warning(
                f"Ran out of GPU memory with {label} {size}; retrying with {size // 2}."
            )
