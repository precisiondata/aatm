from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
import torch
import time

from aatm.logs import get_logger

logger = get_logger(__name__, level=logging.DEBUG)


@dataclass
class TimingProfiler:
    """Simple cumulative timing profiler for batch processing."""

    enabled: bool = True
    synchronize_cuda: bool = False
    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @contextmanager
    def time_block(self, name: str):
        """Measure elapsed wall-clock time for a code block."""
        if not self.enabled:
            yield
            return

        if self.synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        try:
            yield
        finally:
            if self.synchronize_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            self.totals[name] += elapsed
            self.counts[name] += 1

    def add(self, name: str, elapsed: float) -> None:
        """Manually add a measured duration."""
        if not self.enabled:
            return

        self.totals[name] += elapsed
        self.counts[name] += 1

    def log_batch(
        self,
        *,
        expression_origin: str,
        row_start: int,
        n_rows: int,
        n_unique: int,
        n_new: int,
        batch_elapsed: float,
    ) -> None:
        """Log a compact per-batch timing summary."""
        if not self.enabled:
            return

        docs_per_second = n_new / batch_elapsed if batch_elapsed > 0 else 0.0

        logger.debug(
            "[PROFILE] batch | "
            f"dataset={expression_origin} | "
            f"row_start={row_start} | "
            f"rows={n_rows} | "
            f"unique={n_unique} | "
            f"new={n_new} | "
            f"elapsed={batch_elapsed:.3f}s | "
            f"docs/s={docs_per_second:.2f}"
        )

    def log_summary(self, label: str = "summary") -> None:
        """Log cumulative timing summary."""
        if not self.enabled:
            return

        total_time = sum(self.totals.values())

        logger.info(f"[PROFILE] {label} | total_profiled_time={total_time:.3f}s")

        for name, elapsed in sorted(
            self.totals.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            count = self.counts[name]
            avg = elapsed / count if count > 0 else 0.0
            pct = elapsed / total_time * 100 if total_time > 0 else 0.0

            logger.info(
                "[PROFILE] "
                f"{name} | "
                f"total={elapsed:.3f}s | "
                f"count={count} | "
                f"avg={avg:.3f}s | "
                f"{pct:.1f}%"
            )
