"""Split a dataset into shards that run independently, then read them as one.

A shard is the notes whose `note_id` hashes to it (`shard_of`), so any worker can
compute its share of a source on its own, with no coordination. Each shard is a
chunked run (`OmopExtractionPipeline.run` with ``shard=``) with its own output
folder, ledger and part files; it resumes, retries and rebuilds like any run. Row ids
are derived from the note (`aatm.omop.builder.note_row_id`), so they never collide
across shards, and the shards together hold the same rows as a single run.

## On one machine

`run_sharded` runs the shards in separate processes and returns their combined
summary:

```python
# my_pipeline.py: the factory must be importable by the worker processes
from aatm.omop import OmopExtractionPipeline

def make_pipeline():
    return OmopExtractionPipeline(
        "ecog_performance_status", model_id="openai/gpt-5-mini", max_concurrency=16,
        cache=True,
    )
```

```python
from aatm.llm import set_rate_limits
from aatm.omop import OmopRun, run_sharded
from my_pipeline import make_pipeline

if __name__ == "__main__":
    set_rate_limits("openai", requests_per_minute=5000)  # split among the processes
    summary = run_sharded(
        make_pipeline, "notes.parquet", "output/ecog", num_shards=4, chunk_size=200
    )
    OmopRun("output/ecog").table("measurement")  # every shard's rows
```

Each shard writes to ``output/ecog/shard-000-of-004/`` and so on. Run the same call
again to resume: each shard skips what it already built. The number of shards of a
folder is fixed once it has shards; a different ``num_shards`` is refused.

The pipeline is built in each process by `make_pipeline`, a function the processes can
import (defined in a module, not in a notebook or the ``__main__`` script of an
interactive session). The source is read by every process, each keeping its own
notes; it must be a path, a DataFrame, a list, or a function that returns a source
(for a database query, a function that opens its own connection and returns
`notes_from_sql`).

## Rate limits

Rate limits apply per process (`aatm.llm.limits`). `run_sharded` takes the limits set
with `aatm.llm.set_rate_limits` in the calling process and gives each worker its share:
rates and concurrency are divided by the number of processes running at once. Limits
set inside `make_pipeline` apply as given, to each process.

## On several machines

Run one shard per machine (or job) with ``shard=(index, count)``, into folders under
one parent, then read the parent:

```python
summary = await pipeline.run(
    "notes.parquet", shard_dir("output/ecog", index, 8), shard=(index, 8)
)
OmopRun("output/ecog")  # once the shard folders are copied under it
```

## The coding cache

The coding cache (`aatm.omop.CodingCache`) is a SQLite file that several processes on
one machine can share: give each pipeline the same path. Workers on other machines
keep their own cache.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import asyncio
import hashlib
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor

from aatm.logs import get_logger

if TYPE_CHECKING:
    from aatm.llm.limits import RateLimits
    from aatm.omop.builder import OmopExtractionPipeline
    from aatm.omop.runner import RunSummary

logger = get_logger(__name__)

_SHARD_DIR = re.compile(r"^shard-(\d+)-of-(\d+)$")


def shard_of(note_id: int, num_shards: int) -> int:
    """The shard of a note: a stable hash of its `note_id`, modulo `num_shards`.

    The hash spreads notes evenly whatever pattern the ids follow (sequential, with a
    stride, by site...), and is the same on every machine and Python version.
    """
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if num_shards == 1:
        return 0
    digest = hashlib.blake2b(note_id.to_bytes(8, "little", signed=True), digest_size=8)
    return int.from_bytes(digest.digest(), "little") % num_shards


def check_shard(shard: tuple[int, int]) -> tuple[int, int]:
    """Validate a ``(index, count)`` pair."""
    index, count = shard
    if count < 1:
        raise ValueError(f"A shard count must be >= 1, got {count}")
    if not 0 <= index < count:
        raise ValueError(f"Shard index {index} is outside 0 to {count - 1}")
    return index, count


def shard_dir(output_dir: str | Path, index: int, num_shards: int) -> Path:
    """The output folder of a shard under `output_dir`, e.g. ``shard-002-of-008``."""
    check_shard((index, num_shards))
    width = max(3, len(str(num_shards)))
    return Path(output_dir) / f"shard-{index:0{width}d}-of-{num_shards:0{width}d}"


def shard_dirs(output_dir: str | Path) -> list[Path]:
    """The shard folders under `output_dir`, in shard order.

    Raises:
        ValueError: If the folders are not all of the same number of shards.
    """
    folder = Path(output_dir)
    if not folder.is_dir():
        return []
    found: list[tuple[int, int, Path]] = []
    for path in folder.iterdir():
        match = _SHARD_DIR.match(path.name)
        if path.is_dir() and match:
            found.append((int(match[1]), int(match[2]), path))
    counts = {count for _, count, _ in found}
    if len(counts) > 1:
        raise ValueError(
            f"{folder} holds shards of different splits ({sorted(counts)} shards): "
            "a dataset must be split the same way in every run"
        )
    return [path for _, _, path in sorted(found)]


def shard_count(folder: Path) -> Optional[int]:
    """The number of shards of a shard folder's split, from its name."""
    match = _SHARD_DIR.match(folder.name)
    return int(match[2]) if match else None


def combine_summaries(
    summaries: Sequence["RunSummary"], output_dir: str | Path
) -> "RunSummary":
    """One summary of runs that went on in parallel, e.g. the shards of a dataset.

    Counts, rows, tokens and costs are added up; ``seconds`` is the wall time, from
    the first start to the last finish.
    """
    from aatm.omop.runner import RunSummary, _add_stats

    combined = RunSummary(output_dir=str(output_dir))
    if not summaries:
        return combined
    combined.kind = summaries[0].kind
    for name in (
        "notes",
        "built",
        "extracted",
        "reused",
        "skipped",
        "duplicates",
        "failed",
        "gave_up",
        "batches_submitted",
        "batches_collected",
        "issues",
    ):
        setattr(combined, name, sum(getattr(s, name) for s in summaries))
    totals = [s.total for s in summaries]
    combined.total = None if None in totals else sum(t or 0 for t in totals)
    for summary in summaries:
        for table, n in summary.rows.items():
            combined.rows[table] = combined.rows.get(table, 0) + n
        combined.cache_stats = _add_stats(combined.cache_stats, summary.cache_stats)
        for name, stage in summary.stages.items():
            into = combined.stages.setdefault(name, type(stage)())
            into.items += stage.items
            into.seconds += stage.seconds
            into.errors += stage.errors
            into.usage.merge(stage.usage)
            if stage.cost is not None:
                into.cost = (into.cost or 0.0) + stage.cost
            into.unpriced = sorted(set(into.unpriced) | set(stage.unpriced))
    costs = [s.cost for s in summaries if s.cost is not None]
    combined.cost = sum(costs) if costs else None
    combined.completed = all(s.completed for s in summaries)
    errors = [s.error for s in summaries if s.error]
    combined.error = "; ".join(errors) if errors else None
    started = [s.started_at for s in summaries if s.started_at]
    finished = [s.finished_at for s in summaries if s.finished_at]
    combined.started_at = min(started) if started else None
    combined.finished_at = max(finished) if finished else None
    if combined.started_at and combined.finished_at:
        combined.seconds = (combined.finished_at - combined.started_at).total_seconds()
    else:
        combined.seconds = max(s.seconds for s in summaries)
    return combined


# --- running shards in processes --------------------------------------------------


class ShardedRunError(RuntimeError):
    """Some shards of `run_sharded` stopped with an error; the others finished.

    Attributes:
        summary: The combined summary of every shard, with ``completed`` false.
        errors: The error of each shard that stopped, by shard index.
    """

    def __init__(self, summary: "RunSummary", errors: dict[int, str]):
        self.summary = summary
        self.errors = errors
        detail = "; ".join(f"shard {i}: {e}" for i, e in sorted(errors.items()))
        super().__init__(
            f"{len(errors)} shard(s) stopped: {detail}. Run again to resume them"
        )


def _run_shard(
    make_pipeline: Callable[[], "OmopExtractionPipeline"],
    source: Any,
    output_dir: str,
    shard: tuple[int, int],
    limits: dict[str, "RateLimits"],
    run_kwargs: dict[str, Any],
) -> "RunSummary":
    """Run one shard; called in a worker process."""
    from aatm.llm.limits import set_rate_limits

    for key, value in limits.items():
        set_rate_limits(key, value)
    pipeline = make_pipeline()
    if callable(source):
        source = source()
    return asyncio.run(pipeline.run(source, output_dir, shard=shard, **run_kwargs))


def _failed_summary(output_dir: Path, error: BaseException) -> "RunSummary":
    from aatm.omop.runner import RunSummary

    message = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
    return RunSummary(
        output_dir=str(output_dir), error=message, finished_at=datetime.now(timezone.utc)
    )


def run_sharded(
    make_pipeline: Callable[[], "OmopExtractionPipeline"],
    source: Any,
    output_dir: str | Path,
    num_shards: int,
    processes: Optional[int] = None,
    total: Optional[int] = None,
    **run_kwargs: Any,
) -> "RunSummary":
    """Run a dataset as `num_shards` shards, in parallel processes.

    Each shard is a chunked run in ``output_dir/shard-<i>-of-<n>``; read them all with
    ``OmopRun(output_dir)``. Calling it again resumes every shard. See
    `aatm.omop.shards`.

    Args:
        make_pipeline: Builds the pipeline in each worker process. It must be
            importable by the workers: a function defined in a module.
        source: The notes: a path, a DataFrame, a list, or a function returning a
            source (e.g. `aatm.omop.notes_from_sql` with its own connection). Every
            process reads it and keeps its shard's notes.
        output_dir: The parent folder of the shards' folders.
        num_shards: Shards to split the dataset into. Fixed for a folder once it
            has shards.
        processes: Shards run at once; ``None`` runs them all at once. The rate
            limits set with `aatm.llm.set_rate_limits` are divided among them.
        total: Notes in the source, for the estimated time left; found by itself
            for Parquet files, DataFrames and lists.
        **run_kwargs: Passed to `OmopExtractionPipeline.run` (``chunk_size``,
            ``columns``, ``max_attempts``, ``prices``, ``batch``...).

    Returns:
        The combined summary of the shards: notes, rows, tokens and cost added up,
        and the wall time.

    Raises:
        ShardedRunError: If a shard stopped with an error, once the others are
            done. Its `summary` holds what every shard did.
    """
    from aatm.llm.limits import configured_rate_limits
    from aatm.omop.runner import count_notes

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if processes is not None and processes < 1:
        raise ValueError("processes must be >= 1 or None")
    if "shard" in run_kwargs:
        raise TypeError("run_sharded sets the shard of each run: do not pass shard")
    output_dir = Path(output_dir)
    existing = shard_dirs(output_dir)
    if existing and shard_count(existing[0]) != num_shards:
        raise ValueError(
            f"{output_dir} holds the shards of a split into "
            f"{shard_count(existing[0])}: use that num_shards, or a new folder"
        )
    workers = min(processes or num_shards, num_shards)
    limits = {
        key: value.divided(workers) for key, value in configured_rate_limits().items()
    }
    if total is None and not callable(source):
        total = count_notes(source)
    if total is not None:
        run_kwargs["total"] = -(-total // num_shards)  # an estimate per shard

    folders = [shard_dir(output_dir, i, num_shards) for i in range(num_shards)]
    logger.info(
        "Running %s shards in %s processes into %s", num_shards, workers, output_dir
    )
    summaries: list[RunSummary] = []
    errors: dict[int, str] = {}
    # Spawned processes start clean on every platform: no inherited threads or
    # clients from the parent.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [
            pool.submit(
                _run_shard,
                make_pipeline,
                source,
                str(folder),
                (index, num_shards),
                limits,
                dict(run_kwargs),
            )
            for index, folder in enumerate(folders)
        ]
        for index, (folder, future) in enumerate(zip(folders, futures)):
            try:
                summaries.append(future.result())
            except Exception as e:
                logger.error("Shard %s stopped: %s", index, e)
                failed = _failed_summary(folder, e)
                errors[index] = failed.error or ""
                summaries.append(failed)

    summary = combine_summaries(summaries, output_dir)
    if total is not None:
        summary.total = total
    logger.info("Sharded run %s\n%s", "finished" if not errors else "stopped", summary)
    if errors:
        raise ShardedRunError(summary, errors)
    return summary
