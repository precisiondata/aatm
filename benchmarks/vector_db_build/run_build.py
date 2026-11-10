"""Benchmark a full local vector database build (issue #61).

Builds the vector database for a local embedding model from the mapping datasets
in a scratch directory, never in the project's own `.aatm/chroma_vector_dbs`,
and records:

- total build time, rows embedded and the batch setting calibration chose;
- the build profiler's per-stage totals (`profile_timing=True`);
- peak dedicated GPU memory and peak shared GPU memory (Windows), sampled while
  the build runs;
- warnings logged during the build, such as out-of-memory backoffs;
- the top-10 search results for a fixed set of queries, to compare builds.

With ``--repair-sample N`` it then deletes N random rows and runs a repair,
checking that exactly those rows are embedded again.

The code under test is whatever `aatm` is importable, so the same script can
benchmark two versions by changing ``PYTHONPATH``. Results go to a JSON file.

Usage, from the repository root:

    python benchmarks/vector_db_build/run_build.py --label after \\
        --work-dir /path/to/scratch --out benchmarks/vector_db_build/results/after.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LAB_RESULTS = REPO_ROOT / "benchmarks" / "concept_search" / "lab_results.csv"
N_SAMPLED_QUERIES = 100
TOP_K = 10


class GpuMemorySampler:
    """Sample dedicated (nvidia-smi) and shared (Windows counter) GPU memory."""

    def __init__(self, interval_seconds: float = 2.0) -> None:
        self.interval = interval_seconds
        self.dedicated_mib: list[int] = []
        self.shared_bytes: list[float] = []
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(target=self._sample_dedicated, daemon=True),
        ]
        if platform.system() == "Windows":
            self._threads.append(
                threading.Thread(target=self._sample_shared, daemon=True)
            )

    def _sample_dedicated(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
                self.dedicated_mib.append(int(out.split()[0]))
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass
            self._stop.wait(self.interval)

    def _sample_shared(self) -> None:
        # Shared usage summed over adapters, as Task Manager shows it.
        command = (
            "(Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage')"
            ".CounterSamples | Measure-Object CookedValue -Sum | "
            "ForEach-Object { $_.Sum }"
        )
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", command],
                    capture_output=True,
                    text=True,
                    timeout=30,
                ).stdout
                self.shared_bytes.append(float(out.strip().replace(",", ".")))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self._stop.wait(self.interval * 5)

    def __enter__(self) -> GpuMemorySampler:
        for thread in self._threads:
            thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=60)

    def summary(self) -> dict[str, Any]:
        gib = 1024**3
        return {
            "dedicated_baseline_gib": (
                round(self.dedicated_mib[0] / 1024, 2) if self.dedicated_mib else None
            ),
            "dedicated_peak_gib": (
                round(max(self.dedicated_mib) / 1024, 2) if self.dedicated_mib else None
            ),
            "shared_baseline_gib": (
                round(self.shared_bytes[0] / gib, 2) if self.shared_bytes else None
            ),
            "shared_peak_gib": (
                round(max(self.shared_bytes) / gib, 2) if self.shared_bytes else None
            ),
        }


class RecordingHandler(logging.Handler):
    """Keep warnings and the batch-setting messages logged during the build."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.warnings: list[str] = []
        self.batch_setting: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            self.warnings.append(f"{record.name}: {message}")
        elif message.startswith("Encoding with"):
            self.batch_setting.append(message)


def answer_prompt(answer: str) -> None:
    """Answer the build's "database exists" prompt without a terminal."""
    import questionary

    class Answer:
        def ask(self) -> str:
            return answer

    questionary.select = lambda *args, **kwargs: Answer()  # type: ignore[assignment]


def capture_profiler(ldu: Any) -> list[Any]:
    """Make the build's TimingProfiler instances reachable after it returns."""
    instances: list[Any] = []
    original = ldu.TimingProfiler

    class RecordedProfiler(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            instances.append(self)

    ldu.TimingProfiler = RecordedProfiler
    return instances


def load_queries(datasets_dir: Path, seed: int = 0) -> list[str]:
    """The concept-search benchmark's lab texts plus sampled dataset expressions."""
    queries = pd.read_csv(LAB_RESULTS)["text"].astype(str).tolist()
    sample = (
        pd.read_csv(
            datasets_dir / "mapped_non_standard_concept_synonym.csv",
            usecols=["expression"],
        )["expression"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sample(n=N_SAMPLED_QUERIES, random_state=seed)
        .tolist()
    )
    return queries + sample


def search(
    db_dir: Path, collection_name: str, spec: Any, dtype: str, queries: list[str]
) -> dict[str, Any]:
    """Return the top-k ids and distances for each query, and the row count."""
    import chromadb

    from aatm.local_database_utils import release_chromadb_clients

    embedding_function = spec.embedding_function_cls(model=spec.model_id, dtype=dtype)
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(collection_name)
    count = collection.count()
    embeddings = embedding_function(queries)
    result = collection.query(
        query_embeddings=embeddings, n_results=TOP_K, include=["distances"]
    )
    del collection, client, embedding_function
    release_chromadb_clients()
    return {
        "count": count,
        "results": {
            query: {"ids": ids, "distances": [round(d, 6) for d in distances]}
            for query, ids, distances in zip(
                queries, result["ids"], result["distances"]
            )
        },
    }


def repair(
    db_dir: Path, collection_name: str, n_delete: int, build: Any
) -> dict[str, Any]:
    """Delete random rows, run a repair and check exactly those come back."""
    import chromadb

    from aatm.local_database_utils import release_chromadb_clients

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(collection_name)
    before = collection.count()
    all_ids = collection.get(include=[])["ids"]
    deleted = random.Random(0).sample(all_ids, n_delete)
    for start in range(0, n_delete, 5000):
        collection.delete(ids=deleted[start : start + 5000])
    after_delete = collection.count()
    del collection, client, all_ids
    release_chromadb_clients()

    answer_prompt("Repair")
    started = time.perf_counter()
    build()
    elapsed = time.perf_counter() - started

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(collection_name)
    after_repair = collection.count()
    restored = collection.get(ids=deleted, include=[])["ids"]
    del collection, client
    release_chromadb_clients()
    return {
        "deleted": n_delete,
        "count_before": before,
        "count_after_delete": after_delete,
        "count_after_repair": after_repair,
        "deleted_ids_restored": len(set(restored)),
        "repair_seconds": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="embeddinggemma-300M")
    parser.add_argument("--dtype", default=None, help="Default: the checkpoint's.")
    parser.add_argument(
        "--label", required=True, help="Name of this run, e.g. 'before'."
    )
    parser.add_argument(
        "--code-ref",
        default=None,
        help="The aatm version under test, e.g. a commit, recorded in the results.",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        type=Path,
        default=REPO_ROOT / ".aatm" / "datasets",
        help="Mapping datasets built by `aatm init`.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repair-sample", type=int, default=0)
    parser.add_argument(
        "--keep-db", action="store_true", help="Keep the scratch database."
    )
    args = parser.parse_args()

    import torch

    import aatm
    from aatm import local_database_utils as ldu
    from aatm.precision import COLLECTION_DTYPE_KEY
    from aatm.registries.retrievers import CHROMADB_RETRIEVER_MODEL_REGISTRY

    aatm.configure_logging(logging.INFO)
    recorder = RecordingHandler()
    logging.getLogger("aatm").addHandler(recorder)

    spec = CHROMADB_RETRIEVER_MODEL_REGISTRY[args.model]
    work_dir = args.work_dir.resolve()
    datasets_dir = work_dir / ".aatm" / "datasets"
    if not datasets_dir.exists():
        shutil.copytree(args.datasets.resolve(), datasets_dir)
    queries = load_queries(datasets_dir)
    db_dir = work_dir / "db"
    if db_dir.exists():
        shutil.rmtree(db_dir)
    db_dir.mkdir(parents=True)
    os.chdir(work_dir)  # the build reads `.aatm/datasets` from the working directory

    profilers = capture_profiler(ldu)

    def build() -> None:
        ldu.build_local_vector_database(
            args.model,
            vector_db_dir=db_dir,
            embedding_dtype=args.dtype,
            profile_timing=True,
        )

    answer_prompt("Overwrite")
    with GpuMemorySampler() as sampler:
        started = time.perf_counter()
        build()
        build_seconds = time.perf_counter() - started

    import chromadb

    client = chromadb.PersistentClient(path=str(db_dir))
    dtype = (client.get_collection(spec.collection_name).metadata or {}).get(
        COLLECTION_DTYPE_KEY, args.dtype or "float32"
    )
    del client
    ldu.release_chromadb_clients()

    profile = profilers[0]
    result: dict[str, Any] = {
        "label": args.label,
        "code_ref": args.code_ref,
        "model": args.model,
        "dtype": dtype,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "build_seconds": round(build_seconds, 1),
        "batch_setting": recorder.batch_setting,
        "warnings_during_build": list(recorder.warnings),
        "gpu_memory": sampler.summary(),
        "profile_totals_seconds": {
            name: round(total, 1)
            for name, total in sorted(profile.totals.items(), key=lambda x: -x[1])
        },
        "db_size_gib": round(
            sum(f.stat().st_size for f in db_dir.rglob("*") if f.is_file()) / 1024**3,
            2,
        ),
    }
    search_result = search(db_dir, spec.collection_name, spec, dtype, queries)
    result["rows"] = search_result["count"]
    result["rows_per_second"] = round(search_result["count"] / build_seconds, 1)
    result["search"] = search_result["results"]

    if args.repair_sample:
        recorder.warnings.clear()
        result["repair"] = repair(
            db_dir, spec.collection_name, args.repair_sample, build
        )
        result["repair"]["warnings"] = list(recorder.warnings)

    if not args.keep_db:
        shutil.rmtree(db_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {k: v for k, v in result.items() if k != "search"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
