"""Does sharding a run across processes make it faster? (issue #56)

Runs the same synthetic dataset through `aatm.omop.run_sharded` with 1, 2, 4...
shards and the **same total concurrency**, so only the split into processes changes:

- Extraction goes through the real `SimpleLLMExtractor` (prompt, parsing, span
  validation) and the real rate limiter, with a simulated LLM that answers after a
  random latency, so no API is called and nothing is paid.
- Building, the vocabulary lookups (``.aatm/omop.db``), the Parquet writes and the
  ledger are the real ones.
- Coding uses one of the scenarios below.

Scenarios:

- ``instant``: a coder that answers at once. Measures the framework itself.
- ``retrieval``: a `TerminologyMapper` with a local embedding model and ChromaDB index
  (embeddinggemma-300M by default), on the device `aatm.device` picks (GPU if any).
- ``agent``: a simulated agentic coder: each distinct text waits a random latency, at
  most ``--coder-concurrency`` at once in all (divided among the shards).
- ``rate-limited``: ``instant``, with the extraction model capped at ``--rpm``
  requests per minute in all (divided among the shards).

Run from the repository root:

```bash
uv run python benchmarks/sharding/run_benchmark.py --notes 2000 --shards 1 2 4
```

Results go to ``results/<label>.csv``: wall time (process start-up included), run
time (first shard start to last finish), notes per minute, CPU time and peak memory
of the worker processes.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import math
import random
import re
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
import psutil

from aatm.llm import reset_rate_limits, set_rate_limits
from aatm.llm.limits import estimate_tokens, get_rate_limiter
from aatm.llm.usage import record_usage
from aatm.omop import run_sharded
from aatm.omop.coding import CodedText

HERE = Path(__file__).parent
LAB_CASES = HERE.parent / "concept_search" / "lab_results.csv"
RESULTS = HERE / "results"
MODEL_ID = "simulated/extractor"
HEMOGLOBIN = 3000963  # a standard Measurement concept, for the simulated coders

# --- synthetic notes -------------------------------------------------------------

PREFIXES = ["", "serum ", "plasma ", "blood ", "fasting ", "random ", "venous "]
FILLER = [
    "Patient seen in clinic for follow-up.",
    "No new complaints since the last visit.",
    "Tolerating current medications without side effects.",
    "Discussed diet and exercise; patient agrees with the plan.",
    "Review of systems otherwise negative.",
    "Physical exam unremarkable apart from mild pallor.",
    "Plan to repeat labs in three months.",
    "Return precautions reviewed with the patient.",
]
LAB_LINE = re.compile(
    r"^Labs (\d\d/\d\d/\d{4}): (?P<test>.+?) (?P<value>[\d.]+) (?P<unit>\S+)\.$"
)


def lab_names() -> list[tuple[str, str]]:
    """(test name, unit) variants: real lab texts with specimens and casings."""
    labs = pd.read_csv(LAB_CASES)[["text", "unit"]].dropna().drop_duplicates()
    names = []
    for text, unit in labs.itertuples(index=False):
        if not re.fullmatch(r"[\w%/^.-]+", str(unit)):
            continue
        for prefix in PREFIXES:
            for name in {prefix + text, (prefix + text).lower()}:
                names.append((name.strip(), str(unit)))
    return names


def make_notes(
    n: int, labs_per_note: int, filler_lines: int, seed: int = 0
) -> pd.DataFrame:
    """Notes with lab results on their own lines, test names drawn Zipf-like."""
    rng = random.Random(seed)
    names = lab_names()
    weights = [1 / (rank + 1) for rank in range(len(names))]
    rows = []
    for note_id in range(1, n + 1):
        lines = [f"Clinic visit {note_id}."]
        lines += rng.sample(FILLER, k=min(filler_lines, len(FILLER)))
        for name, unit in rng.choices(names, weights, k=labs_per_note):
            value = round(rng.uniform(0.5, 300), 1)
            lines.append(f"Labs 03/15/2025: {name} {value} {unit}.")
        rows.append(
            {
                "note_id": note_id,
                "text": "\n".join(lines),
                "person_id": 1000 + note_id % 500,
                "note_date": "2025-03-15",
            }
        )
    return pd.DataFrame(rows)


# --- simulated models ------------------------------------------------------------------


def lognormal(rng: random.Random, mean: float, sigma: float) -> float:
    """A latency with the given mean and a long right tail."""
    if mean <= 0:
        return 0.0
    return rng.lognormvariate(math.log(mean) - sigma**2 / 2, sigma)


class SimulatedLLM:
    """Answers the extraction prompt from the note's lab lines, after a latency.

    Goes through the model's rate limiter, like the real adapters.
    """

    model_id = MODEL_ID

    def __init__(self, latency: float, sigma: float):
        self.latency = latency
        self.sigma = sigma
        self.rng = random.Random()

    async def generate(self, *, system: str, user: str, output_type: Any) -> Any:
        return await get_rate_limiter(self.model_id).run(
            lambda: self._generate(user, output_type),
            estimated_tokens=estimate_tokens(system, user),
        )

    async def _generate(self, user: str, output_type: Any) -> Any:
        entities: list[dict[str, Any]] = []
        for line in user.splitlines():
            line_id, _, text = line.partition("] ")
            match = LAB_LINE.match(text)
            if not match:
                continue
            line_id = line_id.lstrip("[")
            parent = f"e{len(entities) + 1}"
            entities.append(
                {
                    "id": parent,
                    "text": match["test"],
                    "key": "lab_test",
                    "line": line_id,
                }
            )
            for key, group in (("lab_value", "value"), ("lab_unit", "unit")):
                entities.append(
                    {
                        "id": f"e{len(entities) + 1}",
                        "text": match[group],
                        "key": key,
                        "line": line_id,
                        "parents": [parent],
                    }
                )
        await asyncio.sleep(lognormal(self.rng, self.latency, self.sigma))
        record_usage(len(user) // 4, 60 * len(entities))
        return output_type.model_validate({"entities": entities})


class InstantCoder:
    """Codes every text at once, to the same concept."""

    def cache_identity(self) -> dict[str, Any]:
        return {"coder": type(self).__name__}

    def code(
        self,
        texts: Sequence[str],
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
    ) -> list[CodedText]:
        return [CodedText(text=t, concept_id=HEMOGLOBIN, score=1.0) for t in texts]


class SimulatedAgentCoder(InstantCoder):
    """Each text waits a random latency, at most `max_concurrency` at once."""

    def __init__(self, latency: float, sigma: float, max_concurrency: int):
        self.latency = latency
        self.sigma = sigma
        self.max_concurrency = max_concurrency
        self.rng = random.Random()

    def code(
        self,
        texts: Sequence[str],
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
    ) -> list[CodedText]:
        async def search_all() -> None:
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def search() -> None:
                async with semaphore:
                    await asyncio.sleep(lognormal(self.rng, self.latency, self.sigma))

            await asyncio.gather(*(search() for _ in texts))

        asyncio.run(search_all())  # the build runs in a thread, like the real agent
        return super().code(texts, domain, vocabularies)


@dataclass
class Config:
    scenario: str
    shards: int
    extraction_concurrency: int  # in all
    coder_concurrency: int  # in all, for the agent
    latency: float
    sigma: float
    coder_latency: float
    retriever: str
    chroma_dir: Optional[str]
    mapper_batch_size: int


def make_pipeline(config: Config) -> Any:
    """Build the pipeline in a worker process."""
    from aatm.extractors import SimpleLLMExtractor
    from aatm.omop import OmopExtractionPipeline
    from aatm.omop.builder import _load_definitions

    definitions = _load_definitions("lab_results")
    per_process = max(1, config.extraction_concurrency // config.shards)
    extractor = SimpleLLMExtractor(
        entity_definitions=definitions,
        llm=SimulatedLLM(config.latency, config.sigma),
        max_concurrency=per_process,
        on_error="record",
    )
    mapper: Any
    if config.scenario == "retrieval":
        from aatm.omop.coding import TerminologyMapperCoder
        from aatm.registries.retrievers import load_retriever
        from aatm.terminology_mapper import TerminologyMapper

        retriever = load_retriever(config.retriever, local_path=config.chroma_dir)
        mapper = TerminologyMapperCoder(
            TerminologyMapper(retriever=retriever, batch_size=config.mapper_batch_size)
        )
    elif config.scenario == "agent":
        mapper = SimulatedAgentCoder(
            config.coder_latency,
            config.sigma,
            max(1, config.coder_concurrency // config.shards),
        )
    else:
        mapper = InstantCoder()
    return OmopExtractionPipeline(definitions, extractor=extractor, mapper=mapper)


# --- measuring ------------------------------------------------------------------------


class ProcessMonitor:
    """Samples the CPU time and memory of this process's children."""

    def __init__(self, every: float = 0.25):
        self.every = every
        self.cpu: dict[int, float] = {}
        self.rss: dict[int, int] = {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self) -> "ProcessMonitor":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop.set()
        self.thread.join()

    def _sample(self) -> None:
        me = psutil.Process()
        while not self.stop.wait(self.every):
            for child in me.children(recursive=True):
                try:
                    times = child.cpu_times()
                    self.cpu[child.pid] = times.user + times.system
                    rss = child.memory_info().rss
                except psutil.Error:
                    continue
                self.rss[child.pid] = max(self.rss.get(child.pid, 0), rss)

    @property
    def cpu_seconds(self) -> float:
        return sum(self.cpu.values())

    @property
    def peak_rss_mb(self) -> float:
        # Workers of one run overlap, so their peaks add up.
        return sum(self.rss.values()) / 2**20


def run_one(
    config: Config, source: Path, notes: int, chunk_size: int, rpm: Optional[float]
) -> dict[str, Any]:
    out = Path(tempfile.mkdtemp(prefix="aatm-shards-"))
    reset_rate_limits()
    if rpm is not None:
        set_rate_limits("simulated", requests_per_minute=rpm)
    try:
        started = time.perf_counter()
        with ProcessMonitor() as monitor:
            summary = run_sharded(
                functools.partial(make_pipeline, config),
                str(source),
                out,
                num_shards=config.shards,
                chunk_size=chunk_size,
                log_every=3600,
            )
        wall = time.perf_counter() - started
    finally:
        reset_rate_limits()
        shutil.rmtree(out, ignore_errors=True)
    stages = summary.stages
    return {
        **asdict(config),
        "rpm": rpm,
        "notes": notes,
        "built": summary.built,
        "failed": summary.failed,
        "wall_s": round(wall, 1),
        "run_s": round(summary.seconds, 1),
        "notes_per_min": round(notes / wall * 60, 1),
        "run_notes_per_min": round(notes / summary.seconds * 60, 1),
        "extraction_s": round(stages["extraction"].seconds, 1),
        "coding_s": round(stages["coding"].seconds, 1),
        "building_s": round(stages["building"].seconds, 1),
        "coded_texts": stages["coding"].items,
        "cpu_s": round(monitor.cpu_seconds, 1),
        "peak_rss_mb": round(monitor.peak_rss_mb),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notes", type=int, default=2000)
    parser.add_argument("--labs-per-note", type=int, default=6)
    parser.add_argument("--filler-lines", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--shards", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["instant", "retrieval", "agent", "rate-limited"],
        choices=["instant", "retrieval", "agent", "rate-limited"],
    )
    parser.add_argument(
        "--concurrency", type=int, default=64, help="Extraction calls at once, in all."
    )
    parser.add_argument(
        "--coder-concurrency",
        type=int,
        default=16,
        help="Agent searches at once, in all.",
    )
    parser.add_argument("--latency", type=float, default=3.0, help="Mean LLM seconds.")
    parser.add_argument("--coder-latency", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=0.5, help="Latency spread.")
    parser.add_argument("--rpm", type=float, default=600, help="For rate-limited.")
    parser.add_argument("--retriever", default="embeddinggemma-300M")
    parser.add_argument(
        "--chroma-dir", default=".aatm/chroma_vector_dbs_subset/embeddinggemma-300M"
    )
    parser.add_argument("--mapper-batch-size", type=int, default=16)
    parser.add_argument("--label", default="results")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="aatm-shards-notes-"))
    source = workdir / "notes.parquet"
    make_notes(args.notes, args.labs_per_note, args.filler_lines).to_parquet(source)

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{args.label}.csv"
    rows = []
    for scenario in args.scenarios:
        for shards in args.shards:
            config = Config(
                scenario=scenario,
                shards=shards,
                extraction_concurrency=args.concurrency,
                coder_concurrency=args.coder_concurrency,
                latency=args.latency,
                sigma=args.sigma,
                coder_latency=args.coder_latency,
                retriever=args.retriever,
                chroma_dir=args.chroma_dir,
                mapper_batch_size=args.mapper_batch_size,
            )
            rpm = args.rpm if scenario == "rate-limited" else None
            row = run_one(config, source, args.notes, args.chunk_size, rpm)
            rows.append(row)
            print(
                f"{scenario:>12} x{shards}: {row['wall_s']:>7} s wall, "
                f"{row['run_s']:>7} s run, {row['notes_per_min']:>8} notes/min, "
                f"cpu {row['cpu_s']} s, {row['peak_rss_mb']} MB, "
                f"built {row['built']}, failed {row['failed']}",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(path, index=False)
    shutil.rmtree(workdir, ignore_errors=True)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
