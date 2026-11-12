"""A dry run of `OmopExtractionPipeline.run`: its settings, an estimate and warnings.

`OmopExtractionPipeline.plan` runs nothing costly: no model call, no note read beyond
what counting them takes. It reports

- the effective settings of the run and where each came from (``default``,
  ``argument`` or ``set_rate_limits``);
- the number of notes, how many an earlier run in the output folder already did, and,
  from a pilot run, the expected LLM calls, tokens, cost and time;
- warnings about combinations that cost time or money, each with a fix.

```python
pipeline = OmopExtractionPipeline("ecog_performance_status", model_id="openai/gpt-5-mini")
print(pipeline.plan("notes.parquet", "output/ecog"))

# A pilot: a run on a sample, in its own folder. Its summaries give the estimate.
await pipeline.run(sample, "output/pilot", prices={"openai/gpt-5-mini": (0.25, 2.0)})
print(pipeline.plan("notes.parquet", "output/ecog", pilot="output/pilot"))
```

The estimate scales the pilot's calls, tokens and cost per note to the notes left to
do, and its time per call to the planned concurrency. It is as good as the pilot's
sample: take one like the dataset (note lengths, note types), of a few hundred notes.
With ``cache=True``, the pilot's coding cache hit rate is lower than a full run's, so
the coding estimate errs high.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel

from aatm.llm.limits import get_rate_limits
from aatm.llm.usage import BATCH_DISCOUNT, Prices, TokenUsage
from aatm.omop.runner import (
    DEFAULT_CONCURRENCY,
    LEDGER_FILE,
    OmopRun,
    RunLedger,
    RunSummary,
    _duration,
    _supports_batch,
    count_notes,
    extraction_concurrency,
    model_location,
)
from aatm.omop.shards import check_shard, shard_dirs

if TYPE_CHECKING:
    from aatm.omop.builder import OmopExtractionPipeline

#: Seconds per extraction call assumed for the throughput in warnings, without a pilot.
ASSUMED_CALL_SECONDS = 10.0

#: Defaults of `OmopExtractionPipeline.run`, for the settings left out.
RUN_DEFAULTS: dict[str, Any] = {
    "chunk_size": 100,
    "batch": False,
    "max_batches": 20,
    "poll_every": 60.0,
    "shard": None,
}


class PlanSetting(BaseModel):
    """One effective setting of a planned run.

    Attributes:
        name: The setting, as an argument or attribute name.
        value: Its value in the run.
        source: ``default``, ``argument`` or ``set_rate_limits``.
        note: What it does, or how it was found.
    """

    name: str
    value: Any
    source: Literal["default", "argument", "set_rate_limits"]
    note: str = ""


class PlanWarning(BaseModel):
    """A combination of settings that costs time or money, and its fix."""

    message: str
    fix: str


class PlanEstimate(BaseModel):
    """What the notes left to do should take, scaled from a pilot run.

    Attributes:
        pilot_notes: Notes the pilot extracted, the base of the estimate.
        notes: Notes the estimate covers: those to extract, and those to build from
            stored results.
        calls: LLM calls, of extraction and coding.
        input_tokens: Input tokens.
        output_tokens: Output tokens.
        cost: Estimated cost, when the pilot was priced or ``prices`` are given.
        seconds: Estimated wall time, for a live run.
        notes_per_minute: Estimated throughput, for a live run.
        basis: How the time was estimated.
    """

    pilot_notes: int
    notes: int
    calls: int
    input_tokens: int
    output_tokens: int
    cost: Optional[float] = None
    seconds: Optional[float] = None
    notes_per_minute: Optional[float] = None
    basis: str = ""


class RunPlan(BaseModel):
    """The plan of a run: see `OmopExtractionPipeline.plan`.

    Attributes:
        output_dir: The output folder, if given.
        total: Notes in the source (in the shard, for a sharded run), when known.
        status: Notes per status in the output folder, from earlier runs.
        to_extract: Notes still to extract (one model call each), when known.
        to_build: Notes extracted by earlier runs, still to build.
        settings: The effective settings and where each came from.
        estimate: Calls, tokens, cost and time, when there is a pilot.
        warnings: Combinations to change, each with a fix.
    """

    output_dir: Optional[str] = None
    total: Optional[int] = None
    status: dict[str, int] = {}
    to_extract: Optional[int] = None
    to_build: int = 0
    settings: list[PlanSetting] = []
    estimate: Optional[PlanEstimate] = None
    warnings: list[PlanWarning] = []

    def setting(self, name: str) -> Optional[PlanSetting]:
        """The setting called `name`, if the plan has it."""
        return next((s for s in self.settings if s.name == name), None)

    def __str__(self) -> str:
        lines = ["Run plan"]
        if self.output_dir:
            lines[0] += f" for {self.output_dir}"
        if self.total is not None:
            notes = f"notes: {self.total:,} in the source"
        else:
            notes = "notes: unknown number in the source (pass total=)"
        built = self.status.get("built", 0)
        if built:
            notes += f", {built:,} already built"
        if self.to_extract is not None:
            notes += f"; {self.to_extract:,} to extract"
        if self.to_build:
            notes += f", {self.to_build:,} to build from stored results"
        lines.append(notes)

        lines.append("settings:")
        width = max((len(s.name) for s in self.settings), default=0)
        values = max((len(_show(s.value)) for s in self.settings), default=0)
        sources = max((len(s.source) for s in self.settings), default=0) + 2
        for s in self.settings:
            line = (
                f"  {s.name.ljust(width)}  {_show(s.value).ljust(values)}  "
                f"{f'[{s.source}]'.ljust(sources)}"
            )
            if s.note:
                line += f"  {s.note}"
            lines.append(line)

        estimate = self.estimate
        if estimate is not None:
            line = (
                f"estimate (from a pilot of {estimate.pilot_notes:,} notes): "
                f"{estimate.calls:,} LLM calls, {estimate.input_tokens:,} input and "
                f"{estimate.output_tokens:,} output tokens"
            )
            if estimate.cost is not None:
                line += f", cost {estimate.cost:,.2f}"
            if estimate.seconds is not None:
                line += f", about {_duration(estimate.seconds)}"
                if estimate.notes_per_minute:
                    line += f" ({estimate.notes_per_minute:,.0f} notes/min)"
            lines.append(line)
            if estimate.basis:
                lines.append(f"  {estimate.basis}")
        else:
            lines.append(
                "estimate: none without a pilot; run a few hundred notes into a "
                "scratch folder and pass pilot=<that folder>"
            )

        if self.warnings:
            lines.append("warnings:")
            for w in self.warnings:
                lines.append(f"  - {w.message}")
                lines.append(f"    fix: {w.fix}")
        else:
            lines.append("warnings: none")
        return "\n".join(lines)


def _show(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _given(value: Any, name: str) -> tuple[Any, Literal["default", "argument"]]:
    if value is None:
        return RUN_DEFAULTS[name], "default"
    return value, "argument"


def _pilot_summaries(pilot: Any) -> list[RunSummary]:
    """The run summaries of a pilot: a summary, a list of them, or an output folder."""
    if pilot is None:
        return []
    if isinstance(pilot, RunSummary):
        summaries = [pilot]
    elif isinstance(pilot, (str, Path)):
        summaries = OmopRun(pilot).summaries() if Path(pilot).exists() else []
    else:
        summaries = list(pilot)
    return [
        s
        for s in summaries
        if s.kind == "run" and s.stages["extraction"].items and s.seconds
    ]


def _coder_of(pipeline: "OmopExtractionPipeline") -> tuple[Any, bool]:
    """The pipeline's coder, unwrapped from its cache, and whether it is cached."""
    from aatm.omop.cache import CachedCoder

    coder = pipeline.builder.coder
    if isinstance(coder, CachedCoder):
        return coder.coder, True
    return coder, False


def _scaled(usage: TokenUsage, factor: float) -> tuple[float, float, float]:
    return (
        usage.calls * factor,
        usage.input_tokens * factor,
        usage.output_tokens * factor,
    )


def _estimate(
    pilots: list[RunSummary],
    to_extract: int,
    to_build: int,
    concurrency: Optional[int],
    batch: bool,
    rpm: Optional[float],
    prices: Optional[Prices],
) -> Optional[PlanEstimate]:
    extraction = TokenUsage()
    coding = TokenUsage()
    extracted = built = 0
    ext_seconds = build_seconds = call_seconds = 0.0
    timed = 0  # extraction calls of pilots that recorded their concurrency
    wall = done = 0.0
    priced: list[Optional[float]] = [None, None]  # extraction, coding
    for s in pilots:
        ext, cod = s.stages["extraction"], s.stages["coding"]
        extraction.merge(ext.usage)
        coding.merge(cod.usage)
        extracted += ext.items
        built += s.built
        build_seconds += s.stages["building"].seconds + cod.seconds
        wall += s.seconds
        done += s.done
        if (
            s.settings is not None
            and s.settings.max_concurrency
            and not s.settings.batch
        ):
            call_seconds += ext.seconds * s.settings.max_concurrency
            timed += ext.items
        ext_seconds += ext.seconds
        for i, stage in enumerate((ext, cod)):
            if stage.cost is not None:
                priced[i] = (priced[i] or 0.0) + stage.cost
    if not extracted:
        return None

    per_extraction = to_extract / extracted
    per_build = (to_extract + to_build) / built if built else 0.0
    e_calls, e_in, e_out = _scaled(extraction, per_extraction)
    c_calls, c_in, c_out = _scaled(coding, per_build)

    def cost_of(usage: TokenUsage, stored: Optional[float]) -> Optional[float]:
        return usage.cost(prices) if prices else stored

    cost: Optional[float] = None
    e_cost = cost_of(extraction, priced[0])
    c_cost = cost_of(coding, priced[1])
    if e_cost is not None or c_cost is not None:
        e_part = (e_cost or 0.0) * per_extraction
        live_pilot = not any(s.settings and s.settings.batch for s in pilots)
        if batch and live_pilot:
            e_part *= BATCH_DISCOUNT
        cost = e_part + (c_cost or 0.0) * per_build

    estimate = PlanEstimate(
        pilot_notes=extracted,
        notes=to_extract + to_build,
        calls=round(e_calls + c_calls),
        input_tokens=round(e_in + c_in),
        output_tokens=round(e_out + c_out),
        cost=cost,
    )
    if batch:
        estimate.basis = (
            "batch jobs return within 24 hours, often much sooner; coding and "
            "building run as their results come in"
        )
        return estimate

    if timed and concurrency:
        per_call = call_seconds / timed
        extract_s = to_extract * per_call / concurrency
        basis = (
            f"about {per_call:.1f} s per extraction call in the pilot, "
            f"{concurrency} in flight"
        )
        if rpm:
            calls_per_note = extraction.calls / extracted if extraction.calls else 1.0
            paced = to_extract * calls_per_note / rpm * 60
            if paced > extract_s:
                extract_s = paced
                basis += f", paced by requests_per_minute={rpm:g}"
        build_s = (to_extract + to_build) * build_seconds / built if built else 0.0
        seconds = max(extract_s, build_s)  # the stages overlap
        if build_s > extract_s:
            basis += "; coding and building are the slower stage"
    else:
        seconds = (to_extract + to_build) * wall / done if done else None
        basis = "at the pilot's throughput (it did not record its concurrency)"
    if seconds is not None:
        estimate.seconds = seconds
        if seconds:
            estimate.notes_per_minute = (to_extract + to_build) / seconds * 60
    estimate.basis = basis
    return estimate


def plan_run(
    pipeline: "OmopExtractionPipeline",
    source: Any = None,
    output_dir: Optional[str | Path] = None,
    *,
    chunk_size: Optional[int] = None,
    batch: Optional[bool] = None,
    max_batches: Optional[int] = None,
    poll_every: Optional[float] = None,
    shard: Optional[tuple[int, int]] = None,
    total: Optional[int] = None,
    pilot: Any = None,
    prices: Optional[Prices] = None,
) -> RunPlan:
    """The plan of `pipeline.run(source, output_dir, ...)`; see
    `OmopExtractionPipeline.plan`."""
    plan = RunPlan(output_dir=str(output_dir) if output_dir is not None else None)
    extractor = pipeline.extractor
    model_id = getattr(extractor, "model_id", None)
    location = model_location(extractor)
    hosted = location == "hosted"
    settings = plan.settings
    warn = plan.warnings

    # --- notes ---------------------------------------------------------------------
    chunk_size, chunk_source = _given(chunk_size, "chunk_size")
    batch, batch_source = _given(batch, "batch")
    shard_value, shard_source = _given(shard, "shard")
    count = (
        total
        if total is not None
        else (count_notes(source) if source is not None else None)
    )
    shards = 1
    if shard_value is not None:
        _, shards = check_shard(shard_value)
        if count is not None and total is None and shards > 1:
            count = -(-count // shards)
    plan.total = count

    folder = Path(output_dir) if output_dir is not None else None
    if folder is not None and (folder / LEDGER_FILE).exists():
        ledger = RunLedger(folder / LEDGER_FILE)
        try:
            plan.status = ledger.counts()
        finally:
            ledger.close()
    elif folder is not None and shard_dirs(folder):
        plan.status = OmopRun(folder).status()
    plan.to_build = plan.status.get("extracted", 0)
    if count is not None:
        started = sum(plan.status.values())
        plan.to_extract = max(count - started, 0) + plan.status.get("failed", 0)

    # --- settings ------------------------------------------------------------------
    limits = get_rate_limits(model_id) if isinstance(model_id, str) else None
    concurrency = extraction_concurrency(extractor)
    if batch:
        settings.append(
            PlanSetting(
                name="batch",
                value=True,
                source=batch_source,
                note="extraction through provider batch jobs, about half the price, "
                "results within 24 hours",
            )
        )
        mb, mb_source = _given(max_batches, "max_batches")
        settings.append(
            PlanSetting(
                name="max_batches",
                value=mb,
                source=mb_source,
                note="batch jobs pending at once",
            )
        )
        pe, pe_source = _given(poll_every, "poll_every")
        settings.append(
            PlanSetting(
                name="poll_every",
                value=pe,
                source=pe_source,
                note="seconds between polls of pending jobs",
            )
        )
    else:
        settings.append(
            PlanSetting(
                name="max_concurrency",
                value=concurrency.value,
                source=concurrency.source,
                note=f"extraction calls in flight; {concurrency.detail}",
            )
        )
    settings.append(
        PlanSetting(
            name="chunk_size",
            value=chunk_size,
            source=chunk_source,
            note="notes built and written together"
            + ("; one batch job per chunk" if batch else "")
            + ": it bounds memory, not extraction",
        )
    )
    if limits is not None:
        for name in ("requests_per_minute", "tokens_per_minute"):
            value = getattr(limits, name)
            if value is not None:
                settings.append(
                    PlanSetting(
                        name=name,
                        value=value,
                        source="set_rate_limits",
                        note=f"for {model_id}, in this process"
                        + (
                            ", divided among the shards by run_sharded"
                            if shards > 1
                            else ""
                        ),
                    )
                )
    if shard_value is not None:
        settings.append(
            PlanSetting(
                name="shard",
                value=f"{shard_value[0]}/{shard_value[1]}",
                source=shard_source,
                note="this process runs one shard of the dataset",
            )
        )

    coder, cached = _coder_of(pipeline)
    if coder is not None:
        name = type(coder).__name__
        coder_concurrency = getattr(coder, "max_concurrency", None)
        note = "codes the texts the mappings send to it"
        if coder_concurrency:
            note += (
                f", up to {coder_concurrency} texts at once in the build thread, "
                "on top of the extraction calls"
            )
        settings.append(
            PlanSetting(name="coder", value=name, source="argument", note=note)
        )
        settings.append(
            PlanSetting(
                name="cache",
                value=cached,
                source="argument" if cached else "default",
                note="each distinct text is coded once, across chunks, runs and rebuilds"
                if cached
                else "every text is coded again in each chunk, run and rebuild",
            )
        )
    labeller = getattr(extractor, "attribute_labeller", None)
    if labeller is not None:
        settings.append(
            PlanSetting(
                name="labeller.mode",
                value=labeller.mode,
                source="argument",
                note="attributes labelled by a decision model",
            )
        )
        settings.append(
            PlanSetting(
                name="labeller.min_probability",
                value=labeller.min_probability,
                source="argument"
                if labeller.min_probability is not None
                else "default",
                note="labels below it go to review",
            )
        )
        settings.append(
            PlanSetting(
                name="labeller.max_concurrency",
                value=labeller.max_concurrency,
                source="argument",
                note="mentions of one note labelled at once: each extraction call "
                "can add this many calls in flight",
            )
        )

    # --- estimate ------------------------------------------------------------------
    pilots = _pilot_summaries(pilot)
    if not pilots and pilot is None and folder is not None:
        pilots = _pilot_summaries(folder)
    rpm = limits.requests_per_minute if limits is not None else None
    if pilots and plan.to_extract is not None:
        plan.estimate = _estimate(
            pilots,
            plan.to_extract,
            plan.to_build,
            None if batch else concurrency.value,
            bool(batch),
            rpm,
            prices,
        )

    # --- warnings ------------------------------------------------------------------
    if batch and not _supports_batch(extractor):
        warn.append(
            PlanWarning(
                message="batch=True needs an extractor that submits provider batch "
                "jobs; this run would be refused",
                fix="use SimpleLLMExtractor with an openai, anthropic or gemini "
                "model, or run live without batch=True",
            )
        )
    hosted_default = DEFAULT_CONCURRENCY["hosted"]
    if hosted and not batch and concurrency.value < hosted_default:
        per_call = ASSUMED_CALL_SECONDS
        basis = f"at about {per_call:.0f} s per call"
        if plan.estimate is not None and plan.estimate.notes_per_minute:
            rate = plan.estimate.notes_per_minute
            basis = "from the pilot"
        else:
            rate = concurrency.value * 60 / per_call
        warn.append(
            PlanWarning(
                message=(
                    f"max_concurrency={concurrency.value} with a hosted API: about "
                    f"{rate:,.0f} notes/min ({basis}), far below what the API serves"
                ),
                fix=(
                    f"set max_concurrency={hosted_default} or more, with "
                    "aatm.llm.set_rate_limits to stay under your quota; or batch=True "
                    "if the run can wait a day"
                ),
            )
        )
    if hosted and shards > 1:
        warn.append(
            PlanWarning(
                message="shards with a hosted API share one rate limit: they add "
                "processes, not quota",
                fix="raise max_concurrency and the rate limits of one process "
                "instead; shard for a local model or to use several machines",
            )
        )
    if (
        location == "local"
        and not batch
        and concurrency.value > 2 * DEFAULT_CONCURRENCY["local"]
    ):
        warn.append(
            PlanWarning(
                message=(
                    f"max_concurrency={concurrency.value} with a local server: calls "
                    "queue on the server rather than run faster"
                ),
                fix=(
                    "match max_concurrency to what the server serves at once (its "
                    "batch size), and shard across GPUs or machines to go faster"
                ),
            )
        )
    if coder is not None and not cached and type(coder).__name__ == "AgenticCoder":
        warn.append(
            PlanWarning(
                message="the agent coder without cache=True pays again for repeated "
                "texts in every chunk, run and rebuild",
                fix="pass cache=True to OmopExtractionPipeline",
            )
        )
    if count is None:
        warn.append(
            PlanWarning(
                message="the number of notes in the source is unknown (a CSV file, a "
                "query or an iterator): no estimate, and no time left in the logs",
                fix="pass total=<number of notes> to plan and run",
            )
        )
    if (
        plan.estimate is not None
        and plan.estimate.cost is None
        and isinstance(model_id, str)
    ):
        warn.append(
            PlanWarning(
                message="the pilot has no prices: no cost estimate",
                fix=f'pass prices={{"{model_id}": (input, output)}}, per million tokens',
            )
        )
    return plan
