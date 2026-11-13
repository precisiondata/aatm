"""The ``aatm omop`` commands: run, rebuild and inspect OMOP extraction runs.

They are a thin layer over `aatm.omop.OmopExtractionPipeline` and `aatm.omop.OmopRun`,
for shell scripts, cron jobs and schedulers (SLURM, Kubernetes):

```bash
aatm omop run notes.parquet output/full \\
  --definitions condition_occurrence --definitions ./body_weight.yaml \\
  --model openai/gpt-5-mini --coder agent \\
  --columns note_text=text --chunk-size 200 --cache \\
  --price openai/gpt-5-mini=0.25,2.00
aatm omop rebuild output/full --definitions ...
aatm omop status output/full
aatm omop failures output/full
```

## Options

Every option of ``run`` and ``rebuild`` can also be written in a YAML file given with
``--config``, under its name with underscores (``chunk_size: 200``). Flags override the
file. Lists and mappings are written as YAML:

```yaml
source: notes.parquet
output: output/full
definitions: [condition_occurrence, drug_exposure, ./body_weight.yaml]
model: openai/gpt-5-mini
coder: agent
columns: {note_text: text, patient_id: person_id}
prices: {openai/gpt-5-mini: [0.25, 2.00]}
chunk_size: 200
cache: true
```

``run`` saves the options it used in ``options.yaml`` of its output folder (the
shard's folder, with ``--shard``). ``rebuild`` starts from them, so only what changed
needs a flag: the model and coder of the run are kept unless given again.

## Exit codes

``0`` when every note is built, ``2`` when the run completed but some notes are
recorded as failed (see ``aatm omop failures``), ``1`` for an error that stopped the
run, ``130`` when stopped with Ctrl+C. Running the same command again resumes.

Progress is logged to standard error. Standard output gets the path of the run's
summary, or with ``--json`` the summary itself, for scripts.
"""

from __future__ import annotations

import asyncio
import json
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional

import typer
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console

if TYPE_CHECKING:
    from aatm.omop.builder import OmopExtractionPipeline
    from aatm.omop.runner import RunSummary

OPTIONS_FILE = "options.yaml"
"""Where `run` saves its options, in the output folder."""

DEFAULT_VOCABULARY = ".aatm/omop.db"
RUN_CHUNK_SIZE = 100
REBUILD_CHUNK_SIZE = 1000

EXIT_FAILED_NOTES = 2
EXIT_INTERRUPTED = 130

out = Console(highlight=False, soft_wrap=True)
err = Console(stderr=True, highlight=False, soft_wrap=True)

omop_app = typer.Typer(help="Run, rebuild and inspect OMOP extraction runs")


class RunOptions(BaseModel):
    """The options of ``aatm omop run`` and ``rebuild``, from a config file and flags.

    Field names are the flags' names with underscores. Unknown names are refused, so
    a misspelt key in a config file fails at once.
    """

    model_config = ConfigDict(extra="forbid")

    source: Optional[str] = None
    output: Optional[str] = None
    sql: Optional[str] = None
    db: Optional[str] = None
    definitions: list[str] = []
    model: Optional[str] = None
    vocabulary: str = DEFAULT_VOCABULARY
    coder: Literal["none", "mapper", "agent"] = "none"
    coder_model: Optional[str] = None
    retriever: Optional[str] = None
    selector: Optional[str] = None
    max_tool_calls: int = 8
    min_score: Optional[float] = None
    labeller: Optional[str] = None
    labeller_mode: Literal["verify", "replace"] = "verify"
    min_probability: Optional[float] = None
    columns: dict[str, str] = {}
    max_concurrency: Optional[int] = None
    chunk_size: Optional[int] = None
    max_attempts: int = 3
    cache: bool = False
    prices: dict[str, tuple[float, float]] = {}
    batch: bool = False
    poll_every: float = 60.0
    max_batches: int = 20
    shard: Optional[str] = None
    limit: Optional[int] = None
    total: Optional[int] = None
    log_every: float = 60.0


_DEFAULTS = RunOptions()

# Options of `run` that `rebuild` takes from the recorded ones, and that a new run
# of a folder needs to match.
_PIPELINE_OPTIONS = (
    "definitions",
    "model",
    "vocabulary",
    "coder",
    "coder_model",
    "retriever",
    "selector",
    "max_tool_calls",
    "min_score",
    "labeller",
    "labeller_mode",
    "min_probability",
    "cache",
    "prices",
)


class OptionsError(ValueError):
    """Options that cannot be used together, or a config file that cannot be read."""


def _default(name: str) -> str:
    """How `--help` shows the default of an option."""
    value = getattr(_DEFAULTS, name)
    if value is None or value == [] or value == {}:
        return "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


# --- parsing flags --------------------------------------------------------------------


def parse_pairs(values: list[str], what: str) -> dict[str, str]:
    """``["a=b", "c=d"]`` as ``{"a": "b", "c": "d"}``."""
    pairs = {}
    for value in values:
        key, sep, item = value.partition("=")
        if not sep or not key.strip() or not item.strip():
            raise OptionsError(f"{what} {value!r} must be written as <name>=<value>")
        pairs[key.strip()] = item.strip()
    return pairs


def parse_prices(values: list[str]) -> dict[str, tuple[float, float]]:
    """``["openai/gpt-5-mini=0.25,2.00"]`` as ``{"openai/gpt-5-mini": (0.25, 2.0)}``."""
    prices = {}
    for value in values:
        model, sep, numbers = value.rpartition("=")
        parts = numbers.split(",")
        try:
            if not sep or not model.strip() or len(parts) != 2:
                raise ValueError
            prices[model.strip()] = (float(parts[0]), float(parts[1]))
        except ValueError:
            raise OptionsError(
                f"--price {value!r} must be written as <model>=<input>,<output>, "
                "in dollars per million tokens, e.g. openai/gpt-5-mini=0.25,2.00"
            ) from None
    return prices


def parse_shard(value: str) -> tuple[int, int]:
    """``"3/8"`` as ``(3, 8)``: shard 3 of 8, counted from 0."""
    from aatm.omop.shards import check_shard

    index, sep, count = value.partition("/")
    try:
        if not sep:
            raise ValueError
        shard = (int(index), int(count))
    except ValueError:
        raise OptionsError(
            f"--shard {value!r} must be written as <index>/<count>, e.g. 3/8"
        ) from None
    try:
        return check_shard(shard)
    except ValueError as e:
        raise OptionsError(str(e)) from None


def read_config(path: Path) -> dict[str, Any]:
    """The options of a YAML config file, with ``-`` in names read as ``_``."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise OptionsError(f"No such config file: {path}") from None
    except yaml.YAMLError as e:
        raise OptionsError(f"Cannot read {path}: {e}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise OptionsError(f"{path} must hold a mapping of options, e.g. 'model: ...'")
    data = {str(k).replace("-", "_"): v for k, v in data.items()}
    if isinstance(data.get("definitions"), str):
        data["definitions"] = [data["definitions"]]
    return data


def resolve_options(*layers: dict[str, Any]) -> RunOptions:
    """Combine option layers, later ones overriding earlier ones; ``None`` is unset."""
    merged: dict[str, Any] = {}
    for layer in layers:
        merged.update({k: v for k, v in layer.items() if v is not None})
    try:
        return RunOptions.model_validate(merged)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in error['loc']) or 'options'}: {error['msg']}"
            for error in e.errors()
        )
        raise OptionsError(f"Invalid options: {problems}") from None


def save_options(options: RunOptions, folder: Path) -> Path:
    """Write the options a run used to its output folder (see `OPTIONS_FILE`)."""
    folder.mkdir(parents=True, exist_ok=True)
    data = options.model_dump(mode="json", exclude_defaults=True)
    path = folder / OPTIONS_FILE
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return path


def recorded_options(output: Path) -> dict[str, Any]:
    """The pipeline options a run saved in `output`, or in its first shard."""
    from aatm.omop.shards import shard_dirs

    for folder in [output, *shard_dirs(output)]:
        path = folder / OPTIONS_FILE
        if path.exists():
            data = read_config(path)
            return {k: v for k, v in data.items() if k in _PIPELINE_OPTIONS}
    return {}


# --- building the pipeline ------------------------------------------------------------


def _set_min_score(value: Any, min_score: float) -> None:
    """Give every ``map`` source without a ``min_score`` this one."""
    from aatm.omop.mapping import MapSpec

    if isinstance(value, MapSpec):
        if value.min_score is None:
            value.min_score = min_score
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            _set_min_score(getattr(value, name), min_score)
    elif isinstance(value, dict):
        for item in value.values():
            _set_min_score(item, min_score)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _set_min_score(item, min_score)


def load_definitions(options: RunOptions) -> Any:
    """The combined entity definitions, with `min_score` applied."""
    from aatm.omop.builder import _load_definitions

    if not options.definitions:
        raise OptionsError(
            "Give the entity definitions with --definitions: a predefined name such "
            "as condition_occurrence, or a YAML file (repeat it for several)"
        )
    definitions = _load_definitions(list(options.definitions))
    if options.min_score is not None:
        _set_min_score(definitions, options.min_score)
    return definitions


def make_coder(options: RunOptions) -> Any:
    """The coder of ``map`` sources that `options` asks for, or ``None``."""
    if options.selector is not None and options.coder != "mapper":
        raise OptionsError("--selector only applies with --coder mapper")
    if options.coder == "none":
        return None
    retriever: Any = options.retriever
    if options.coder == "mapper":
        from aatm.terminology_mapper import TerminologyMapper

        if retriever == "none":
            raise OptionsError("--coder mapper needs a retriever")
        return TerminologyMapper(retriever=retriever, selector=options.selector)
    from aatm.omop.coding import AgenticCoder

    model = options.coder_model or options.model
    if model is None:
        raise OptionsError("--coder agent needs --coder-model (or --model)")
    return AgenticCoder(
        model,
        retriever=False if retriever == "none" else retriever,
        vocabulary=options.vocabulary,
        max_tool_calls=options.max_tool_calls,
    )


def make_labeller(options: RunOptions) -> Any:
    """The decision model that labels attributes, if `options` asks for one."""
    if options.labeller is None:
        return None
    from aatm.extractors import DecisionAttributeLabeller

    return DecisionAttributeLabeller(
        options.labeller,
        mode=options.labeller_mode,
        min_probability=options.min_probability,
    )


def make_extractor(options: RunOptions, definitions: Any) -> Any:
    """The extractor of the run (tests replace it with a fake one)."""
    from aatm.extractors import SimpleLLMExtractor

    if options.model is None:
        raise OptionsError(
            "Give the extraction model with --model, e.g. openai/gpt-5-mini"
        )
    kwargs: dict[str, Any] = {}
    if options.max_concurrency is not None:
        kwargs["max_concurrency"] = options.max_concurrency
    labeller = make_labeller(options)
    if labeller is not None:
        kwargs["attribute_labeller"] = labeller
    return SimpleLLMExtractor(
        model_id=options.model, entity_definitions=definitions, **kwargs
    )


def make_vocabulary(options: RunOptions) -> Any:
    """The vocabulary database (tests replace it with a fake one)."""
    return options.vocabulary


def make_pipeline(options: RunOptions) -> "OmopExtractionPipeline":
    """The pipeline `options` describe."""
    from aatm.omop.builder import OmopExtractionPipeline

    if options.labeller is None and options.min_probability is not None:
        raise OptionsError("--min-probability only applies with --labeller")
    definitions = load_definitions(options)
    return OmopExtractionPipeline(
        definitions,
        extractor=make_extractor(options, definitions),
        vocabulary=make_vocabulary(options),
        coder=make_coder(options),
        cache=True if options.cache else None,
    )


# --- sources --------------------------------------------------------------------------


def _connect(url: str) -> Any:
    """A database connection for a SQLAlchemy URL; ``sqlite:///`` works without it."""
    try:
        from sqlalchemy import create_engine  # type: ignore[import-not-found]
    except ImportError:
        if url.startswith("sqlite:///"):
            import sqlite3

            return sqlite3.connect(url.removeprefix("sqlite:///"))
        raise OptionsError(
            "--db needs SQLAlchemy and the database's driver: pip install sqlalchemy "
            "(e.g. with psycopg for PostgreSQL)"
        ) from None
    return create_engine(url)


def make_source(options: RunOptions) -> tuple[Any, Optional[dict[str, str]]]:
    """The notes to read and the column renames still to apply to them."""
    from aatm.omop.runner import _iter_notes, notes_from_sql

    columns = options.columns or None
    if options.sql is not None:
        if options.source is not None:
            raise OptionsError("Give either a source file or --sql, not both")
        if options.db is None:
            raise OptionsError("--sql needs --db, a SQLAlchemy URL of the database")
        source: Any = notes_from_sql(options.sql, _connect(options.db))
    elif options.db is not None:
        raise OptionsError("--db only applies with --sql")
    elif options.source is None:
        raise OptionsError(
            "Give the notes: a CSV or Parquet file, a folder of Parquet files, or "
            "--sql with --db"
        )
    else:
        source = options.source
        if not Path(source).exists():
            raise OptionsError(f"No such file or folder: {source}")
    if options.limit is not None:
        if options.limit < 1:
            raise OptionsError("--limit must be >= 1")
        # The first notes of the source, already renamed.
        notes = _iter_notes(source, columns, min(options.limit, 10_000))
        return list(islice(notes, options.limit)), None
    return source, columns


# --- running --------------------------------------------------------------------------


def _summary_path(folder: Path) -> Optional[Path]:
    """The summary file of the latest run or rebuild in `folder`."""
    from aatm.omop.runner import RUNS

    files = sorted((folder / RUNS).glob("*.json")) if (folder / RUNS).is_dir() else []
    return files[-1] if files else None


def _failed_notes(folder: Path) -> int:
    from aatm.omop.runner import OmopRun

    try:
        return OmopRun(folder).status().get("failed", 0)
    except FileNotFoundError:
        return 0


def _finish(summary: "RunSummary", folder: Path, as_json: bool) -> None:
    """Print the summary path (or the summary), and exit 2 if notes failed."""
    path = _summary_path(folder)
    if as_json:
        data = summary.model_dump(mode="json")
        data["summary_path"] = str(path) if path else None
        out.print(json.dumps(data, indent=2), markup=False)
    elif path is not None:
        out.print(str(path), markup=False)
    failed = _failed_notes(folder)
    if failed:
        err.print(
            f"{failed:,} notes failed: see 'aatm omop failures {folder}'; run the "
            "same command again to try them again",
            markup=False,
        )
        raise typer.Exit(code=EXIT_FAILED_NOTES)


def _execute(work: Any, folder: Path, as_json: bool, command: str) -> None:
    """Run a coroutine, mapping errors and Ctrl+C to exit codes."""
    try:
        summary = asyncio.run(work)
    except KeyboardInterrupt:
        err.print(
            f"Stopped. Run the same command again to resume the {command}.",
            markup=False,
        )
        raise typer.Exit(code=EXIT_INTERRUPTED) from None
    except (OptionsError, ValueError, TypeError, FileNotFoundError) as e:
        err.print(f"Error: {e}", markup=False)
        raise typer.Exit(code=1) from None
    _finish(summary, folder, as_json)


def _fail(message: str) -> None:
    err.print(f"Error: {message}", markup=False)
    raise typer.Exit(code=1)


# --- the commands ---------------------------------------------------------------------

ConfigOption = Annotated[
    Optional[Path],
    typer.Option(
        "--config",
        help="A YAML file of options (names with underscores); flags override it",
        show_default="none",
    ),
]
DefinitionsOption = Annotated[
    Optional[list[str]],
    typer.Option(
        "--definitions",
        "-d",
        help=(
            "Entity definitions: a predefined name (e.g. condition_occurrence) or a "
            "YAML file; repeat for several"
        ),
        show_default="none",
    ),
]
ModelOption = Annotated[
    Optional[str],
    typer.Option(
        "--model",
        "-m",
        help="The extraction model, e.g. openai/gpt-5-mini",
        show_default=_default("model"),
    ),
]
VocabularyOption = Annotated[
    Optional[str],
    typer.Option(
        "--vocabulary",
        help="The OMOP vocabulary database built by 'aatm init'",
        show_default=_default("vocabulary"),
    ),
]
CoderOption = Annotated[
    Optional[str],
    typer.Option(
        "--coder",
        help=(
            "How 'map' sources are coded: none (stored as 0), mapper (the terminology "
            "mapper) or agent (a concept search agent)"
        ),
        show_default=_default("coder"),
    ),
]
CoderModelOption = Annotated[
    Optional[str],
    typer.Option(
        "--coder-model",
        help="The agent's model (--coder agent)",
        show_default="--model",
    ),
]
RetrieverOption = Annotated[
    Optional[str],
    typer.Option(
        "--retriever",
        help=(
            "The retriever of the coder, e.g. text-embedding-3-small; 'none' gives "
            "the agent the vocabulary tools only"
        ),
        show_default="the one recorded by 'aatm init'",
    ),
]
SelectorOption = Annotated[
    Optional[str],
    typer.Option(
        "--selector",
        help="The mapper's selector, a registry name (--coder mapper)",
        show_default="the first result",
    ),
]
MaxToolCallsOption = Annotated[
    Optional[int],
    typer.Option(
        "--max-tool-calls",
        help="Tool calls per text of the agent (--coder agent)",
        show_default=_default("max_tool_calls"),
    ),
]
MinScoreOption = Annotated[
    Optional[float],
    typer.Option(
        "--min-score",
        help=(
            "Codes scored lower are stored as 0, for 'map' sources without a "
            "min_score of their own"
        ),
        show_default=_default("min_score"),
    ),
]
LabellerOption = Annotated[
    Optional[str],
    typer.Option(
        "--labeller",
        help=(
            "A decision model that labels the attributes of mentions, e.g. "
            "openrouter/typesafe/jev-1.13"
        ),
        show_default=_default("labeller"),
    ),
]
LabellerModeOption = Annotated[
    Optional[str],
    typer.Option(
        "--labeller-mode",
        help=(
            "verify: keep a label only when the decision model agrees; replace: "
            "the decision model labels instead"
        ),
        show_default=_default("labeller_mode"),
    ),
]
MinProbabilityOption = Annotated[
    Optional[float],
    typer.Option(
        "--min-probability",
        help="Labels the decision model is less sure of go to review",
        show_default=_default("min_probability"),
    ),
]
CacheOption = Annotated[
    Optional[bool],
    typer.Option(
        "--cache/--no-cache",
        help="Code each distinct text once, across chunks and runs (.aatm/coding_cache.db)",
        show_default=_default("cache"),
    ),
]
PriceOption = Annotated[
    Optional[list[str]],
    typer.Option(
        "--price",
        help=(
            "A model's price per million input and output tokens, for the cost: "
            "<model>=<input>,<output>; repeat for several"
        ),
        show_default="none",
    ),
]
LogEveryOption = Annotated[
    Optional[float],
    typer.Option(
        "--log-every",
        help="Seconds between progress logs",
        show_default=_default("log_every"),
    ),
]
JsonOption = Annotated[
    bool,
    typer.Option("--json", help="Print the summary as JSON on standard output"),
]


@omop_app.command(
    "run",
    help=(
        "Extract notes into OMOP tables, in chunks, resuming where an earlier run "
        "stopped. Exit codes: 0 every note built, 2 some notes failed, 1 error, "
        "130 stopped"
    ),
)
def run_command(
    source: Annotated[
        Optional[str],
        typer.Argument(
            help=(
                "The notes: a CSV or Parquet file or a folder of Parquet files "
                "(with --sql, the only argument is the output folder)"
            ),
            show_default=False,
        ),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Argument(help="The output folder", show_default=False),
    ] = None,
    config: ConfigOption = None,
    definitions: DefinitionsOption = None,
    model: ModelOption = None,
    sql: Annotated[
        Optional[str],
        typer.Option(
            "--sql",
            help="Read the notes from this query instead of a file (needs --db)",
            show_default="none",
        ),
    ] = None,
    db: Annotated[
        Optional[str],
        typer.Option(
            "--db",
            help="The SQLAlchemy URL of the database of --sql, e.g. sqlite:///cdm.db",
            show_default="none",
        ),
    ] = None,
    columns: Annotated[
        Optional[list[str]],
        typer.Option(
            "--columns",
            "-c",
            help="Rename a source column to a note field: <column>=<field>, e.g. note_text=text",
            show_default="none",
        ),
    ] = None,
    vocabulary: VocabularyOption = None,
    coder: CoderOption = None,
    coder_model: CoderModelOption = None,
    retriever: RetrieverOption = None,
    selector: SelectorOption = None,
    max_tool_calls: MaxToolCallsOption = None,
    min_score: MinScoreOption = None,
    labeller: LabellerOption = None,
    labeller_mode: LabellerModeOption = None,
    min_probability: MinProbabilityOption = None,
    max_concurrency: Annotated[
        Optional[int],
        typer.Option(
            "--max-concurrency",
            help="Extraction calls in flight at once",
            show_default="16 for a hosted API, 4 for a local server",
        ),
    ] = None,
    chunk_size: Annotated[
        Optional[int],
        typer.Option(
            "--chunk-size",
            help="Notes built and written together",
            show_default=str(RUN_CHUNK_SIZE),
        ),
    ] = None,
    max_attempts: Annotated[
        Optional[int],
        typer.Option(
            "--max-attempts",
            help="Times a failing note is tried, across runs",
            show_default=_default("max_attempts"),
        ),
    ] = None,
    cache: CacheOption = None,
    price: PriceOption = None,
    batch: Annotated[
        Optional[bool],
        typer.Option(
            "--batch/--live",
            help="Extract through the provider's batch endpoint: about half the price, within 24 hours",
            show_default="live",
        ),
    ] = None,
    poll_every: Annotated[
        Optional[float],
        typer.Option(
            "--poll-every",
            help="Seconds between polls of pending batch jobs",
            show_default=_default("poll_every"),
        ),
    ] = None,
    max_batches: Annotated[
        Optional[int],
        typer.Option(
            "--max-batches",
            help="Batch jobs pending at once",
            show_default=_default("max_batches"),
        ),
    ] = None,
    shard: Annotated[
        Optional[str],
        typer.Option(
            "--shard",
            help=(
                "Process only shard <index>/<count> (from 0), e.g. 3/8; the output "
                "goes to <output>/shard-003-of-008"
            ),
            show_default="the whole dataset",
        ),
    ] = None,
    limit: Annotated[
        Optional[int],
        typer.Option(
            "--limit",
            help="Process only the first N notes of the source, e.g. for a pilot",
            show_default="all",
        ),
    ] = None,
    total: Annotated[
        Optional[int],
        typer.Option(
            "--total",
            help="Notes in the source, for the time left (CSV files and queries)",
            show_default="counted for Parquet",
        ),
    ] = None,
    log_every: LogEveryOption = None,
    as_json: JsonOption = False,
) -> None:
    """Run `OmopExtractionPipeline.run` with options from flags and a config file."""
    from aatm.omop.shards import shard_dir

    try:
        flags = {
            "source": source,
            "output": output,
            "sql": sql,
            "db": db,
            "definitions": definitions or None,
            "model": model,
            "vocabulary": vocabulary,
            "coder": coder,
            "coder_model": coder_model,
            "retriever": retriever,
            "selector": selector,
            "max_tool_calls": max_tool_calls,
            "min_score": min_score,
            "labeller": labeller,
            "labeller_mode": labeller_mode,
            "min_probability": min_probability,
            "columns": parse_pairs(columns, "--columns") if columns else None,
            "max_concurrency": max_concurrency,
            "chunk_size": chunk_size,
            "max_attempts": max_attempts,
            "cache": cache,
            "prices": parse_prices(price) if price else None,
            "batch": batch,
            "poll_every": poll_every,
            "max_batches": max_batches,
            "shard": shard,
            "limit": limit,
            "total": total,
            "log_every": log_every,
        }
        if sql is not None and source is not None and output is None:
            # With --sql, the one argument is the output folder.
            flags["source"], flags["output"] = None, source
        options = resolve_options(read_config(config) if config else {}, flags)
        if options.output is None:
            raise OptionsError("Give the output folder")
        folder = Path(options.output)
        shard_pair = parse_shard(options.shard) if options.shard else None
        if shard_pair is not None:
            folder = shard_dir(folder, *shard_pair)
        pipeline = make_pipeline(options)
        notes, renames = make_source(options)
    except (OptionsError, ValueError, TypeError, FileNotFoundError) as e:
        _fail(str(e))
        return
    save_options(options, folder)
    work = pipeline.run(
        notes,
        folder,
        chunk_size=options.chunk_size or RUN_CHUNK_SIZE,
        columns=renames,
        max_attempts=options.max_attempts,
        total=options.total,
        prices=options.prices or None,
        log_every=options.log_every,
        batch=options.batch,
        poll_every=options.poll_every,
        max_batches=options.max_batches,
        shard=shard_pair,
    )
    _execute(work, folder, as_json, "run")


@omop_app.command(
    "rebuild",
    help=(
        "Build the tables of a run again from its stored extraction results, with no "
        "extraction call (after changing the omop blocks or the coder). Options not "
        "given are those the run recorded in options.yaml"
    ),
)
def rebuild_command(
    output: Annotated[
        Path, typer.Argument(help="The output folder of the run", show_default=False)
    ],
    config: ConfigOption = None,
    definitions: DefinitionsOption = None,
    model: ModelOption = None,
    vocabulary: VocabularyOption = None,
    coder: CoderOption = None,
    coder_model: CoderModelOption = None,
    retriever: RetrieverOption = None,
    selector: SelectorOption = None,
    max_tool_calls: MaxToolCallsOption = None,
    min_score: MinScoreOption = None,
    labeller: LabellerOption = None,
    labeller_mode: LabellerModeOption = None,
    min_probability: MinProbabilityOption = None,
    chunk_size: Annotated[
        Optional[int],
        typer.Option(
            "--chunk-size",
            help="Notes built and written together",
            show_default=str(REBUILD_CHUNK_SIZE),
        ),
    ] = None,
    cache: CacheOption = None,
    price: PriceOption = None,
    log_every: LogEveryOption = None,
    as_json: JsonOption = False,
) -> None:
    """Run `OmopExtractionPipeline.rebuild` on the folder of an earlier run."""
    try:
        if not output.is_dir():
            raise OptionsError(f"No such folder: {output}")
        flags = {
            "definitions": definitions or None,
            "model": model,
            "vocabulary": vocabulary,
            "coder": coder,
            "coder_model": coder_model,
            "retriever": retriever,
            "selector": selector,
            "max_tool_calls": max_tool_calls,
            "min_score": min_score,
            "labeller": labeller,
            "labeller_mode": labeller_mode,
            "min_probability": min_probability,
            "chunk_size": chunk_size,
            "cache": cache,
            "prices": parse_prices(price) if price else None,
            "log_every": log_every,
        }
        file = read_config(config) if config else {}
        # A config file of `run` may hold source options a rebuild does not use.
        used = (*_PIPELINE_OPTIONS, "chunk_size", "log_every")
        file = {k: v for k, v in file.items() if k in used}
        options = resolve_options(recorded_options(output), file, flags)
        pipeline = make_pipeline(options)
    except (OptionsError, ValueError, TypeError, FileNotFoundError) as e:
        _fail(str(e))
        return
    work = pipeline.rebuild(
        output,
        chunk_size=options.chunk_size or REBUILD_CHUNK_SIZE,
        prices=options.prices or None,
        log_every=options.log_every,
    )
    _execute(work, output, as_json, "rebuild")


def _open_run(output: Path) -> Any:
    from aatm.omop.runner import OmopRun

    try:
        return OmopRun(output)
    except (FileNotFoundError, ValueError) as e:
        _fail(str(e))


def _last_summary(run: Any) -> Optional["RunSummary"]:
    """The latest summary, or the latest of each shard combined."""
    from aatm.omop.shards import combine_summaries

    if not run.sharded:
        summaries = run.summaries()
        return summaries[-1] if summaries else None
    latest = []
    for folder in run.folders:
        path = _summary_path(folder)
        if path is not None:
            from aatm.omop.runner import RunSummary

            latest.append(
                RunSummary.model_validate_json(path.read_text(encoding="utf-8"))
            )
    return combine_summaries(latest, run.output_dir) if latest else None


@omop_app.command(
    "status", help="Notes built and failed, issues by reason, and the last summary"
)
def status_command(
    output: Annotated[Path, typer.Argument(help="The output folder of a run")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the status as JSON")
    ] = False,
) -> None:
    """Print what an output folder holds and how its last run or rebuild went."""
    run = _open_run(output)
    counts = run.status()
    issues = run.issues()
    reasons = (
        {str(k): int(v) for k, v in issues["reason"].value_counts().items()}
        if not issues.empty
        else {}
    )
    summary = _last_summary(run)
    if as_json:
        data = {
            "output_dir": str(output),
            "notes": counts,
            "rows": {name: run.dataset(name).count_rows() for name in run.tables},
            "issues": reasons,
            "last_summary": summary.model_dump(mode="json") if summary else None,
        }
        out.print(json.dumps(data, indent=2), markup=False)
        return
    notes = ", ".join(f"{n:,} {status}" for status, n in sorted(counts.items()))
    out.print(f"{output}: {notes or 'no notes'}", markup=False)
    rows = ", ".join(
        f"{name} {run.dataset(name).count_rows():,}" for name in run.tables
    )
    out.print(f"rows: {rows or 'none'}", markup=False)
    if reasons:
        out.print(
            f"issues: {sum(reasons.values()):,} ("
            + ", ".join(f"{reason} {n:,}" for reason, n in reasons.items())
            + ")",
            markup=False,
        )
    else:
        out.print("issues: none", markup=False)
    if summary is not None:
        out.print("\nlast " + str(summary), markup=False)


@omop_app.command("failures", help="The notes that failed: stage, error and attempts")
def failures_command(
    output: Annotated[Path, typer.Argument(help="The output folder of a run")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the failures as JSON records")
    ] = False,
) -> None:
    """Print the failed notes of an output folder."""
    run = _open_run(output)
    failures = run.failures()
    if as_json:
        out.print(failures.to_json(orient="records", indent=2), markup=False)
        return
    if failures.empty:
        out.print(f"No failed notes in {output}", markup=False)
        return
    from rich.table import Table

    table = Table("note_id", "stage", "attempts", "error", "updated_at")
    for row in failures.itertuples(index=False):
        table.add_row(
            str(row.note_id),
            str(row.stage),
            str(row.attempts),
            str(row.error),
            str(row.updated_at),
        )
    out.print(table)
    out.print(f"{len(failures):,} failed notes", markup=False)
