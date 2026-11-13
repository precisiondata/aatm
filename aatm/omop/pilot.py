"""Try entity definitions on a fixed sample of notes, and compare the attempts.

Most fixes happen in the wording of the entity definitions: run a pilot, read the
issues and rows, change a definition, run again. A `Pilot` keeps those attempts apart,
as numbered versions of one workspace, all on the same notes:

```python
from aatm.omop import OmopExtractionPipeline, Pilot

pilot = Pilot("pilot/", source="notes.parquet", sample=50, seed=0)
pipeline = OmopExtractionPipeline("ecog_performance_status", model_id="openai/gpt-5-mini")
v = await pilot.run(pipeline)    # pilot/v001-3f9a2c1/

# ... change a definition's wording, then:
pipeline = OmopExtractionPipeline("my_ecog.yaml", model_id="openai/gpt-5-mini")
v = await pilot.run(pipeline)    # pilot/v002-8be01d4/
print(pilot.compare("v001", "v002"))
```

## The sample

The notes are picked once, at random with `seed`, from `source` (any source
`OmopExtractionPipeline.run` reads), the first time the pilot needs them. Their ids go
to ``sample.json`` and the notes themselves to ``notes.jsonl``, so every version sees
the same notes, and later calls need no `source`.

## Versions

`Pilot.run` picks what to do from the fingerprints the runner already keeps (see
`aatm.omop.runner.extraction_fingerprint` and `aatm.omop.runner.mapping_fingerprint`):

- **new extraction** (other definitions, model, instructions or labeller): a new
  version folder, extracted from scratch;
- **same extraction, other mapping** (an ``omop`` block, ``dayfirst`` or the coder):
  a new version, built from the stored results of the latest version with that
  extraction, with no extraction call (`OmopExtractionPipeline.rebuild`);
- **nothing changed**: the version that matches is reused, and finished if a run of it
  stopped partway.

When the pipeline's coder has no cache, the pilot gives it one, ``coding_cache.db`` in
the workspace, shared by every version: a text coded once is not coded again.

Each version records what changed from the previous one: the model, the extractor, the
coder, and a diff of the definitions (``definitions.yaml`` in each folder). `Pilot.log`
lists them.

## Comparing

`Pilot.compare` puts two versions side by side: rows per table, issues per reason,
the mentions added, removed or changed in each note (with the line of the note), the
codes that changed, and tokens and cost per note. Each part is a DataFrame; printing
the comparison gives a readable report.

## Workspace

```text
pilot/
  sample.json               the note ids, source, sample size and seed
  notes.jsonl               the sampled notes
  coding_cache.db           shared by the versions, when the coder had no cache
  v001-3f9a2c1/
    version.json            fingerprints, model, coder, what changed
    definitions.yaml        the definitions of this version
    ledger.db, measurement/, issues/, ...   the run (see aatm.omop.runner)
  v002-8be01d4/
```
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

import contextlib
import difflib
import random
import sqlite3

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from aatm.data_models.extractors import EntityValidationResult
from aatm.logs import get_logger
from aatm.omop.mapping import ClinicalNote
from aatm.omop.runner import (
    LEDGER_FILE,
    STAGES,
    OmopRun,
    RunLedger,
    _fingerprint,
    _iter_notes,
    extraction_fingerprint,
    mapping_fingerprint,
)
from aatm.omop.tables import OMOP_TABLES

if TYPE_CHECKING:
    from aatm.llm.usage import Prices
    from aatm.omop.builder import OmopExtractionPipeline, OmopTableBuilder
    from aatm.omop.runner import RunSummary

logger = get_logger(__name__)

SAMPLE_FILE = "sample.json"
NOTES_FILE = "notes.jsonl"
VERSION_FILE = "version.json"
DEFINITIONS_FILE = "definitions.yaml"
CODING_CACHE_FILE = "coding_cache.db"
DEFAULT_SAMPLE = 50


# --- the sample ---------------------------------------------------------------------


class PilotSample(BaseModel):
    """The notes every version of a pilot runs on.

    Attributes:
        source: The source the notes were picked from, as text.
        sample: Notes asked for.
        seed: The seed of the random pick.
        note_ids: The ids of the notes picked, sorted. Fewer than `sample` when the
            source had fewer notes.
        source_notes: Notes in the source when the sample was picked.
        created_at: When the sample was picked (UTC).
    """

    source: Optional[str] = None
    sample: int
    seed: int
    note_ids: list[int]
    source_notes: int
    created_at: datetime


def _describe_source(source: Any) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).as_posix()
    return type(source).__name__


def _pick(
    source: Any, columns: Optional[dict[str, str]], size: int, seed: int
) -> tuple[list[ClinicalNote], int]:
    """`size` notes picked at random from `source` in one pass, and the notes seen.

    Reservoir sampling: memory holds the notes picked and the ids seen, not the
    source. Notes whose id repeats an earlier one are left out.
    """
    rng = random.Random(seed)
    picked: list[ClinicalNote] = []
    seen: set[int] = set()
    for note in _iter_notes(source, columns, 1000):
        if note.note_id is None:
            raise ValueError("Every note needs a note_id to be sampled in a pilot")
        if note.note_id in seen:
            continue
        seen.add(note.note_id)
        if len(picked) < size:
            picked.append(note)
        else:
            index = rng.randrange(len(seen))
            if index < size:
                picked[index] = note
    picked.sort(key=lambda n: n.note_id)  # type: ignore[arg-type, return-value]
    return picked, len(seen)


# --- versions -----------------------------------------------------------------------


class PilotVersion(BaseModel):
    """One attempt of a pilot: its settings, what changed, and where its run is.

    Saved as ``version.json`` in the version's folder. `Pilot.run` also sets `path`,
    `outcome` and `summary`, which are not saved.

    Attributes:
        name: The folder name, e.g. ``"v003-3f9a2c1"``.
        number: 3 for ``v003``.
        created_at: When the version was made (UTC).
        action: ``run`` (extracted from scratch) or ``rebuild`` (built from the
            extraction results of `based_on`).
        based_on: For a rebuild, the version whose extraction results it reused.
        previous: The latest version when this one was made, which `changes`
            compare it to.
        extraction: The extraction fingerprint.
        mapping: The mapping fingerprint (the ``omop`` blocks and ``dayfirst``).
        coding: A hash of the mapping fingerprint and the coder's identity.
        model_id: The extraction model.
        extractor: The extractor's class.
        coder: The coder's class, and its model when it has one.
        coder_identity: A hash of what the coder's results depend on.
        changes: What changed from `previous`, one line each.
        definitions_diff: A unified diff of the definitions from `previous`.
        path: The version's folder.
        outcome: What `Pilot.run` did: ``new`` (extracted), ``rebuilt``,
            ``resumed`` (finished a version that had stopped partway) or ``reused``.
        summary: The summary of the last run or rebuild `Pilot.run` made, if any.
    """

    name: str
    number: int
    created_at: datetime
    action: Literal["run", "rebuild"]
    based_on: Optional[str] = None
    previous: Optional[str] = None
    extraction: str
    mapping: str
    coding: str
    model_id: Optional[str] = None
    extractor: str
    coder: Optional[str] = None
    coder_identity: Optional[str] = None
    changes: list[str] = []
    definitions_diff: str = ""
    path: Path = Field(Path("."), exclude=True)
    outcome: Optional[Literal["new", "rebuilt", "resumed", "reused"]] = Field(
        None, exclude=True
    )
    summary: Optional[Any] = Field(None, exclude=True)

    def output(self) -> OmopRun:
        """The version's run: its tables, issues, mappings and summaries."""
        return OmopRun(self.path)

    def definitions(self) -> str:
        """The definitions of this version, as YAML."""
        return (self.path / DEFINITIONS_FILE).read_text(encoding="utf-8")

    def __str__(self) -> str:
        stamp = self.created_at.strftime("%Y-%m-%d %H:%M")
        how = (
            f"rebuilt from {self.based_on}" if self.action == "rebuild" else "extracted"
        )
        lines = [f"{self.name}  {stamp}  {how}, model {self.model_id or 'unknown'}"]
        if self.previous is None:
            lines.append("  first version")
        else:
            lines += [f"  {c}" for c in self.changes] or ["  no change"]
        return "\n".join(lines)


def _definitions_yaml(pipeline: "OmopExtractionPipeline") -> str:
    entities = [
        e.model_dump(mode="json", by_alias=True, exclude_none=True)
        for e in pipeline.entity_definitions.entities
    ]
    return yaml.safe_dump(entities, sort_keys=False, allow_unicode=True, width=88)


def _base_coder(builder: "OmopTableBuilder") -> Any:
    from aatm.omop.cache import CachedCoder

    coder = builder.coder
    return coder.coder if isinstance(coder, CachedCoder) else coder


def _coder_identity(builder: "OmopTableBuilder") -> Optional[str]:
    from aatm.omop.cache import CachedCoder

    coder = builder.coder
    if coder is None:
        return None
    if isinstance(coder, CachedCoder):
        return _fingerprint(coder.identity["identity"])
    identity = getattr(coder, "cache_identity", None)
    if callable(identity):
        with contextlib.suppress(Exception):
            return _fingerprint(identity())
    return _fingerprint(
        {"class": f"{type(coder).__module__}.{type(coder).__qualname__}"}
    )


def _describe_coder(builder: "OmopTableBuilder") -> Optional[str]:
    coder = _base_coder(builder)
    if coder is None:
        return None
    text = type(coder).__name__
    for owner in (coder, getattr(coder, "mapper", None), getattr(coder, "agent", None)):
        model_id = getattr(owner, "model_id", None)
        if isinstance(model_id, str):
            return f"{text} ({model_id})"
    return text


def _changes(previous: PilotVersion, new: PilotVersion, diff: str) -> list[str]:
    changes = []
    if previous.model_id != new.model_id:
        changes.append(f"model: {previous.model_id} -> {new.model_id}")
    if previous.extractor != new.extractor:
        changes.append(f"extractor: {previous.extractor} -> {new.extractor}")
    if diff:
        added = sum(
            1 for line in diff.splitlines() if line[:1] == "+" and line[:3] != "+++"
        )
        removed = sum(
            1 for line in diff.splitlines() if line[:1] == "-" and line[:3] != "---"
        )
        lines = "line" if added == 1 else "lines"
        changes.append(f"definitions: {added} {lines} added, {removed} removed")
    elif previous.extraction != new.extraction:
        changes.append("extraction settings (instructions or attribute labeller)")
    if previous.coder != new.coder:
        changes.append(f"coder: {previous.coder} -> {new.coder}")
    elif previous.coder_identity != new.coder_identity:
        changes.append(f"coder settings ({new.coder})")
    if previous.mapping != new.mapping and not diff:
        changes.append("mapping settings (dayfirst)")
    return changes


# --- the workspace ------------------------------------------------------------------


class Pilot:
    """A workspace of versioned runs on a fixed sample of notes; see `aatm.omop.pilot`.

    Args:
        path: The workspace folder, made if missing.
        source: Where to pick the notes from, the first time: any source
            `OmopExtractionPipeline.run` reads. Not needed once the sample is stored.
        sample: Notes to pick, 50 by default.
        seed: Seed of the random pick, 0 by default.
        columns: Renames source columns to `ClinicalNote` fields, as for `run`.

    Raises:
        ValueError: If the workspace holds a sample of another size or seed than
            the one given.
    """

    def __init__(
        self,
        path: str | Path,
        source: Any = None,
        sample: Optional[int] = None,
        seed: Optional[int] = None,
        columns: Optional[dict[str, str]] = None,
    ):
        if sample is not None and sample < 1:
            raise ValueError("sample must be >= 1")
        self.path = Path(path)
        self.source = source
        self.size = DEFAULT_SAMPLE if sample is None else sample
        self.seed = 0 if seed is None else seed
        self.columns = columns
        stored = self._stored_sample()
        if stored is not None:
            for name, given, kept in (
                ("sample", sample, stored.sample),
                ("seed", seed, stored.seed),
            ):
                if given is not None and given != kept:
                    raise ValueError(
                        f"{self.path} holds a sample of {stored.sample} notes (seed "
                        f"{stored.seed}), not {name}={given}: use another folder, or "
                        "leave sample and seed out"
                    )

    # The sample.

    def _stored_sample(self) -> Optional[PilotSample]:
        path = self.path / SAMPLE_FILE
        if not path.exists():
            return None
        return PilotSample.model_validate_json(path.read_text(encoding="utf-8"))

    @property
    def sample(self) -> PilotSample:
        """The sample, picked from `source` and stored the first time it is needed."""
        stored = self._stored_sample()
        if stored is not None:
            return stored
        if self.source is None:
            raise ValueError(
                f"{self.path} has no sample yet: give the source of the notes, "
                "Pilot(path, source=...)"
            )
        notes, seen = _pick(self.source, self.columns, self.size, self.seed)
        if not notes:
            raise ValueError("The source has no notes")
        self.path.mkdir(parents=True, exist_ok=True)
        _write_atomic(
            self.path / NOTES_FILE,
            "".join(n.model_dump_json() + "\n" for n in notes),
        )
        sample = PilotSample(
            source=_describe_source(self.source),
            sample=self.size,
            seed=self.seed,
            note_ids=[n.note_id for n in notes],  # type: ignore[misc]
            source_notes=seen,
            created_at=datetime.now(timezone.utc),
        )
        _write_atomic(self.path / SAMPLE_FILE, sample.model_dump_json(indent=2))
        logger.info(
            "Picked %d of %d notes for the pilot in %s", len(notes), seen, self.path
        )
        return sample

    def notes(self) -> list[ClinicalNote]:
        """The notes of the sample."""
        self.sample  # noqa: B018 - picks and stores it the first time
        text = (self.path / NOTES_FILE).read_text(encoding="utf-8")
        return [ClinicalNote.model_validate_json(line) for line in text.splitlines()]

    # Versions.

    def versions(self) -> list[PilotVersion]:
        """Every version, oldest first."""
        versions = []
        if self.path.is_dir():
            for folder in self.path.iterdir():
                file = folder / VERSION_FILE
                if folder.is_dir() and file.exists():
                    version = PilotVersion.model_validate_json(
                        file.read_text(encoding="utf-8")
                    )
                    version.path = folder
                    versions.append(version)
        return sorted(versions, key=lambda v: v.number)

    def log(self) -> list[PilotVersion]:
        """Every version with what changed from the one before, oldest first.

        ``print(*pilot.log(), sep="\\n")`` prints them; each has its
        ``definitions_diff``.
        """
        return self.versions()

    def version(self, name: "str | int | PilotVersion") -> PilotVersion:
        """A version by name (``"v003"`` or ``"v003-3f9a2c1"``), number (``3``),
        or ``"latest"``.

        Raises:
            KeyError: If there is no such version.
        """
        if isinstance(name, PilotVersion):
            return name
        versions = self.versions()
        if name == "latest" and versions:
            return versions[-1]
        for version in versions:
            if name in (version.number, version.name, version.name.split("-")[0]):
                return version
        if isinstance(name, str) and name[:1] == "v" and name[1:].isdigit():
            for version in versions:
                if version.number == int(name[1:]):
                    return version
        known = ", ".join(v.name for v in versions) or "none"
        raise KeyError(f"No version {name!r} in {self.path} (versions: {known})")

    def _is_complete(self, version: PilotVersion) -> bool:
        if not (version.path / LEDGER_FILE).exists():
            return False
        ledger = RunLedger(version.path / LEDGER_FILE)
        try:
            built = ledger.counts().get("built", 0)
        finally:
            ledger.close()
        return built >= len(self.sample.note_ids)

    def _new_version(
        self,
        pipeline: "OmopExtractionPipeline",
        fingerprints: dict[str, Any],
        action: Literal["run", "rebuild"],
        based_on: Optional[PilotVersion],
    ) -> PilotVersion:
        versions = self.versions()
        previous = versions[-1] if versions else None
        number = previous.number + 1 if previous else 1
        digest = _fingerprint(
            {"extraction": fingerprints["extraction"], "coding": fingerprints["coding"]}
        )
        name = f"v{number:03d}-{digest[:7]}"
        folder = self.path / name
        folder.mkdir(parents=True)
        definitions = _definitions_yaml(pipeline)
        version = PilotVersion(
            name=name,
            number=number,
            created_at=datetime.now(timezone.utc),
            action=action,
            based_on=based_on.name if based_on else None,
            previous=previous.name if previous else None,
            **fingerprints,
        )
        version.path = folder
        if previous is not None:
            old = (previous.path / DEFINITIONS_FILE).read_text(encoding="utf-8")
            version.definitions_diff = "".join(
                difflib.unified_diff(
                    old.splitlines(keepends=True),
                    definitions.splitlines(keepends=True),
                    fromfile=f"{previous.name}/{DEFINITIONS_FILE}",
                    tofile=f"{name}/{DEFINITIONS_FILE}",
                )
            )
            version.changes = _changes(previous, version, version.definitions_diff)
        _write_atomic(folder / DEFINITIONS_FILE, definitions)
        _write_atomic(folder / VERSION_FILE, version.model_dump_json(indent=2))
        return version

    async def run(
        self,
        pipeline: "OmopExtractionPipeline",
        *,
        chunk_size: int = 100,
        max_attempts: int = 3,
        prices: Optional["Prices"] = None,
        log_every: float = 60.0,
        batch: bool = False,
        poll_every: float = 60.0,
    ) -> PilotVersion:
        """Run `pipeline` on the sample, in a new version or one that matches.

        A new extraction fingerprint makes a new version, extracted from scratch.
        The same extraction with other mappings or another coder makes a new version
        rebuilt from the stored results of the latest version with that extraction,
        with no extraction call. With nothing changed, the matching version is reused
        (and finished, if its run stopped partway).

        Args:
            pipeline: The pipeline to try.
            chunk_size: As for `OmopExtractionPipeline.run`.
            max_attempts: As for `OmopExtractionPipeline.run`.
            prices: Model prices, for the cost in the summaries and `compare`.
            log_every: Seconds between progress logs.
            batch: Extract through the provider's batch endpoint.
            poll_every: Seconds between polls of batch jobs.

        Returns:
            The version, with its `path`, what was done (`outcome`) and the run's
            `summary`. ``version.output()`` reads its tables.
        """
        notes = self.notes()
        with self._shared_cache(pipeline):
            fingerprints = {
                "extraction": extraction_fingerprint(pipeline),
                "mapping": mapping_fingerprint(pipeline.builder),
                "coding": "",
                "model_id": getattr(pipeline.extractor, "model_id", None),
                "extractor": type(pipeline.extractor).__qualname__,
                "coder": _describe_coder(pipeline.builder),
                "coder_identity": _coder_identity(pipeline.builder),
            }
            fingerprints["coding"] = _fingerprint(
                {
                    "mapping": fingerprints["mapping"],
                    "coder": fingerprints["coder_identity"],
                }
            )
            versions = self.versions()
            same = [
                v
                for v in versions
                if (v.extraction, v.coding)
                == (fingerprints["extraction"], fingerprints["coding"])
            ]
            run_kwargs: dict[str, Any] = dict(
                chunk_size=chunk_size,
                max_attempts=max_attempts,
                prices=prices,
                log_every=log_every,
                batch=batch,
                poll_every=poll_every,
            )
            if same:
                version = same[-1]
                if self._is_complete(version):
                    version.outcome = "reused"
                    logger.info("Nothing changed: reusing %s", version.path)
                    return version
                logger.info("Finishing %s, which stopped partway", version.path)
                version.summary = await pipeline.run(notes, version.path, **run_kwargs)
                version.outcome = "resumed"
                return version

            extracted = [
                v for v in versions if v.extraction == fingerprints["extraction"]
            ]
            if extracted:
                base = extracted[-1]
                version = self._new_version(pipeline, fingerprints, "rebuild", base)
                logger.info(
                    "Same extraction as %s, other mappings: rebuilding into %s, "
                    "with no extraction call",
                    base.name,
                    version.path,
                )
                _copy_ledger(base.path / LEDGER_FILE, version.path / LEDGER_FILE)
                version.summary = await pipeline.rebuild(
                    version.path,
                    chunk_size=max(chunk_size, 1000),
                    prices=prices,
                    log_every=log_every,
                )
                version.outcome = "rebuilt"
                return version

            version = self._new_version(pipeline, fingerprints, "run", None)
            logger.info("New extraction settings: running into %s", version.path)
            version.summary = await pipeline.run(notes, version.path, **run_kwargs)
            version.outcome = "new"
            return version

    @contextlib.contextmanager
    def _shared_cache(self, pipeline: "OmopExtractionPipeline") -> Any:
        """Give the pipeline's coder the workspace's cache while it runs, if it has
        none and can be cached."""
        from aatm.omop.cache import CachedCoder
        from aatm.omop.coding import CacheableCoder

        builder = pipeline.builder
        coder = builder.coder
        if coder is None or isinstance(coder, CachedCoder):
            yield
            return
        if not isinstance(coder, CacheableCoder):
            yield
            return
        self.path.mkdir(parents=True, exist_ok=True)
        builder.coder = CachedCoder(coder, self.path / CODING_CACHE_FILE)
        try:
            yield
        finally:
            builder.coder = coder

    # Comparing.

    def compare(
        self,
        a: "str | int | PilotVersion | None" = None,
        b: "str | int | PilotVersion | None" = None,
    ) -> "PilotComparison":
        """Compare two versions: rows, issues, mentions, codes, tokens and cost.

        Args:
            a: The earlier version (a name, a number or ``"latest"``). By default the
                one before `b`.
            b: The later version. By default the latest.

        Returns:
            A `PilotComparison`: DataFrames of the differences; ``print`` it for a
            report.
        """
        versions = self.versions()
        if len(versions) < 2 and (a is None or b is None):
            raise ValueError(f"{self.path} needs two versions to compare")
        new = self.version(b) if b is not None else versions[-1]
        if a is not None:
            old = self.version(a)
        else:
            earlier = [v for v in versions if v.number < new.number]
            if not earlier:
                raise ValueError(
                    f"{new.name} is the first version: name one to compare"
                )
            old = earlier[-1]
        if old.name == new.name:
            raise ValueError(
                f"{new.name} is compared with itself: name two different versions"
            )
        return compare_versions(old, new, versions)


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _copy_ledger(source: Path, target: Path) -> None:
    """Copy a run's ledger, its write-ahead log included, with SQLite's backup."""
    src = sqlite3.connect(source)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


# --- the comparison -----------------------------------------------------------------

MENTION_COLUMNS = [
    "note_id",
    "entity",
    "change",
    "line_number",
    "line",
    "start_a",
    "end_a",
    "text_a",
    "attributes_a",
    "start_b",
    "end_b",
    "text_b",
    "attributes_b",
]
MAPPING_COLUMNS = [
    "table",
    "column",
    "text",
    "change",
    "concept_id_a",
    "concept_name_a",
    "stored_a",
    "concept_id_b",
    "concept_name_b",
    "stored_b",
]


class PilotComparison:
    """The differences between two versions of a pilot; made by `Pilot.compare`.

    Attributes:
        a: The earlier version.
        b: The later version.
        notes: Notes of the sample extracted in `a`, in `b`, and in both. Mentions
            are compared on the notes extracted in both.
        rows: Rows per table: columns `a`, `b` and ``change``.
        issues: Issues per reason, likewise.
        mentions: One row per mention added, removed or changed (``change``), with
            the note id, the entity key, the line of the note, and the span, text and
            attributes in each version. A mention whose span overlaps one of the
            other version, for the same entity, is ``changed``.
        mappings: Texts whose code changed (``changed``), or that only one version
            coded (``added``, ``removed``): table, column, text, and the concept and
            stored value in each version.
        usage: LLM calls, tokens and cost, in all and per note, of each version. A
            rebuilt version counts the extraction of the version it was built from.
    """

    def __init__(
        self,
        a: PilotVersion,
        b: PilotVersion,
        notes: dict[str, int],
        rows: pd.DataFrame,
        issues: pd.DataFrame,
        mentions: pd.DataFrame,
        mappings: pd.DataFrame,
        usage: pd.DataFrame,
    ):
        self.a = a
        self.b = b
        self.notes = notes
        self.rows = rows
        self.issues = issues
        self.mentions = mentions
        self.mappings = mappings
        self.usage = usage

    def __repr__(self) -> str:
        return f"PilotComparison({self.a.name!r}, {self.b.name!r})"

    def __str__(self) -> str:
        a, b = self.a.name, self.b.name
        lines = [f"Comparison of {a} and {b}"]
        if self.b.previous == a:
            lines += [f"  {c}" for c in self.b.changes] or ["  no change recorded"]
        else:
            lines.append(f"  {b} was not made right after {a}: see pilot.log()")
        n = self.notes
        lines.append(
            f"notes extracted: {n['a']} in {a}, {n['b']} in {b}, {n['both']} in both"
        )
        lines += ["", "rows per table:", _indent(_frame_text(self.rows, "no rows"))]
        lines += ["", "issues per reason:", _indent(_frame_text(self.issues, "none"))]
        counts: Any = self.mentions["change"].value_counts()
        lines += [
            "",
            "mentions: "
            + ", ".join(
                f"{counts.get(c, 0)} {c}" for c in ("added", "removed", "changed")
            ),
        ]
        for note_id, group in self.mentions.groupby("note_id", sort=True):
            lines.append(f"  note {note_id}:")
            for row in group.itertuples(index=False):
                lines.append(f"    {_describe_mention(row)}")
        lines += ["", f"codes changed: {len(self.mappings)}"]
        for row in self.mappings.itertuples(index=False):
            old = _concept(row.concept_id_a, row.concept_name_a, row.stored_a)
            new = _concept(row.concept_id_b, row.concept_name_b, row.stored_b)
            lines.append(
                f"  {row.table}.{row.column} {row.text!r}: {old} -> {new}"
                + (f" ({row.change})" if row.change != "changed" else "")
            )
        lines += ["", "tokens and cost:", _indent(_frame_text(self.usage, "none"))]
        return "\n".join(lines)


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


def _frame_text(frame: pd.DataFrame, empty: str) -> str:
    return empty if frame.empty else frame.to_string()


def _concept(concept_id: Any, name: Any, stored: Any) -> str:
    if concept_id is None or pd.isna(concept_id):
        if stored is None or pd.isna(stored):
            return "none"
        return f"no concept (stored {int(stored)})"
    text = f"{int(concept_id)} {name}" if isinstance(name, str) else str(concept_id)
    if stored is not None and not pd.isna(stored) and int(stored) != int(concept_id):
        text += f" (stored {int(stored)})"
    return text


def _describe_mention(row: Any) -> str:
    if row.change == "added":
        what = f"+ {row.entity} {row.text_b!r} [{row.start_b}:{row.end_b}]"
        attributes = row.attributes_b
    elif row.change == "removed":
        what = f"- {row.entity} {row.text_a!r} [{row.start_a}:{row.end_a}]"
        attributes = row.attributes_a
    else:
        what = (
            f"~ {row.entity} {row.text_a!r} [{row.start_a}:{row.end_a}] -> "
            f"{row.text_b!r} [{row.start_b}:{row.end_b}]"
        )
        attributes = (
            f"{row.attributes_a or '-'} -> {row.attributes_b or '-'}"
            if row.attributes_a != row.attributes_b
            else row.attributes_b
        )
    if attributes:
        what += f" {{{attributes}}}"
    return f"{what}  (line {row.line_number}: {row.line!r})"


def _results(path: Path) -> dict[int, tuple[ClinicalNote, EntityValidationResult]]:
    """The notes of a version with a stored extraction result."""
    file = path / LEDGER_FILE
    if not file.exists():
        return {}
    con = sqlite3.connect(file)
    try:
        rows = con.execute(
            "SELECT note_id, note, result FROM notes WHERE result IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    return {
        note_id: (
            ClinicalNote.model_validate_json(note),
            EntityValidationResult.model_validate_json(result),
        )
        for note_id, note, result in rows
    }


def _line(text: str, start: int, end: int) -> tuple[int, str]:
    """The number (from 1) and text of the line a span starts on."""
    begin = text.rfind("\n", 0, start) + 1
    stop = text.find("\n", start)
    line = text[begin : len(text) if stop == -1 else stop]
    return text.count("\n", 0, begin) + 1, line.strip()


def _mentions(result: EntityValidationResult) -> list[dict[str, Any]]:
    return [
        {
            "entity": e.key,
            "start": e.start,
            "end": e.end,
            "text": e.text,
            "attributes": "; ".join(
                f"{at.key}={at.value}"
                for at in sorted(e.attributes, key=lambda at: at.key)
                if at.value is not None
            ),
        }
        for e in result.validated_entities
    ]


def _diff_mentions(
    note: ClinicalNote, old: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def row(change: str, a: Optional[dict], b: Optional[dict]) -> None:
        anchor = b or a
        assert anchor is not None
        number, line = _line(note.text, anchor["start"], anchor["end"])
        rows.append(
            {
                "note_id": note.note_id,
                "entity": anchor["entity"],
                "change": change,
                "line_number": number,
                "line": line,
                **{f"{k}_a": (a or {}).get(k) for k in ("start", "end", "text")},
                "attributes_a": (a or {}).get("attributes"),
                **{f"{k}_b": (b or {}).get(k) for k in ("start", "end", "text")},
                "attributes_b": (b or {}).get("attributes"),
            }
        )

    left = list(old)
    right = list(new)
    # Same entity and span: changed if its text or attributes differ.
    for a in list(left):
        b = next(
            (
                m
                for m in right
                if (m["entity"], m["start"], m["end"])
                == (a["entity"], a["start"], a["end"])
            ),
            None,
        )
        if b is None:
            continue
        left.remove(a)
        right.remove(b)
        if (a["text"], a["attributes"]) != (b["text"], b["attributes"]):
            row("changed", a, b)
    # Overlapping spans of the same entity: the span changed.
    for a in list(left):
        overlapping = [
            m
            for m in right
            if m["entity"] == a["entity"]
            and m["start"] < a["end"]
            and a["start"] < m["end"]
        ]
        if not overlapping:
            continue
        b = max(
            overlapping,
            key=lambda m: min(m["end"], a["end"]) - max(m["start"], a["start"]),
        )
        left.remove(a)
        right.remove(b)
        row("changed", a, b)
    for a in left:
        row("removed", a, None)
    for b in right:
        row("added", None, b)
    return rows


def _side_by_side(a: pd.Series, b: pd.Series, a_name: str, b_name: str) -> pd.DataFrame:
    frame = pd.concat([a.rename(a_name), b.rename(b_name)], axis=1).fillna(0)
    frame = frame.astype(int)
    frame["change"] = frame[b_name] - frame[a_name]
    return frame.sort_index()


def _row_counts(run: Optional[OmopRun]) -> pd.Series:
    if run is None:
        return pd.Series(dtype=int)
    counts = {}
    for table in OMOP_TABLES:
        if run._parts(table):
            counts[table] = run.dataset(table).count_rows()
    return pd.Series(counts, dtype=int)


def _issue_counts(run: Optional[OmopRun]) -> pd.Series:
    if run is None:
        return pd.Series(dtype=int)
    issues = run.issues()
    if issues.empty:
        return pd.Series(dtype=int)
    return issues.groupby("reason").size()


def _codes(run: Optional[OmopRun]) -> pd.DataFrame:
    columns = ["table", "column", "text", "concept_id", "concept_name", "stored"]
    if run is None:
        return pd.DataFrame(columns=columns)
    mappings = run.mappings()
    if mappings.empty:
        return pd.DataFrame(columns=columns)
    return mappings[columns].drop_duplicates(["table", "column", "text"])


def _diff_codes(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    keys = ["table", "column", "text"]
    merged = old.merge(new, on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)

    def same(x: Any, y: Any) -> bool:
        if pd.isna(x) and pd.isna(y):
            return True
        return not (pd.isna(x) or pd.isna(y)) and int(x) == int(y)

    change = []
    for side, id_a, id_b, stored_a, stored_b in zip(
        merged["_merge"],
        merged["concept_id_a"],
        merged["concept_id_b"],
        merged["stored_a"],
        merged["stored_b"],
    ):
        if side == "left_only":
            change.append("removed")
        elif side == "right_only":
            change.append("added")
        elif same(id_a, id_b) and same(stored_a, stored_b):
            change.append("")
        else:
            change.append("changed")
    merged["change"] = change
    merged = merged[merged["change"] != ""]
    order = {"changed": 0, "added": 1, "removed": 2}
    merged = merged.sort_values(
        by=["change", *keys], key=lambda s: s.map(order) if s.name == "change" else s
    )
    return merged[MAPPING_COLUMNS].reset_index(drop=True)


def _summaries(version: PilotVersion) -> list["RunSummary"]:
    try:
        return OmopRun(version.path).summaries()
    except FileNotFoundError:
        return []


def _usage(version: PilotVersion, versions: dict[str, PilotVersion]) -> dict[str, Any]:
    """Calls, tokens and cost of a version; a rebuild counts its base's extraction."""
    stages: dict[str, list[Any]] = {name: [] for name in STAGES}
    for summary in _summaries(version):
        for name, stage in summary.stages.items():
            stages.setdefault(name, []).append(stage)
    base = versions.get(version.based_on or "")
    if version.action == "rebuild" and base is not None:
        stages["extraction"] = [
            s.stages["extraction"] for s in _summaries(base) if "extraction" in s.stages
        ]
    calls = input_tokens = output_tokens = 0
    cost: Optional[float] = None
    for stage_list in stages.values():
        for stage in stage_list:
            calls += stage.usage.calls
            input_tokens += stage.usage.input_tokens
            output_tokens += stage.usage.output_tokens
            if stage.cost is not None:
                cost = (cost or 0.0) + stage.cost
    return {
        "llm_calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }


def compare_versions(
    a: PilotVersion,
    b: PilotVersion,
    versions: Optional[Sequence[PilotVersion]] = None,
) -> PilotComparison:
    """Compare two versions of a pilot; see `Pilot.compare`.

    Args:
        a: The earlier version.
        b: The later version.
        versions: Every version of the pilot, to find the extraction a rebuilt
            version reused. By default, the versions in the folders next to `b`.
    """
    if versions is None:
        versions = Pilot(b.path.parent).versions()
    by_name = {v.name: v for v in versions}
    runs: dict[str, Optional[OmopRun]] = {}
    for version in (a, b):
        try:
            runs[version.name] = OmopRun(version.path)
        except FileNotFoundError:
            runs[version.name] = None
    run_a, run_b = runs[a.name], runs[b.name]

    rows = _side_by_side(_row_counts(run_a), _row_counts(run_b), a.name, b.name)
    issues = _side_by_side(_issue_counts(run_a), _issue_counts(run_b), a.name, b.name)

    results_a, results_b = _results(a.path), _results(b.path)
    both = sorted(set(results_a) & set(results_b))
    mention_rows: list[dict[str, Any]] = []
    for note_id in both:
        note, old = results_a[note_id]
        _, new = results_b[note_id]
        mention_rows += _diff_mentions(note, _mentions(old), _mentions(new))
    mentions = pd.DataFrame(mention_rows, columns=MENTION_COLUMNS)

    mappings = _diff_codes(_codes(run_a), _codes(run_b))

    usage_rows = {}
    for version, results in ((a, results_a), (b, results_b)):
        values = _usage(version, by_name)
        notes = len(results)
        for key in ("input_tokens", "output_tokens", "cost"):
            value = values[key]
            values[f"{key}_per_note"] = (
                value / notes if notes and value is not None else None
            )
        usage_rows[version.name] = {"notes": notes, **values}
    usage = pd.DataFrame(usage_rows, dtype=object)
    usage.index.name = "metric"

    return PilotComparison(
        a,
        b,
        notes={"a": len(results_a), "b": len(results_b), "both": len(both)},
        rows=rows,
        issues=issues,
        mentions=mentions,
        mappings=mappings,
        usage=usage,
    )
