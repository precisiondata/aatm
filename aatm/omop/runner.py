"""Process datasets larger than memory, in chunks, and resume runs that stop.

`OmopExtractionPipeline.run` reads notes in chunks, extracts and builds each chunk,
and writes the rows to Parquet files as it goes, so memory use does not grow with the
dataset:

```python
from aatm.omop import OmopExtractionPipeline, OmopRun

pipeline = OmopExtractionPipeline(
    "ecog_performance_status", model_id="openai/gpt-5-mini", max_concurrency=16
)
summary = await pipeline.run(
    "notes.parquet", "output/ecog", chunk_size=200, columns={"note_text": "text"}
)
print(summary)

run = OmopRun("output/ecog")
run.table("measurement")  # a DataFrame; run.dataset("measurement") reads lazily
run.issues()
run.failures()            # notes that could not be extracted or built, and why
```

## Resuming

Call `run` again with the same output folder and source: notes already built are
skipped, and notes already extracted are built from their stored result, without
another model call. A note that fails (an API error, a result that cannot be built) is
recorded with its error and the run goes on; a later run tries it again, up to
``max_attempts`` times in all. `note_id` is the key that tells notes apart, so every
note needs a unique one.

While a chunk is built (coded and written, in a thread), the next one is extracted;
at most two chunks are in memory.

## Rebuilding

Extraction, the costly step, is stored apart from the tables. After changing the
``omop`` blocks, `OmopExtractionPipeline.rebuild` builds every table again from the
stored results without any extraction call. A run refuses to add rows built with other
mappings to a folder: rebuild first. It also refuses a folder made with other entity
definitions or another extractor, whose stored results would not match.

## Output folder

```text
output/ecog/
  ledger.db                     notes, their extraction results, status and errors
  measurement/part-000001.parquet
  provenance/part-000001.parquet  note id, entity id and span of each row
  issues/part-000001.parquet
  mappings/part-000001.parquet
```

Each chunk writes one part file per dataset. A part only counts once the ledger
records it: parts left by a run that stopped while writing are deleted when the next
run starts, so no row is written twice.

## Row ids

Primary keys are derived from the note: ``note_id * 65536 + n``, where ``n`` is the
position of the mention in the note (see `aatm.omop.builder.note_row_id`). They are
the same whichever chunk, run or shard builds the note, and after a rebuild, and they
are unique across tables. They need a 64-bit (``BIGINT``) column.
"""

from datetime import date, datetime
from itertools import islice
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Union,
    get_args,
    get_origin,
)

import asyncio
import contextlib
import hashlib
import json
import os
import sqlite3
import time
import types

import pandas as pd
from pydantic import BaseModel

from aatm.data_models.extractors import EntityValidationResult
from aatm.logs import get_logger
from aatm.omop.builder import (
    MAX_NOTE_ID,
    ConceptMapping,
    OmopIssue,
    OmopTables,
    RowProvenance,
)
from aatm.omop.cache import CacheStats
from aatm.omop.mapping import ClinicalNote
from aatm.omop.tables import OMOP_TABLES

if TYPE_CHECKING:
    import pyarrow as pa

    from aatm.omop.builder import OmopExtractionPipeline

logger = get_logger(__name__)

LEDGER_FILE = "ledger.db"
PROVENANCE, ISSUES, MAPPINGS = "provenance", "issues", "mappings"
_NOTE_FIELDS = set(ClinicalNote.model_fields)
# SQLite limits the number of parameters of a statement.
_SQL_BATCH = 500


# --- sources ------------------------------------------------------------------


class _FrameSource:
    """A source that yields DataFrames, e.g. a database query read in chunks."""

    def __init__(self, frames: Iterable[pd.DataFrame]):
        self.frames = frames


def notes_from_sql(
    query: str, con: Any, chunk_size: int = 10_000, params: Any = None
) -> _FrameSource:
    """Read notes from a database query, in chunks, for `OmopExtractionPipeline.run`.

    Args:
        query: A query returning a ``text`` column and any other `ClinicalNote`
            fields (``note_id`` is required); rename columns with ``AS`` or with the
            ``columns`` argument of `run`.
        con: A `sqlite3` connection, or a SQLAlchemy engine or connection for other
            databases (see `pandas.read_sql`).
        chunk_size: Rows fetched at a time.
        params: Query parameters, passed to `pandas.read_sql`.
    """
    return _FrameSource(pd.read_sql(query, con, chunksize=chunk_size, params=params))


def _plain(value: Any) -> Any:
    """A pandas or numpy value as a plain Python value, and missing values as None."""
    if value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()  # numpy scalars
    return value


def _notes_from_frame(
    frame: pd.DataFrame, columns: Optional[dict[str, str]]
) -> Iterator[ClinicalNote]:
    frame = frame.rename(columns=columns or {})
    if "text" not in frame.columns:
        raise ValueError(
            f"The source has no 'text' column (columns: {list(frame.columns)}). "
            "Rename the note text column with columns={'<name>': 'text'}"
        )
    keep = [c for c in frame.columns if c in _NOTE_FIELDS]
    for row in frame[keep].to_dict("records"):
        values = {k: _plain(v) for k, v in row.items()}
        if isinstance(values.get("note_date"), datetime):
            values["note_date"] = values["note_date"].date()
        yield ClinicalNote.model_validate(values)


def _iter_notes(
    source: Any, columns: Optional[dict[str, str]], read_size: int
) -> Iterator[ClinicalNote]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"No such file or folder: {path}")
        if path.is_dir() or path.suffix == ".parquet":
            import pyarrow.dataset as ds

            dataset = ds.dataset(path, format="parquet")
            renamed = columns or {}
            wanted = [
                c for c in dataset.schema.names if renamed.get(c, c) in _NOTE_FIELDS
            ]
            for batch in dataset.to_batches(columns=wanted, batch_size=read_size):
                yield from _notes_from_frame(batch.to_pandas(), columns)
        elif path.suffix in (".csv", ".tsv", ".gz"):
            sep = "\t" if ".tsv" in path.suffixes else ","
            for frame in pd.read_csv(path, sep=sep, chunksize=read_size):
                yield from _notes_from_frame(frame, columns)
        else:
            raise ValueError(
                f"Cannot read {path}: use a .csv, .tsv or .parquet file, or a folder "
                "of Parquet files"
            )
    elif isinstance(source, pd.DataFrame):
        for start in range(0, len(source), read_size):
            yield from _notes_from_frame(
                source.iloc[start : start + read_size], columns
            )
    elif isinstance(source, _FrameSource):
        for frame in source.frames:
            yield from _notes_from_frame(frame, columns)
    else:
        rename = columns or {}
        for item in source:
            if isinstance(item, ClinicalNote):
                yield item
            elif isinstance(item, dict):
                renamed = {rename[k] if k in rename else k: v for k, v in item.items()}
                yield ClinicalNote.model_validate(renamed)
            else:
                raise TypeError(
                    "Notes must be ClinicalNote objects or dicts, not "
                    f"{type(item).__name__}: plain strings have no note_id to resume "
                    "from"
                )


def read_chunks(
    source: Any, chunk_size: int, columns: Optional[dict[str, str]] = None
) -> Iterator[list[ClinicalNote]]:
    """Read notes from a source in lists of `chunk_size` (the last may be shorter)."""
    notes = _iter_notes(source, columns, read_size=max(chunk_size, 1000))
    while chunk := list(islice(notes, chunk_size)):
        yield chunk


# --- Parquet output -------------------------------------------------------------


def _arrow_type(annotation: Any) -> "pa.DataType":
    import pyarrow as pa

    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        (inner,) = [a for a in get_args(annotation) if a is not type(None)][:1]
        return _arrow_type(inner)
    if origin is Annotated:
        return _arrow_type(get_args(annotation)[0])
    if origin is Literal:
        return pa.string()
    if annotation is bool:
        return pa.bool_()
    if annotation is int:
        return pa.int64()
    if annotation is float:
        return pa.float64()
    if annotation is str:
        return pa.string()
    if annotation is datetime:
        return pa.timestamp("us")
    if annotation is date:
        return pa.date32()
    return pa.string()  # lists, dicts and Any are stored as JSON


def _schema(
    model: type[BaseModel],
    exclude: Sequence[str] = (),
    extra: Sequence[tuple[str, "pa.DataType"]] = (),
) -> "pa.Schema":
    import pyarrow as pa

    fields = list(extra) + [
        (name, _arrow_type(info.annotation))
        for name, info in model.model_fields.items()
        if name not in exclude
    ]
    return pa.schema(fields)


def _write(
    folder: Path, part: int, schema: "pa.Schema", rows: list[dict[str, Any]]
) -> None:
    """Write one part file, atomically: readers skip the hidden temporary file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        return
    folder.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for name in schema.names:
            if pa.types.is_string(schema.field(name).type) and isinstance(
                row.get(name), (list, dict)
            ):
                row[name] = json.dumps(row[name], ensure_ascii=False, default=str)
    table = pa.Table.from_pylist(rows, schema=schema)
    final = folder / _part_name(part)
    temporary = folder / f".{final.name}.tmp"
    pq.write_table(table, temporary)
    os.replace(temporary, final)


def _part_name(part: int) -> str:
    return f"part-{part:06d}.parquet"


def _write_part(output_dir: Path, part: int, tables: OmopTables) -> dict[str, int]:
    """Write the rows, provenance, issues and mappings of a chunk; return row counts."""
    import pyarrow as pa

    counts: dict[str, int] = {}
    provenance: list[dict[str, Any]] = []
    for table, rows in tables.rows.items():
        model = OMOP_TABLES[table]
        _write(output_dir / table, part, _schema(model), [r.model_dump() for r in rows])
        counts[table] = len(rows)
        for row, prov in zip(rows, tables.provenance[table]):
            provenance.append(
                {
                    "table": table,
                    "row_id": getattr(row, model.primary_key),
                    **prov.model_dump(exclude={"note_index"}),
                }
            )
    _write(
        output_dir / PROVENANCE,
        part,
        _schema(
            RowProvenance,
            exclude=["note_index"],
            extra=[("table", pa.string()), ("row_id", pa.int64())],
        ),
        provenance,
    )
    _write(
        output_dir / ISSUES,
        part,
        _schema(OmopIssue, exclude=["note_index"]),
        [i.model_dump(exclude={"note_index"}) for i in tables.issues],
    )
    _write(
        output_dir / MAPPINGS,
        part,
        _schema(ConceptMapping),
        [m.model_dump() for m in tables.mappings],
    )
    return counts


# --- the ledger -------------------------------------------------------------------

_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    note_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,         -- extracted, built or failed
    stage TEXT,                   -- extract or build, for failed notes
    note TEXT NOT NULL,           -- the ClinicalNote, as JSON
    result TEXT,                  -- the EntityValidationResult, as JSON
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    part INTEGER,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_status ON notes (status, note_id);
CREATE TABLE IF NOT EXISTS parts (
    part INTEGER PRIMARY KEY AUTOINCREMENT,
    committed INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class _NoteState(BaseModel):
    status: Literal["extracted", "built", "failed"]
    stage: Optional[Literal["extract", "build"]]
    attempts: int


class RunLedger:
    """The notes of a run: their status, extraction result and error, in SQLite."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.con = sqlite3.connect(self.path)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.executescript(_LEDGER_SCHEMA)

    def close(self) -> None:
        self.con.close()

    def get_meta(self, key: str) -> Optional[str]:
        row = self.con.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.con:
            self.con.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
            )

    def _select(self, columns: str, ids: Sequence[int]) -> Iterator[tuple]:
        for start in range(0, len(ids), _SQL_BATCH):
            batch = list(ids[start : start + _SQL_BATCH])
            marks = ",".join("?" * len(batch))
            yield from self.con.execute(
                f"SELECT {columns} FROM notes WHERE note_id IN ({marks})", batch
            )

    def states(self, ids: Sequence[int]) -> dict[int, _NoteState]:
        return {
            note_id: _NoteState(status=status, stage=stage, attempts=attempts)
            for note_id, status, stage, attempts in self._select(
                "note_id, status, stage, attempts", ids
            )
        }

    def load(
        self, ids: Sequence[int]
    ) -> list[tuple[ClinicalNote, EntityValidationResult]]:
        """The stored notes and extraction results, in the order of `ids`."""
        found = {
            note_id: (
                ClinicalNote.model_validate_json(note),
                EntityValidationResult.model_validate_json(result),
            )
            for note_id, note, result in self._select("note_id, note, result", ids)
        }
        return [found[i] for i in ids]

    def save_extracted(
        self, items: Sequence[tuple[ClinicalNote, EntityValidationResult]]
    ) -> None:
        now = time.time()
        with self.con:
            self.con.executemany(
                "INSERT INTO notes (note_id, status, note, result, attempts, updated_at)"
                " VALUES (?, 'extracted', ?, ?, 1, ?) ON CONFLICT (note_id) DO UPDATE"
                " SET status = 'extracted', stage = NULL, note = excluded.note,"
                " result = excluded.result, error = NULL, attempts = attempts + 1,"
                " updated_at = excluded.updated_at",
                [
                    (n.note_id, n.model_dump_json(), r.model_dump_json(), now)
                    for n, r in items
                ],
            )

    def save_failed(
        self,
        items: Sequence[tuple[ClinicalNote, str]],
        stage: Literal["extract", "build"],
    ) -> None:
        """Record failed notes. A build failure keeps the stored extraction result."""
        now = time.time()
        with self.con:
            if stage == "extract":
                self.con.executemany(
                    "INSERT INTO notes (note_id, status, stage, note, error, attempts,"
                    " updated_at) VALUES (?, 'failed', 'extract', ?, ?, 1, ?)"
                    " ON CONFLICT (note_id) DO UPDATE SET status = 'failed',"
                    " stage = 'extract', note = excluded.note, error = excluded.error,"
                    " attempts = attempts + 1, updated_at = excluded.updated_at",
                    [(n.note_id, n.model_dump_json(), e, now) for n, e in items],
                )
            else:
                self.con.executemany(
                    "UPDATE notes SET status = 'failed', stage = 'build', error = ?,"
                    " attempts = attempts + 1, updated_at = ? WHERE note_id = ?",
                    [(e, now, n.note_id) for n, e in items],
                )

    def new_part(self) -> int:
        with self.con:
            cursor = self.con.execute(
                "INSERT INTO parts (created_at) VALUES (?)", (time.time(),)
            )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def commit_part(self, part: int, note_ids: Sequence[int]) -> None:
        """Mark the notes built in `part`, and the part complete, in one transaction."""
        now = time.time()
        with self.con:
            self.con.executemany(
                "UPDATE notes SET status = 'built', stage = NULL, error = NULL,"
                " part = ?, updated_at = ? WHERE note_id = ?",
                [(part, now, i) for i in note_ids],
            )
            self.con.execute("UPDATE parts SET committed = 1 WHERE part = ?", (part,))

    def uncommitted_parts(self) -> list[int]:
        return [
            p for (p,) in self.con.execute("SELECT part FROM parts WHERE committed = 0")
        ]

    def drop_part(self, part: int) -> None:
        with self.con:
            self.con.execute("DELETE FROM parts WHERE part = ?", (part,))

    def has_built(self) -> bool:
        return (
            self.con.execute(
                "SELECT 1 FROM notes WHERE status = 'built' LIMIT 1"
            ).fetchone()
            is not None
        )

    def reset_for_rebuild(self) -> None:
        """Mark every part for deletion and every built note to be built again."""
        with self.con:
            self.con.execute("UPDATE parts SET committed = 0")
            self.con.execute(
                "UPDATE notes SET status = 'extracted', stage = NULL, error = NULL,"
                " part = NULL WHERE status = 'built'"
                " OR (status = 'failed' AND stage = 'build')"
            )

    def extracted_ids(self, after: int, limit: int) -> list[int]:
        return [
            i
            for (i,) in self.con.execute(
                "SELECT note_id FROM notes WHERE status = 'extracted' AND note_id > ?"
                " ORDER BY note_id LIMIT ?",
                (after, limit),
            )
        ]

    def counts(self) -> dict[str, int]:
        """Notes per status: ``extracted`` (not built yet), ``built``, ``failed``."""
        return dict(
            self.con.execute("SELECT status, COUNT(*) FROM notes GROUP BY status")
        )

    def failures(self) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT note_id, stage, error, attempts, updated_at FROM notes"
            " WHERE status = 'failed' ORDER BY note_id",
            self.con,
        )


# --- the run ------------------------------------------------------------------------


class RunSummary(BaseModel):
    """What one call of `run` or `rebuild` did.

    Attributes:
        output_dir: The output folder.
        notes: Notes read from the source (or from the ledger, for a rebuild).
        built: Notes whose rows were written by this call.
        extracted: Notes extracted by this call: one model call each.
        reused: Notes built from a stored extraction result, with no model call.
        skipped: Notes already built by an earlier run.
        duplicates: Notes whose note_id repeated one earlier in the same chunk or
            the chunk before.
        failed: Notes that failed in this call; see `OmopRun.failures`.
        gave_up: Notes not tried again: they failed ``max_attempts`` times.
        rows: Rows written per table.
        issues: Issues written.
        cache_stats: Coding cache hits and misses, when the builder has a cache.
        seconds: Wall time.
    """

    output_dir: str
    notes: int = 0
    built: int = 0
    extracted: int = 0
    reused: int = 0
    skipped: int = 0
    duplicates: int = 0
    failed: int = 0
    gave_up: int = 0
    rows: dict[str, int] = {}
    issues: int = 0
    cache_stats: Optional[CacheStats] = None
    seconds: float = 0.0

    def __str__(self) -> str:
        rate = self.notes / self.seconds * 3600 if self.seconds else 0
        lines = [
            f"{self.notes:,} notes in {self.seconds:,.1f} s ({rate:,.0f} notes/hour): "
            f"{self.built:,} built ({self.extracted:,} extracted, {self.reused:,} "
            f"from stored results), {self.skipped:,} already built, "
            f"{self.failed:,} failed, {self.gave_up:,} given up, "
            f"{self.duplicates:,} duplicates",
            "rows: "
            + (", ".join(f"{t} {n:,}" for t, n in self.rows.items()) or "none")
            + f"; {self.issues:,} issues",
        ]
        if self.cache_stats is not None:
            rate_text = (
                f"{self.cache_stats.hit_rate:.0%}"
                if self.cache_stats.hit_rate is not None
                else "n/a"
            )
            lines.append(
                f"coding cache: {self.cache_stats.hits:,} hits, "
                f"{self.cache_stats.misses:,} misses ({rate_text})"
            )
        lines.append(f"output: {self.output_dir}")
        return "\n".join(lines)


def _fingerprint(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _merge(parts: Sequence[OmopTables]) -> OmopTables:
    merged = OmopTables()
    for tables in parts:
        for table, rows in tables.rows.items():
            merged.rows.setdefault(table, []).extend(rows)
            merged.provenance.setdefault(table, []).extend(tables.provenance[table])
        merged.issues.extend(tables.issues)
        merged.mappings.extend(tables.mappings)
        if tables.cache_stats is not None:
            merged.cache_stats = _add_stats(merged.cache_stats, tables.cache_stats)
    return merged


def _add_stats(
    a: Optional[CacheStats], b: Optional[CacheStats]
) -> Optional[CacheStats]:
    if a is None or b is None:
        return a or b
    return CacheStats(
        hits=a.hits + b.hits, misses=a.misses + b.misses, stored=a.stored + b.stored
    )


class ChunkedRun:
    """Runs a pipeline over a dataset in chunks; see `OmopExtractionPipeline.run`."""

    def __init__(self, pipeline: "OmopExtractionPipeline", output_dir: str | Path):
        self.pipeline = pipeline
        self.builder = pipeline.builder
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Fingerprints of what the stored results depend on.

    def _extraction_fingerprint(self) -> str:
        extractor = self.pipeline.extractor
        return _fingerprint(
            {
                "entities": [
                    e.model_dump(mode="json", exclude={"omop"})
                    for e in self.pipeline.entity_definitions.entities
                ],
                "extractor": type(extractor).__qualname__,
                "model_id": getattr(extractor, "model_id", None),
                "instructions": getattr(extractor, "instructions", None),
            }
        )

    def _mapping_fingerprint(self) -> str:
        return _fingerprint(
            {
                "omop": {
                    key: m.model_dump(mode="json", by_alias=True)
                    for key, m in self.builder.mappings.items()
                },
                "dayfirst": self.builder.dayfirst,
            }
        )

    def _check(self, ledger: RunLedger, rebuilding: bool) -> None:
        extraction = self._extraction_fingerprint()
        stored = ledger.get_meta("extraction")
        if stored is None:
            ledger.set_meta("extraction", extraction)
        elif stored != extraction:
            message = (
                f"{self.output_dir} holds results extracted with other entity "
                "definitions or another extractor"
            )
            if not rebuilding:
                raise ValueError(
                    f"{message}. Use a new output folder, or the definitions and "
                    "extractor of that run"
                )
            logger.warning("%s; rebuilding from them anyway", message)

        mapping = self._mapping_fingerprint()
        stored = ledger.get_meta("mapping")
        if not rebuilding and stored not in (None, mapping) and ledger.has_built():
            raise ValueError(
                f"The omop mappings changed since tables were written to "
                f"{self.output_dir}. Rebuild them first, with no extraction calls: "
                "await pipeline.rebuild(output_dir)"
            )
        ledger.set_meta("mapping", mapping)

    def _recover(self, ledger: RunLedger) -> None:
        """Delete the files of parts that were not recorded as complete."""
        for part in ledger.uncommitted_parts():
            for folder in self.output_dir.iterdir():
                if folder.is_dir():
                    for name in (_part_name(part), f".{_part_name(part)}.tmp"):
                        with contextlib.suppress(FileNotFoundError):
                            (folder / name).unlink()
            ledger.drop_part(part)

    @contextlib.contextmanager
    def _ledger(self) -> Iterator[RunLedger]:
        ledger = RunLedger(self.output_dir / LEDGER_FILE)
        try:
            yield ledger
        finally:
            ledger.close()

    async def run(
        self,
        source: Any,
        chunk_size: int = 100,
        columns: Optional[dict[str, str]] = None,
        max_attempts: int = 3,
    ) -> RunSummary:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        started = time.monotonic()
        summary = RunSummary(output_dir=str(self.output_dir))
        with self._ledger() as ledger:
            self._recover(ledger)
            self._check(ledger, rebuilding=False)
            # While a chunk is built (in a thread), the next one is extracted.
            building: Optional[asyncio.Task] = None
            in_flight: set[int] = set()
            try:
                for index, chunk in enumerate(read_chunks(source, chunk_size, columns)):
                    pairs = await self._prepare(
                        ledger, chunk, summary, max_attempts, in_flight
                    )
                    if building is not None:
                        await building
                    in_flight = {n.note_id for n, _ in pairs}  # type: ignore[misc]
                    building = asyncio.create_task(self._build(ledger, pairs, summary))
                    summary.seconds = time.monotonic() - started
                    self._log_progress(index, summary)
                if building is not None:
                    await building
            finally:
                # Stopped partway: still write the chunk being built.
                if building is not None and not building.done():
                    with contextlib.suppress(Exception):
                        await building
        summary.seconds = time.monotonic() - started
        logger.info("Run finished\n%s", summary)
        return summary

    async def rebuild(self, chunk_size: int = 1000) -> RunSummary:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        started = time.monotonic()
        summary = RunSummary(output_dir=str(self.output_dir))
        with self._ledger() as ledger:
            self._recover(ledger)
            self._check(ledger, rebuilding=True)
            ledger.reset_for_rebuild()
            self._recover(ledger)
            after, index = -1, 0
            while ids := ledger.extracted_ids(after, chunk_size):
                after = ids[-1]
                pairs = ledger.load(ids)
                summary.notes += len(pairs)
                await self._build(ledger, pairs, summary)
                summary.seconds = time.monotonic() - started
                self._log_progress(index, summary)
                index += 1
        summary.reused = summary.built
        summary.seconds = time.monotonic() - started
        logger.info("Rebuild finished\n%s", summary)
        return summary

    @staticmethod
    def _log_progress(index: int, summary: RunSummary) -> None:
        rate = summary.notes / summary.seconds * 3600 if summary.seconds else 0
        logger.info(
            "Chunk %d: %d notes so far, %d built, %d skipped, %d failed "
            "(%.0f notes/hour)",
            index + 1,
            summary.notes,
            summary.built,
            summary.skipped,
            summary.failed,
            rate,
        )

    async def _prepare(
        self,
        ledger: RunLedger,
        chunk: list[ClinicalNote],
        summary: RunSummary,
        max_attempts: int,
        in_flight: set[int],
    ) -> list[tuple[ClinicalNote, EntityValidationResult]]:
        """Extract the notes of a chunk that need it; return those to build.

        `in_flight` holds the notes of the chunk being built meanwhile: they count as
        duplicates, since the ledger does not record them as built yet.
        """
        summary.notes += len(chunk)
        notes: dict[int, ClinicalNote] = {}
        for note in chunk:
            if note.note_id is None:
                raise ValueError(
                    "Every note needs a note_id in a chunked run: it is the key that "
                    "lets a stopped run resume"
                )
            if not 0 <= note.note_id <= MAX_NOTE_ID:
                raise ValueError(
                    f"note_id {note.note_id} is outside 0 to {MAX_NOTE_ID}"
                )
            if note.note_id in notes or note.note_id in in_flight:
                summary.duplicates += 1
                continue
            notes[note.note_id] = note

        states = ledger.states(list(notes))
        to_extract: list[ClinicalNote] = []
        stored: list[int] = []
        for note_id, note in notes.items():
            state = states.get(note_id)
            if state is None:
                to_extract.append(note)
            elif state.status == "built":
                summary.skipped += 1
            elif state.status == "extracted":
                stored.append(note_id)
            elif state.attempts >= max_attempts:
                summary.gave_up += 1
            elif state.stage == "extract":
                to_extract.append(note)
            else:
                stored.append(note_id)

        extracted = await self._extract(ledger, to_extract, summary)
        reused = ledger.load(stored)
        summary.reused += len(reused)
        return extracted + reused

    async def _extract(
        self, ledger: RunLedger, notes: list[ClinicalNote], summary: RunSummary
    ) -> list[tuple[ClinicalNote, EntityValidationResult]]:
        """Extract notes one by one, so that one failure does not lose the others."""
        if not notes:
            return []
        extractor = self.pipeline.extractor

        async def extract(note: ClinicalNote) -> EntityValidationResult | Exception:
            try:
                (result,) = await extractor([note.text])
                return result
            except Exception as e:
                return e

        results = await asyncio.gather(*(extract(n) for n in notes))
        done: list[tuple[ClinicalNote, EntityValidationResult]] = []
        failed: list[tuple[ClinicalNote, str]] = []
        for note, result in zip(notes, results):
            if isinstance(result, Exception):
                failed.append((note, f"{type(result).__name__}: {result}"))
            elif result.error is not None:
                failed.append((note, result.error))
            else:
                done.append((note, result))
        for note, error in failed:
            logger.warning("Extraction of note %s failed: %s", note.note_id, error)
        # Store the results right away: they are what the run paid for.
        ledger.save_extracted(done)
        ledger.save_failed(failed, "extract")
        summary.extracted += len(done)
        summary.failed += len(failed)
        return done

    async def _build(
        self,
        ledger: RunLedger,
        pairs: list[tuple[ClinicalNote, EntityValidationResult]],
        summary: RunSummary,
    ) -> None:
        if not pairs:
            return
        notes = [n for n, _ in pairs]
        results = [r for _, r in pairs]
        try:
            # Building may call the terminology mapper, which blocks.
            tables = await asyncio.to_thread(
                self.builder.build, notes, results, row_ids="note"
            )
            built = notes
        except Exception:
            # Build the notes one by one to find those that fail.
            parts, built, failed = [], [], []
            for note, result in pairs:
                try:
                    parts.append(
                        await asyncio.to_thread(
                            self.builder.build, [note], [result], row_ids="note"
                        )
                    )
                    built.append(note)
                except Exception as e:
                    logger.warning("Building note %s failed: %s", note.note_id, e)
                    failed.append((note, f"{type(e).__name__}: {e}"))
            ledger.save_failed(failed, "build")
            summary.failed += len(failed)
            tables = _merge(parts)

        part = ledger.new_part()
        counts = await asyncio.to_thread(_write_part, self.output_dir, part, tables)
        ledger.commit_part(part, [n.note_id for n in built])  # type: ignore[misc]
        summary.built += len(built)
        for table, n in counts.items():
            summary.rows[table] = summary.rows.get(table, 0) + n
        summary.issues += len(tables.issues)
        summary.cache_stats = _add_stats(summary.cache_stats, tables.cache_stats)


# --- reading the output -------------------------------------------------------------


class OmopRun:
    """Read the output folder of `OmopExtractionPipeline.run`.

    `table`, `issues`, `mappings` and `provenance` load a whole dataset into a
    DataFrame. For datasets larger than memory, `dataset` returns a
    `pyarrow.dataset.Dataset` to filter and scan lazily (or read the folders with
    DuckDB, Polars or Spark).
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        if not (self.output_dir / LEDGER_FILE).exists():
            raise FileNotFoundError(f"{self.output_dir} is not the output of a run")

    @property
    def tables(self) -> list[str]:
        """The OMOP tables with rows."""
        return sorted(
            p.name
            for p in self.output_dir.iterdir()
            if p.is_dir() and p.name in OMOP_TABLES and any(p.glob("part-*.parquet"))
        )

    def dataset(self, name: str) -> Any:
        """A `pyarrow.dataset.Dataset` of a table, or of ``provenance``, ``issues``
        or ``mappings``."""
        import pyarrow.dataset as ds

        return ds.dataset(self.output_dir / name, format="parquet")

    def _read(self, name: str, schema: "pa.Schema") -> pd.DataFrame:
        import pyarrow as pa

        folder = self.output_dir / name
        if not any(folder.glob("part-*.parquet")):
            return schema.empty_table().to_pandas()
        table = self.dataset(name).to_table()
        mapping = {pa.int64(): pd.Int64Dtype()}
        return table.to_pandas(types_mapper=mapping.get)

    def table(self, name: str) -> pd.DataFrame:
        """The rows of an OMOP table, e.g. ``"measurement"``."""
        if name not in OMOP_TABLES:
            raise KeyError(f"Unknown table {name!r}")
        return self._read(name, _schema(OMOP_TABLES[name]))

    def provenance(self) -> pd.DataFrame:
        """The note, entity and span of every row, keyed by ``table`` and ``row_id``."""
        import pyarrow as pa

        return self._read(
            PROVENANCE,
            _schema(
                RowProvenance,
                exclude=["note_index"],
                extra=[("table", pa.string()), ("row_id", pa.int64())],
            ),
        )

    def issues(self) -> pd.DataFrame:
        return self._read(ISSUES, _schema(OmopIssue, exclude=["note_index"]))

    def mappings(self) -> pd.DataFrame:
        """The coded texts; ``trace`` and ``usage`` are JSON strings."""
        return self._read(MAPPINGS, _schema(ConceptMapping))

    def failures(self) -> pd.DataFrame:
        """Notes that could not be extracted or built: their stage, error and tries."""
        ledger = RunLedger(self.output_dir / LEDGER_FILE)
        try:
            return ledger.failures()
        finally:
            ledger.close()

    def status(self) -> dict[str, int]:
        """Notes per status: ``built``, ``extracted`` (not built yet), ``failed``."""
        ledger = RunLedger(self.output_dir / LEDGER_FILE)
        try:
            return ledger.counts()
        finally:
            ledger.close()
