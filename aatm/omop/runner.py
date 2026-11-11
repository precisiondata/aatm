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

## Batch extraction

For large offline runs, ``batch=True`` extracts through the provider's batch endpoint
(OpenAI Batch API, Anthropic Message Batches, Gemini batch mode), about half the price
of live calls, with results within 24 hours:

```python
summary = await pipeline.run(
    "notes.parquet", "output/ecog", chunk_size=1000, batch=True, poll_every=300
)
```

Each chunk's notes to extract are submitted as one job; up to ``max_batches`` jobs
are pending at once. Pending jobs are polled every ``poll_every`` seconds; the notes
of a finished job are stored in the ledger and built like live ones. Only extraction
is batched: coding stays live.

Jobs are recorded in the ledger when submitted. A run that stops (a crash, Ctrl+C)
leaves them pending, and the next run, in batch mode or not, collects them instead of
submitting their notes again. Requests that fail within a job are recorded as failed
notes, tried again by a later run, live or in a batch, up to ``max_attempts``.
`OmopRun.batches` lists the jobs.

Batch calls are counted under the model id with ``@batch`` appended, and priced at
half the model's price unless ``prices`` has the ``@batch`` id (see
`aatm.llm.usage.TokenUsage.cost`).

## Progress, tokens and cost

A run logs a progress line every ``log_every`` seconds (60 by default), when a chunk
is done: notes read, throughput, the estimated time left, LLM calls, tokens and cost
so far, and the coding cache's hit rate. The time left needs the number of notes in
the source: it is read from Parquet metadata, DataFrames and lists, and can be given
with ``total`` for CSV files and queries.

At the end, the summary breaks the work down by stage (extraction, coding and
building): items, time, errors, LLM calls and tokens per model, counted with
`aatm.llm.track_usage`. Cost is estimated from the prices you give, per million input
and output tokens:

```python
summary = await pipeline.run(
    "notes.parquet", "output/ecog",
    prices={"openai/gpt-5-mini": (0.25, 2.00)},  # check your provider's prices
)
summary.stages["coding"].usage.by_model, summary.cost
```

The summary is printed (logged) and saved as JSON in ``runs/``, one file per call;
`OmopRun.summaries` reads them back. The file is also saved at every progress log,
so a long run can be watched from elsewhere, and kept when a run crashes.

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
  runs/000001-20261110-091500-run.json  the summary of each run and rebuild
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

from datetime import date, datetime, timezone
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
    Sized,
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
from pydantic import BaseModel, Field

from aatm.data_models.extractors import EntityValidationResult
from aatm.llm.usage import Prices, TokenUsage, model_prices, track_usage
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
from aatm.omop.shards import check_shard, shard_dirs, shard_of
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
    source: Any,
    chunk_size: int,
    columns: Optional[dict[str, str]] = None,
    shard: Optional[tuple[int, int]] = None,
) -> Iterator[list[ClinicalNote]]:
    """Read notes from a source in lists of `chunk_size` (the last may be shorter).

    With `shard` ``(index, count)``, only the notes of that shard are kept (see
    `aatm.omop.shards.shard_of`); notes without a `note_id` are kept, to be refused.
    """
    notes = _iter_notes(source, columns, read_size=max(chunk_size, 1000))
    if shard is not None and shard[1] > 1:
        index, count = check_shard(shard)
        notes = (
            n for n in notes if n.note_id is None or shard_of(n.note_id, count) == index
        )
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
    status TEXT NOT NULL,         -- submitted, extracted, built or failed
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
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,    -- the provider's job id
    model_id TEXT NOT NULL,
    status TEXT NOT NULL,         -- pending, or done once its results are stored
    notes INTEGER NOT NULL,
    submitted_at REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS batch_notes (
    batch_id TEXT NOT NULL,
    note_id INTEGER NOT NULL,
    PRIMARY KEY (batch_id, note_id)
);
"""


class _NoteState(BaseModel):
    status: Literal["submitted", "extracted", "built", "failed"]
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

    def _extracted(
        self, items: Sequence[tuple[ClinicalNote, EntityValidationResult]]
    ) -> None:
        self.con.executemany(
            "INSERT INTO notes (note_id, status, note, result, attempts, updated_at)"
            " VALUES (?, 'extracted', ?, ?, 1, ?) ON CONFLICT (note_id) DO UPDATE"
            " SET status = 'extracted', stage = NULL, note = excluded.note,"
            " result = excluded.result, error = NULL, attempts = attempts + 1,"
            " updated_at = excluded.updated_at",
            [
                (n.note_id, n.model_dump_json(), r.model_dump_json(), time.time())
                for n, r in items
            ],
        )

    def _failed(
        self,
        items: Sequence[tuple[ClinicalNote, str]],
        stage: Literal["extract", "build"],
    ) -> None:
        now = time.time()
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

    def save_extracted(
        self, items: Sequence[tuple[ClinicalNote, EntityValidationResult]]
    ) -> None:
        with self.con:
            self._extracted(items)

    def save_failed(
        self,
        items: Sequence[tuple[ClinicalNote, str]],
        stage: Literal["extract", "build"],
    ) -> None:
        """Record failed notes. A build failure keeps the stored extraction result."""
        with self.con:
            self._failed(items, stage)

    # Batch jobs.

    def save_submitted(
        self, batch_id: str, model_id: str, notes: Sequence[ClinicalNote]
    ) -> None:
        """Record a submitted batch job and mark its notes ``submitted``."""
        now = time.time()
        with self.con:
            self.con.execute(
                "INSERT INTO batches (batch_id, model_id, status, notes, submitted_at)"
                " VALUES (?, ?, 'pending', ?, ?)",
                (batch_id, model_id, len(notes), now),
            )
            self.con.executemany(
                "INSERT INTO batch_notes (batch_id, note_id) VALUES (?, ?)",
                [(batch_id, n.note_id) for n in notes],
            )
            # Attempts are counted when the result comes back.
            self.con.executemany(
                "INSERT INTO notes (note_id, status, note, updated_at)"
                " VALUES (?, 'submitted', ?, ?) ON CONFLICT (note_id) DO UPDATE"
                " SET status = 'submitted', stage = NULL, note = excluded.note,"
                " error = NULL, updated_at = excluded.updated_at",
                [(n.note_id, n.model_dump_json(), now) for n in notes],
            )

    def pending_batches(self) -> list[tuple[str, str, list[int]]]:
        """``(batch_id, model_id, note ids)`` of the jobs whose results are not stored."""
        batches = self.con.execute(
            "SELECT batch_id, model_id FROM batches WHERE status = 'pending'"
            " ORDER BY submitted_at"
        ).fetchall()
        return [
            (
                batch_id,
                model_id,
                [
                    i
                    for (i,) in self.con.execute(
                        "SELECT note_id FROM batch_notes WHERE batch_id = ?"
                        " ORDER BY note_id",
                        (batch_id,),
                    )
                ],
            )
            for batch_id, model_id in batches
        ]

    def notes(self, ids: Sequence[int]) -> list[ClinicalNote]:
        """The stored notes, in the order of `ids`."""
        found = {
            note_id: ClinicalNote.model_validate_json(note)
            for note_id, note in self._select("note_id, note", ids)
        }
        return [found[i] for i in ids]

    def finish_batch(
        self,
        batch_id: str,
        done: Sequence[tuple[ClinicalNote, EntityValidationResult]],
        failed: Sequence[tuple[ClinicalNote, str]],
    ) -> None:
        """Store the results of a finished job and mark it done, in one transaction."""
        with self.con:
            self._extracted(done)
            self._failed(failed, "extract")
            self.con.execute(
                "UPDATE batches SET status = 'done', finished_at = ?"
                " WHERE batch_id = ?",
                (time.time(), batch_id),
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

    def has_notes(self) -> bool:
        return self.con.execute("SELECT 1 FROM notes LIMIT 1").fetchone() is not None

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
        """Notes per status: ``extracted`` (not built yet), ``built``, ``failed``,
        and ``submitted`` (in a batch job not collected yet)."""
        return dict(
            self.con.execute("SELECT status, COUNT(*) FROM notes GROUP BY status")
        )

    def batches(self) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT batch_id, model_id, status, notes, submitted_at, finished_at"
            " FROM batches ORDER BY submitted_at",
            self.con,
        )

    def failures(self) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT note_id, stage, error, attempts, updated_at FROM notes"
            " WHERE status = 'failed' ORDER BY note_id",
            self.con,
        )


# --- the run ------------------------------------------------------------------------


STAGES = ("extraction", "coding", "building")


class StageStats(BaseModel):
    """What one stage of a run did, and what it cost.

    The stages overlap: the next chunk is extracted while one is coded and built, so
    their ``seconds`` can add up to more than the wall time.

    Attributes:
        items: Notes extracted (``extraction``), distinct texts coded (``coding``) or
            notes built (``building``).
        seconds: Time spent in the stage. For ``building``, the time to build and
            write the rows, coding excluded.
        errors: Notes whose extraction or build failed, or texts the coder failed on.
        usage: LLM calls and tokens, in all and per model.
        cost: Estimated cost, when the run was given ``prices``.
        unpriced: Models called that had no price.
    """

    items: int = 0
    seconds: float = 0.0
    errors: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost: Optional[float] = None
    unpriced: list[str] = []

    @property
    def per_minute(self) -> Optional[float]:
        return self.items / self.seconds * 60 if self.seconds else None

    def describe(self, name: str) -> str:
        unit = "texts" if name == "coding" else "notes"
        text = f"{name}: {self.items:,} {unit} in {_duration(self.seconds)}"
        if self.per_minute is not None:
            text += f" ({self.per_minute:,.0f}/min)"
        if self.errors:
            text += f", {self.errors:,} errors"
        if self.usage.calls:
            text += (
                f", {self.usage.calls:,} LLM calls, "
                f"{self.usage.input_tokens:,} input and "
                f"{self.usage.output_tokens:,} output tokens"
            )
        if self.cost is not None and self.usage.calls:
            text += f", cost {self.cost:,.4f}"
            if self.unpriced:
                text += f" (no price for {', '.join(self.unpriced)})"
        return text


def _duration(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


class RunSummary(BaseModel):
    """What one call of `run` or `rebuild` did.

    It is logged at the end and saved as JSON in the ``runs`` folder of the output
    (see `OmopRun.summaries`). While the call goes on, the file is saved again at
    every progress log, with ``completed`` false.

    Attributes:
        output_dir: The output folder.
        kind: ``run`` or ``rebuild``.
        started_at: When the call started (UTC).
        finished_at: When it ended, also when it stopped with an error.
        completed: Whether it went through the whole source.
        error: The error that stopped it, if any.
        total: Notes in the source, when known (see the ``total`` argument of
            `run`); it gives the estimated time left.
        notes: Notes read from the source (or from the ledger, for a rebuild).
        built: Notes whose rows were written by this call.
        extracted: Notes extracted by this call: one model call each.
        reused: Notes built from a stored extraction result, with no model call.
        skipped: Notes already built by an earlier run.
        duplicates: Notes whose note_id repeated one earlier in the same chunk or
            the chunk before.
        failed: Notes that failed in this call; see `OmopRun.failures`.
        gave_up: Notes not tried again: they failed ``max_attempts`` times.
        batches_submitted: Provider batch jobs submitted by this call.
        batches_collected: Batch jobs whose results this call stored, those of
            earlier runs included.
        rows: Rows written per table.
        issues: Issues written.
        cache_stats: Coding cache hits and misses, when the builder has a cache.
        stages: Items, time, errors, tokens and cost of ``extraction``, ``coding``
            and ``building``.
        cost: Estimated cost of all stages, when the run was given ``prices``.
        seconds: Wall time.
    """

    output_dir: str
    kind: Literal["run", "rebuild"] = "run"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    completed: bool = False
    error: Optional[str] = None
    total: Optional[int] = None
    notes: int = 0
    built: int = 0
    extracted: int = 0
    reused: int = 0
    skipped: int = 0
    duplicates: int = 0
    failed: int = 0
    gave_up: int = 0
    batches_submitted: int = 0
    batches_collected: int = 0
    rows: dict[str, int] = {}
    issues: int = 0
    cache_stats: Optional[CacheStats] = None
    stages: dict[str, StageStats] = Field(
        default_factory=lambda: {name: StageStats() for name in STAGES}
    )
    cost: Optional[float] = None
    seconds: float = 0.0

    @property
    def usage(self) -> TokenUsage:
        """LLM calls and tokens of all stages."""
        usage = TokenUsage()
        for stage in self.stages.values():
            usage.merge(stage.usage)
        return usage

    @property
    def done(self) -> int:
        """Notes finished: built, skipped, failed, given up or duplicates.

        During a run it lags `notes` by the chunk being extracted ahead.
        """
        return self.built + self.skipped + self.failed + self.gave_up + self.duplicates

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated time left, from the notes finished so far and `total`."""
        if self.total is None or not self.done or not self.seconds:
            return None
        return max(self.total - self.done, 0) * self.seconds / self.done

    def price(self, prices: Optional[Prices]) -> None:
        """Set the estimated cost of each stage, and in all, from `prices`."""
        if not model_prices(prices):
            return
        for stage in self.stages.values():
            stage.cost = stage.usage.cost(prices)
            stage.unpriced = stage.usage.unpriced(prices)
        self.cost = sum(s.cost or 0.0 for s in self.stages.values())

    def progress(self) -> str:
        """One log line: progress, throughput, time left, tokens and cost."""
        rate = self.done / self.seconds * 60 if self.seconds else 0
        if self.total:
            share = self.done / self.total
            done = f"{self.done:,} of {self.total:,} notes done ({share:.1%})"
        else:
            done = f"{self.done:,} notes done"
        parts = [f"{done} in {_duration(self.seconds)}, {rate:,.0f} notes/min"]
        eta = self.eta_seconds
        if eta is not None:
            parts.append(f"about {_duration(eta)} left")
        parts.append(
            f"{self.built:,} built, {self.skipped:,} skipped, {self.failed:,} failed"
        )
        if self.batches_submitted or self.batches_collected:
            parts.append(
                f"{self.batches_submitted:,} batch jobs submitted, "
                f"{self.batches_collected:,} collected"
            )
        usage = self.usage
        if usage.calls:
            tokens = usage.input_tokens + usage.output_tokens
            parts.append(f"{usage.calls:,} LLM calls, {tokens:,} tokens")
        if self.cost is not None:
            parts.append(f"cost {self.cost:,.4f}")
        if self.cache_stats is not None and self.cache_stats.hit_rate is not None:
            parts.append(f"cache hits {self.cache_stats.hit_rate:.0%}")
        return "; ".join(parts)

    def __str__(self) -> str:
        rate = self.notes / self.seconds * 3600 if self.seconds else 0
        status = "completed" if self.completed else "stopped"
        if self.error:
            status += f" ({self.error})"
        lines = [
            f"{self.kind} {status}: {self.notes:,} notes in "
            f"{_duration(self.seconds)} ({rate:,.0f} notes/hour): "
            f"{self.built:,} built ({self.extracted:,} extracted, {self.reused:,} "
            f"from stored results), {self.skipped:,} already built, "
            f"{self.failed:,} failed, {self.gave_up:,} given up, "
            f"{self.duplicates:,} duplicates",
            "rows: "
            + (", ".join(f"{t} {n:,}" for t, n in self.rows.items()) or "none")
            + f"; {self.issues:,} issues",
        ]
        if self.batches_submitted or self.batches_collected:
            lines.append(
                f"batch jobs: {self.batches_submitted:,} submitted, "
                f"{self.batches_collected:,} collected"
            )
        for name, stage in self.stages.items():
            if stage.items or stage.usage.calls or stage.errors:
                lines.append(stage.describe(name))
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
        if self.cost is not None:
            lines.append(f"estimated cost: {self.cost:,.4f}")
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
        merged.coded_texts += tables.coded_texts
        merged.coding_errors += tables.coding_errors
        merged.coding_seconds += tables.coding_seconds
    return merged


def _add_stats(
    a: Optional[CacheStats], b: Optional[CacheStats]
) -> Optional[CacheStats]:
    if a is None or b is None:
        return a or b
    return CacheStats(
        hits=a.hits + b.hits, misses=a.misses + b.misses, stored=a.stored + b.stored
    )


def _batch_key(note_id: int) -> str:
    return f"note-{note_id}"


def _supports_batch(extractor: Any) -> bool:
    if not callable(getattr(extractor, "submit_batch", None)):
        return False
    return bool(getattr(extractor, "supports_batch", True))


class _BatchQueue:
    """The provider batch jobs of a run: submitted, polled and collected.

    Jobs and their notes are recorded in the ledger when submitted, so a run that
    stops leaves them there, and the next run (`adopt`) collects them instead of
    submitting the notes again.
    """

    def __init__(
        self,
        extractor: Any,
        ledger: RunLedger,
        summary: RunSummary,
        poll_every: float,
    ):
        self.extractor = extractor
        self.ledger = ledger
        self.summary = summary
        self.poll_every = poll_every
        self.pending: dict[str, list[int]] = {}
        self.note_ids: set[int] = set()
        # Notes of jobs submitted by earlier runs: the source lists them again.
        self.resumed_ids: set[int] = set()
        self.next_poll = 0.0

    def adopt(self) -> None:
        """Take over the jobs earlier runs left pending."""
        pending = self.ledger.pending_batches()
        if not pending:
            return
        if not callable(getattr(self.extractor, "poll_batch", None)):
            raise ValueError(
                f"{len(pending)} batch jobs submitted by an earlier run are still "
                "pending, and this extractor cannot poll batch jobs: use the "
                "extractor of that run"
            )
        for batch_id, _, ids in pending:
            self.pending[batch_id] = ids
            self.note_ids.update(ids)
            self.resumed_ids.update(ids)
        logger.info(
            "Collecting %d batch jobs submitted by an earlier run (%d notes)",
            len(pending),
            len(self.resumed_ids),
        )

    async def submit(self, notes: list[ClinicalNote]) -> None:
        stage = self.summary.stages["extraction"]
        started = time.perf_counter()
        batch_id = await self.extractor.submit_batch(
            [(_batch_key(n.note_id), n.text) for n in notes]  # type: ignore[arg-type]
        )
        stage.seconds += time.perf_counter() - started
        model_id = getattr(self.extractor, "model_id", None) or "unknown"
        self.ledger.save_submitted(batch_id, model_id, notes)
        ids = [n.note_id for n in notes]
        self.pending[batch_id] = ids  # type: ignore[assignment]
        self.note_ids.update(ids)  # type: ignore[arg-type]
        self.summary.batches_submitted += 1
        logger.info("Submitted batch job %s with %d notes", batch_id, len(notes))

    async def collect(
        self, wait: bool
    ) -> list[tuple[ClinicalNote, EntityValidationResult]]:
        """The notes extracted by jobs that finished since the last call.

        Polls at most every `poll_every` seconds. With `wait`, waits until a job
        finishes (or none is left); otherwise returns at once.
        """
        logged = False
        while self.pending:
            now = time.monotonic()
            if now >= self.next_poll:
                self.next_poll = now + self.poll_every
                finished, ready = await self._poll()
                if finished or not wait:
                    return ready
            elif not wait:
                return []
            if not logged:
                logged = True
                logger.info(
                    "Waiting for %d batch jobs (%d notes)",
                    len(self.pending),
                    len(self.note_ids),
                )
            await asyncio.sleep(max(self.next_poll - time.monotonic(), 0.0))
        return []

    async def _poll(
        self,
    ) -> tuple[int, list[tuple[ClinicalNote, EntityValidationResult]]]:
        stage = self.summary.stages["extraction"]
        started = time.perf_counter()
        with track_usage() as usage:
            outcomes = await asyncio.gather(
                *(self._poll_one(b, ids) for b, ids in list(self.pending.items()))
            )
        stage.seconds += time.perf_counter() - started
        stage.usage.merge(usage)
        finished = [o for o in outcomes if o is not None]
        return len(finished), [pair for pairs in finished for pair in pairs]

    async def _poll_one(
        self, batch_id: str, ids: list[int]
    ) -> Optional[list[tuple[ClinicalNote, EntityValidationResult]]]:
        notes = self.ledger.notes(ids)
        items = [(_batch_key(n.note_id), n.text) for n in notes]  # type: ignore[arg-type]
        try:
            results = await self.extractor.poll_batch(batch_id, items)
        except Exception as e:  # noqa: BLE001 - try again at the next poll
            logger.warning("Polling batch job %s failed: %s", batch_id, e)
            return None
        if results is None:
            return None

        done: list[tuple[ClinicalNote, EntityValidationResult]] = []
        failed: list[tuple[ClinicalNote, str]] = []
        for note, (key, _) in zip(notes, items):
            result = results.get(key)
            if result is None:
                failed.append((note, f"batch {batch_id} returned no result"))
            elif result.error is not None:
                failed.append((note, result.error))
            else:
                done.append((note, result))
        for note, error in failed:
            logger.warning("Extraction of note %s failed: %s", note.note_id, error)
        self.ledger.finish_batch(batch_id, done, failed)
        del self.pending[batch_id]
        self.note_ids.difference_update(ids)

        summary = self.summary
        summary.batches_collected += 1
        summary.extracted += len(done)
        summary.failed += len(failed)
        stage = summary.stages["extraction"]
        stage.items += len(notes)
        stage.errors += len(failed)
        logger.info(
            "Batch job %s finished: %d notes extracted, %d failed",
            batch_id,
            len(done),
            len(failed),
        )
        return done


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

    def _check_shard(self, ledger: RunLedger, shard: tuple[int, int]) -> None:
        """Refuse a folder that holds another shard, or notes of the whole dataset."""
        wanted = f"{shard[0]}/{shard[1]}"
        stored = ledger.get_meta("shard")
        if stored is None and ledger.has_notes():
            stored = "0/1"  # made before shards were recorded: a whole dataset
        if stored not in (None, wanted):
            def describe(value: str) -> str:
                index, count = value.split("/")
                if count == "1":
                    return "the whole dataset"
                return f"shard {index} of {count}"

            raise ValueError(
                f"{self.output_dir} holds {describe(stored)}, not "
                f"{describe(wanted)}: use another output folder"
            )
        ledger.set_meta("shard", wanted)

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
        total: Optional[int] = None,
        prices: Optional[Prices] = None,
        log_every: float = 60.0,
        batch: bool = False,
        poll_every: float = 60.0,
        max_batches: int = 20,
        shard: Optional[tuple[int, int]] = None,
    ) -> RunSummary:
        shard = check_shard(shard) if shard is not None else (0, 1)
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if max_batches < 1:
            raise ValueError("max_batches must be >= 1")
        if poll_every < 0:
            raise ValueError("poll_every must be >= 0")
        if batch and not _supports_batch(self.pipeline.extractor):
            raise ValueError(
                "batch=True needs an extractor that can submit provider batch jobs: "
                "SimpleLLMExtractor with an openai, anthropic or gemini model"
            )
        if total is None:
            total = count_notes(source)
            if total is not None and shard[1] > 1:
                total = -(-total // shard[1])  # an estimate of the shard's notes
        if shard_dirs(self.output_dir):
            raise ValueError(
                f"{self.output_dir} holds the folders of a sharded run: run the "
                "shards again with run_sharded (or shard= and each shard's folder)"
            )
        summary = RunSummary(output_dir=str(self.output_dir), kind="run", total=total)
        with (
            self._ledger() as ledger,
            self._report(summary, prices, log_every) as report,
        ):
            self._recover(ledger)
            self._check_shard(ledger, shard)
            self._check(ledger, rebuilding=False)
            # Jobs left pending by an earlier run are collected in either mode.
            queue = _BatchQueue(self.pipeline.extractor, ledger, summary, poll_every)
            queue.adopt()
            # While a chunk is built (in a thread), the next one is extracted.
            building: Optional[asyncio.Task] = None
            in_flight: set[int] = set()
            try:
                for chunk in read_chunks(source, chunk_size, columns, shard):
                    pairs = await self._prepare(
                        ledger,
                        chunk,
                        summary,
                        max_attempts,
                        in_flight,
                        queue,
                        max_batches if batch else None,
                    )
                    if building is not None:
                        await building
                        report.tick()
                    in_flight = {n.note_id for n, _ in pairs}  # type: ignore[misc]
                    building = asyncio.create_task(self._build(ledger, pairs, summary))
                if building is not None:
                    await building
                    building = None
                while queue.pending:
                    await self._build(ledger, await queue.collect(wait=True), summary)
                    report.tick()
            finally:
                # Stopped partway: still write the chunk being built.
                if building is not None and not building.done():
                    with contextlib.suppress(Exception):
                        await building
            summary.completed = True
        return summary

    async def rebuild(
        self,
        chunk_size: int = 1000,
        prices: Optional[Prices] = None,
        log_every: float = 60.0,
    ) -> RunSummary:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        shards = shard_dirs(self.output_dir)
        if shards and not (self.output_dir / LEDGER_FILE).exists():
            from aatm.omop.shards import combine_summaries

            summaries = [
                await ChunkedRun(self.pipeline, folder).rebuild(
                    chunk_size, prices, log_every
                )
                for folder in shards
            ]
            return combine_summaries(summaries, self.output_dir)
        summary = RunSummary(output_dir=str(self.output_dir), kind="rebuild")
        with (
            self._ledger() as ledger,
            self._report(summary, prices, log_every) as report,
        ):
            self._recover(ledger)
            self._check(ledger, rebuilding=True)
            ledger.reset_for_rebuild()
            self._recover(ledger)
            summary.total = ledger.counts().get("extracted", 0)
            after = -1
            while ids := ledger.extracted_ids(after, chunk_size):
                after = ids[-1]
                pairs = ledger.load(ids)
                summary.notes += len(pairs)
                await self._build(ledger, pairs, summary)
                summary.reused = summary.built
                report.tick()
            summary.completed = True
        return summary

    @contextlib.contextmanager
    def _report(
        self, summary: RunSummary, prices: Optional[Prices], log_every: float
    ) -> Iterator["_Reporter"]:
        """Log progress while the block runs, then log and save the summary."""
        reporter = _Reporter(summary, self.output_dir, prices, log_every)
        try:
            yield reporter
        except BaseException as e:
            summary.error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            raise
        finally:
            reporter.finish()

    async def _prepare(
        self,
        ledger: RunLedger,
        chunk: list[ClinicalNote],
        summary: RunSummary,
        max_attempts: int,
        in_flight: set[int],
        queue: _BatchQueue,
        max_batches: Optional[int],
    ) -> list[tuple[ClinicalNote, EntityValidationResult]]:
        """Extract the notes of a chunk that need it; return those to build.

        `in_flight` holds the notes of the chunk being built meanwhile: they count as
        duplicates, since the ledger does not record them as built yet.

        With `max_batches` (batch mode), the notes to extract are submitted as one
        batch job, waiting first while `max_batches` jobs are pending. The notes of
        jobs that finished meanwhile are returned with the chunk's.
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
                if note.note_id not in queue.resumed_ids:
                    summary.duplicates += 1
                continue
            notes[note.note_id] = note

        states = ledger.states(list(notes))
        to_extract: list[ClinicalNote] = []
        stored: list[int] = []
        for note_id, note in notes.items():
            state = states.get(note_id)
            resumed = note_id in queue.resumed_ids  # counted when its job is collected
            if state is None:
                to_extract.append(note)
            elif state.status == "built":
                if not resumed:
                    summary.skipped += 1
            elif state.status == "submitted" and note_id in queue.note_ids:
                if not resumed:
                    summary.duplicates += 1
            elif state.status == "submitted":
                to_extract.append(note)  # its job is gone from the ledger
            elif state.status == "extracted":
                stored.append(note_id)
            elif state.attempts >= max_attempts:
                summary.gave_up += 1
            elif state.stage == "extract":
                to_extract.append(note)
            else:
                stored.append(note_id)

        ready: list[tuple[ClinicalNote, EntityValidationResult]] = []
        if max_batches is not None and to_extract:
            while len(queue.pending) >= max_batches:
                ready += await queue.collect(wait=True)
            await queue.submit(to_extract)
            extracted = []
        else:
            extracted = await self._extract(ledger, to_extract, summary)
        ready += await queue.collect(wait=False)
        reused = ledger.load(stored)
        summary.reused += len(reused)
        return extracted + reused + ready

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

        stage = summary.stages["extraction"]
        started = time.perf_counter()
        # The chunk being built meanwhile runs in another task: its calls are not
        # counted here.
        with track_usage() as usage:
            results = await asyncio.gather(*(extract(n) for n in notes))
        stage.seconds += time.perf_counter() - started
        stage.usage.merge(usage)
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
        stage.items += len(notes)
        stage.errors += len(failed)
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
        started = time.perf_counter()
        failed: list[tuple[ClinicalNote, str]] = []
        # The coder's calls, made in the build thread, count as coding.
        with track_usage() as usage:
            try:
                # Building may call the terminology mapper, which blocks.
                tables = await asyncio.to_thread(
                    self.builder.build, notes, results, row_ids="note"
                )
                built = notes
            except Exception:
                # Build the notes one by one to find those that fail.
                parts, built = [], []
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

        coding = summary.stages["coding"]
        coding.items += tables.coded_texts
        coding.errors += tables.coding_errors
        coding.seconds += tables.coding_seconds
        coding.usage.merge(usage)
        building = summary.stages["building"]
        building.items += len(built)
        building.errors += len(failed)
        building.seconds += max(
            time.perf_counter() - started - tables.coding_seconds, 0.0
        )


RUNS = "runs"


class _Reporter:
    """Logs the progress of a run every `log_every` seconds and saves its summary."""

    def __init__(
        self,
        summary: RunSummary,
        output_dir: Path,
        prices: Optional[Prices],
        log_every: float,
    ):
        self.summary = summary
        self.prices = prices
        self.log_every = log_every
        self.started = time.perf_counter()
        self.last_log: Optional[float] = None
        summary.started_at = datetime.now(timezone.utc)
        # Numbered, so that the files sort in order even when the clock does not.
        folder = output_dir / RUNS
        number = 1 + max(
            (int(p.name[:6]) for p in folder.glob("*.json") if p.name[:6].isdigit()),
            default=0,
        )
        stamp = summary.started_at.strftime("%Y%m%d-%H%M%S")
        self.path = folder / f"{number:06d}-{stamp}-{summary.kind}.json"
        self.save()

    def _update(self) -> None:
        self.summary.seconds = time.perf_counter() - self.started
        self.summary.price(self.prices)

    def save(self) -> None:
        """Write the summary as JSON, atomically, so it can be read during the run."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(self.summary.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def tick(self) -> None:
        """Called after each chunk: log and save if `log_every` seconds went by."""
        now = time.perf_counter()
        if self.last_log is not None and now - self.last_log < self.log_every:
            return
        self.last_log = now
        self._update()
        logger.info("Progress: %s", self.summary.progress())
        with contextlib.suppress(OSError):
            self.save()

    def finish(self) -> None:
        self._update()
        self.summary.finished_at = datetime.now(timezone.utc)
        title = "Run" if self.summary.kind == "run" else "Rebuild"
        state = "finished" if self.summary.completed else "stopped"
        logger.info("%s %s\n%s", title, state, self.summary)
        try:
            self.save()
            logger.info("Summary saved to %s", self.path)
        except OSError as e:
            logger.warning("Could not save the run summary to %s: %s", self.path, e)


def count_notes(source: Any) -> Optional[int]:
    """The number of notes in a source, when it is cheap to know, else ``None``.

    Known for DataFrames, sized collections (lists) and Parquet files or folders
    (from their metadata). Not for CSV files, queries or iterators: pass ``total``.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists() and (path.is_dir() or path.suffix == ".parquet"):
            import pyarrow.dataset as ds

            with contextlib.suppress(Exception):
                return ds.dataset(path, format="parquet").count_rows()
        return None
    if isinstance(source, _FrameSource):
        return None
    if isinstance(source, Sized):
        return len(source)
    return None


# --- reading the output -------------------------------------------------------------


class OmopRun:
    """Read the output folder of `OmopExtractionPipeline.run`.

    `table`, `issues`, `mappings` and `provenance` load a whole dataset into a
    DataFrame. For datasets larger than memory, `dataset` returns a
    `pyarrow.dataset.Dataset` to filter and scan lazily (or read the folders with
    DuckDB, Polars or Spark).

    Given the parent folder of a sharded run (`aatm.omop.run_sharded`), it reads every
    shard as one run: the tables hold the rows of all shards, and `status`,
    `failures`, `batches` and `summaries` cover them all.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        if (self.output_dir / LEDGER_FILE).exists():
            self.folders = [self.output_dir]
        else:
            self.folders = [
                f for f in shard_dirs(self.output_dir) if (f / LEDGER_FILE).exists()
            ]
        if not self.folders:
            raise FileNotFoundError(f"{self.output_dir} is not the output of a run")

    @property
    def sharded(self) -> bool:
        """Whether this is the parent folder of a sharded run."""
        return self.folders != [self.output_dir]

    def _parts(self, name: str) -> list[Path]:
        return [p for f in self.folders for p in sorted((f / name).glob("part-*.parquet"))]

    @property
    def tables(self) -> list[str]:
        """The OMOP tables with rows."""
        return sorted(name for name in OMOP_TABLES if self._parts(name))

    def dataset(self, name: str) -> Any:
        """A `pyarrow.dataset.Dataset` of a table, or of ``provenance``, ``issues``
        or ``mappings``."""
        import pyarrow.dataset as ds

        if not self.sharded:
            return ds.dataset(self.output_dir / name, format="parquet")
        return ds.dataset([str(p) for p in self._parts(name)], format="parquet")

    def _read(self, name: str, schema: "pa.Schema") -> pd.DataFrame:
        import pyarrow as pa

        if not self._parts(name):
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

    def _ledgers(self) -> Iterator[RunLedger]:
        for folder in self.folders:
            ledger = RunLedger(folder / LEDGER_FILE)
            try:
                yield ledger
            finally:
                ledger.close()

    def _concat(self, frames: list[pd.DataFrame], by: str) -> pd.DataFrame:
        if len(frames) == 1:
            return frames[0]
        frames = [f for f in frames if not f.empty] or frames[:1]
        return pd.concat(frames, ignore_index=True).sort_values(by, ignore_index=True)

    def failures(self) -> pd.DataFrame:
        """Notes that could not be extracted or built: their stage, error and tries."""
        return self._concat([ledger.failures() for ledger in self._ledgers()], "note_id")

    def summaries(self) -> list[RunSummary]:
        """The summaries of the runs and rebuilds made in this folder, oldest first.

        A run still going on, or one that stopped without finishing (a crash), has
        ``completed`` false. For a sharded run, each shard's runs are listed, by
        shard; combine them with `aatm.omop.shards.combine_summaries`.
        """
        summaries = []
        for folder in self.folders:
            runs = folder / RUNS
            if runs.is_dir():
                summaries += [
                    RunSummary.model_validate_json(path.read_text(encoding="utf-8"))
                    for path in sorted(runs.glob("*.json"))
                ]
        return summaries

    def status(self) -> dict[str, int]:
        """Notes per status: ``built``, ``extracted`` (not built yet), ``failed``,
        and ``submitted`` (in a pending batch job)."""
        counts: dict[str, int] = {}
        for ledger in self._ledgers():
            for status, n in ledger.counts().items():
                counts[status] = counts.get(status, 0) + n
        return counts

    def batches(self) -> pd.DataFrame:
        """The provider batch jobs of batch runs: id, model, status and notes."""
        return self._concat(
            [ledger.batches() for ledger in self._ledgers()], "submitted_at"
        )
