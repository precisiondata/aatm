"""PostgreSQL with the pgvector extension as a vector database for AATM's index.

`PgVectorStore` implements `aatm.vector_stores.VectorStore` over one table per
embedding model (see `pg_table_name`):

```sql
CREATE TABLE aatm_expressions_qwen3_06b (
    id        text PRIMARY KEY,
    embedding vector(1024) NOT NULL,
    document  text,
    metadata  jsonb NOT NULL
);
```

Next to it, the table `aatm_index_metadata` holds each index's dimension and
AATM metadata, e.g. the embedding precision (see `aatm.precision`).

**Distances** are the *squared* Euclidean distance, as in ChromaDB's ``l2``
space and AATM's Pinecone indexes, so ``1 - distance``, selectors and thresholds
work unchanged. pgvector's ``<->`` operator returns the Euclidean distance;
`PgVectorStore.query` squares it.

**Filters** use the neutral format of `aatm.vector_stores` and become ``WHERE``
clauses on ``metadata`` (see `to_pg_filter`), applied during the vector search:
equality and ``$in`` use jsonb containment, which the GIN index on ``metadata``
serves. With pgvector 0.8 or later, `PgVectorStore` turns on *iterative index
scans*, so a filtered HNSW search keeps scanning until it has ``top_k`` matches
instead of returning fewer.

**Indexes** (HNSW on the embeddings, GIN on the metadata) are created by
`create_pg_indexes` after the bulk load, which is much faster than updating
them row by row. HNSW indexes ``vector`` columns of up to 2,000 dimensions;
larger embeddings (up to 4,000) are indexed as ``halfvec``, half precision,
and their distances still come from the full-precision vectors. Without an
HNSW index the search is exact.

Needs the ``aatm[pgvector]`` extra (``psycopg`` and ``pgvector``). Only
`PgVectorStore` and the table functions import them, so filter translation and
naming work without it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np

from aatm.logs import get_logger
from aatm.vector_db import PG_ITERATIVE_SCAN_MODES, VectorDatabaseConfigError
from aatm.vector_stores import Hit, UnsupportedFilterError, VectorStore

logger = get_logger(__name__)

INDEX_METADATA_TABLE = "aatm_index_metadata"
"""Table holding each index's dimension and AATM metadata."""

PG_IDENTIFIER_MAX_LENGTH = 63
"""PostgreSQL truncates identifiers longer than this many bytes."""

HNSW_MAX_VECTOR_DIMENSIONS = 2000
"""Largest ``vector`` dimension an HNSW index accepts."""

HNSW_MAX_HALFVEC_DIMENSIONS = 4000
"""Largest ``halfvec`` dimension an HNSW index accepts."""

HNSW_MAX_EF_SEARCH = 1000
"""Largest value of ``hnsw.ef_search``."""

DEFAULT_EF_SEARCH = 100
"""Candidates the HNSW search keeps (``hnsw.ef_search``); raised to ``top_k`` per query."""

ITERATIVE_SCAN_MODES = PG_ITERATIVE_SCAN_MODES
"""Values of pgvector's ``hnsw.iterative_scan``."""

DEFAULT_ITERATIVE_SCAN = "relaxed_order"
"""Keep scanning a filtered HNSW search until ``top_k`` rows match.

Results come back in roughly the right order; `PgVectorStore.query` sorts them.
"""

ITERATIVE_SCAN_MIN_VERSION = (0, 8, 0)
"""First pgvector version with iterative index scans."""

DEFAULT_HNSW_M = 24
"""HNSW ``m``: links per node. pgvector's own default is 16.

With pgvector's defaults (``m=16``, ``ef_construction=64``), filtered searches on the
full index found 0.877 (Measurement) and 0.955 (Condition) of the true 20 nearest
records; with ``m=24``, ``ef_construction=128``, 0.925 and 0.967, and unfiltered 0.982
instead of 0.963, in an index of the same size that takes about a third longer to
build. See ``benchmarks/vector_db_backends``.
"""

DEFAULT_HNSW_EF_CONSTRUCTION = 128
"""HNSW ``ef_construction``: candidates examined per vector while building.

pgvector's own default is 64. See `DEFAULT_HNSW_M`.
"""

PSYCOPG_MISSING_MESSAGE = (
    "PostgreSQL support needs the 'psycopg' and 'pgvector' packages. Install them "
    "with `pip install 'aatm[pgvector]'` (or `uv sync --extra pgvector`)."
)


class PgVectorConfigError(VectorDatabaseConfigError):
    """Raised when the database cannot hold AATM's index, e.g. pgvector is missing."""


class PgVectorConnectionError(PgVectorConfigError):
    """Raised when PostgreSQL cannot be reached at all, e.g. no server is running."""


# --- Names -----------------------------------------------------------------------

_INVALID_NAME_CHARS = re.compile(r"[^a-z0-9_]+")


def pg_table_name(prefix: str, model_name: str) -> str:
    """Return a safe PostgreSQL table name for one embedding model's index.

    Names hold lowercase letters, digits and ``_``, start with a letter or
    ``_``, and have at most 63 characters. Other characters become ``_``. A
    name that is too long is shortened and gets a short hash of the full name,
    so two models never share a table.

    Args:
        prefix: Common prefix, e.g. ``aatm_expressions``. May be empty.
        model_name: Registry name of the embedding model, e.g. ``qwen3-06B``.

    Returns:
        The table name, e.g. ``aatm_expressions_qwen3_06b``.

    Raises:
        ValueError: If nothing valid is left of the name.
    """
    raw = f"{prefix}_{model_name}" if prefix else model_name
    name = _INVALID_NAME_CHARS.sub("_", raw.lower())
    name = re.sub(r"_{2,}", "_", name).strip("_")
    if not name:
        raise ValueError(f"Cannot build a PostgreSQL table name from {raw!r}.")
    if name[0].isdigit():
        name = f"t_{name}"
    if len(name) > PG_IDENTIFIER_MAX_LENGTH:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        keep = PG_IDENTIFIER_MAX_LENGTH - len(digest) - 1
        name = f"{name[:keep].rstrip('_')}_{digest}"
    return name


_SAFE_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def _ident(name: str) -> str:
    """Quote a table or index name. Only names `pg_table_name` makes are accepted."""
    if not _SAFE_NAME.match(name):
        raise ValueError(f"Not a valid AATM table name: {name!r}")
    return f'"{name}"'


# --- Filters ---------------------------------------------------------------------

_COMPARISONS = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def to_pg_filter(
    where: Mapping[str, Any], column: str = "metadata"
) -> tuple[str, list[Any]]:
    """Translate a neutral (ChromaDB-style) ``where`` filter to a SQL condition.

    The result is a condition with ``%s`` placeholders and its parameters, for
    psycopg. Values are always parameters, never part of the SQL text.

    - ``{"field": value}`` and ``$eq``: ``metadata @> '{"field": value}'``
      (jsonb containment, so strings, numbers and booleans stay apart, and the
      GIN index serves it).
    - ``$in``: one containment per value, joined with ``OR``.
    - ``$ne`` / ``$nin``: the field exists and the containment does not hold.
    - ``$gt``, ``$gte``, ``$lt``, ``$lte``: numbers only, on numeric fields.
    - ``$and`` / ``$or``, and a dict with several fields (their ``$and``).

    As in ChromaDB, a record that lacks a field never matches a condition on it.

    Args:
        where: The filter.
        column: The jsonb column holding the metadata.

    Returns:
        The SQL condition and its parameters.

    Raises:
        UnsupportedFilterError: For other operators or values.
    """
    if not isinstance(where, Mapping) or not where:
        raise UnsupportedFilterError(f"Not a filter: {where!r}")
    parts = [_pg_clause(key, value, column) for key, value in where.items()]
    return _join(parts, "AND")


def _join(parts: list[tuple[str, list[Any]]], op: str) -> tuple[str, list[Any]]:
    if len(parts) == 1:
        return parts[0]
    sql = f" {op} ".join(f"({part})" for part, _ in parts)
    params = [param for _, part_params in parts for param in part_params]
    return sql, params


def _contains(column: str, key: str, value: Any) -> tuple[str, list[Any]]:
    return f"{column} @> %s::jsonb", [json.dumps({key: value})]


def _pg_clause(key: str, value: Any, column: str) -> tuple[str, list[Any]]:
    if key in ("$and", "$or"):
        if isinstance(value, str) or not isinstance(value, Sequence) or not value:
            raise UnsupportedFilterError(f"{key} needs a non-empty list: {value!r}")
        return _join(
            [to_pg_filter(clause, column) for clause in value], key[1:].upper()
        )
    if key.startswith("$"):
        raise UnsupportedFilterError(f"PostgreSQL filters do not support {key!r}.")
    if _is_scalar(value):
        return _contains(column, key, value)
    if not isinstance(value, Mapping) or not value:
        raise UnsupportedFilterError(f"Unsupported condition on {key!r}: {value!r}")
    return _join(
        [_pg_condition(key, op, operand, column) for op, operand in value.items()],
        "AND",
    )


def _pg_condition(
    key: str, op: str, operand: Any, column: str
) -> tuple[str, list[Any]]:
    if op == "$eq" and _is_scalar(operand):
        return _contains(column, key, operand)
    if op == "$ne" and _is_scalar(operand):
        sql, params = _contains(column, key, operand)
        return f"jsonb_exists({column}, %s) AND NOT {sql}", [key, *params]
    if op in ("$in", "$nin") and _is_scalar_list(operand):
        if not operand:
            # Nothing is in an empty list; every record with the field is outside it.
            return (
                ("FALSE", []) if op == "$in" else (f"jsonb_exists({column}, %s)", [key])
            )
        any_of = _join([_contains(column, key, item) for item in operand], "OR")
        if op == "$in":
            return any_of
        return f"jsonb_exists({column}, %s) AND NOT ({any_of[0]})", [key, *any_of[1]]
    if op in _COMPARISONS and _is_number(operand):
        if isinstance(operand, float) and not math.isfinite(operand):
            raise UnsupportedFilterError(f"Cannot compare {key!r} with {operand!r}.")
        return (
            f"jsonb_typeof({column} -> %s) = 'number' "
            f"AND ({column} ->> %s)::float8 {_COMPARISONS[op]} %s",
            [key, key, float(operand)],
        )
    raise UnsupportedFilterError(
        f"PostgreSQL filters do not support {op!r} on {key!r}: {operand!r}"
    )


def _is_scalar_list(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and all(_is_scalar(item) for item in value)
    )


# --- Metadata --------------------------------------------------------------------


def to_pg_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Make a record's metadata JSON-safe for a jsonb column.

    ``None`` stays ``null``. Non-finite floats (which JSON cannot hold) become
    ``null``, enums their value, lists stay lists, and other values strings.
    """
    converted: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, Enum):
            value = value.value
        if value is None or isinstance(value, (str, bool, int)):
            converted[key] = value
        elif isinstance(value, float):
            converted[key] = value if math.isfinite(value) else None
        elif isinstance(value, (np.integer, np.floating, np.bool_)):
            converted[key] = to_pg_metadata({key: value.item()})[key]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            converted[key] = [to_pg_metadata({"v": item})["v"] for item in value]
        else:
            converted[key] = str(value)
    return converted


# --- Connections -----------------------------------------------------------------


def import_psycopg() -> Any:
    """Import ``psycopg`` and check ``pgvector`` is installed, with a clear error.

    Raises:
        PgVectorConfigError: If either package is missing.
    """
    try:
        import pgvector.psycopg  # noqa: F401
        import psycopg
    except ImportError as e:
        raise PgVectorConfigError(PSYCOPG_MISSING_MESSAGE) from e
    return psycopg


def connect(dsn: str, register: bool = True) -> Any:
    """Open an autocommit connection, with pgvector's types registered.

    Args:
        dsn: libpq connection string or URI. Empty uses libpq's defaults and the
            ``PG*`` environment variables.
        register: Register the ``vector`` / ``halfvec`` types, which needs the
            extension to be installed in the database.

    Raises:
        PgVectorConfigError: If ``register`` and the database has no pgvector.
    """
    psycopg = import_psycopg()
    from pgvector.psycopg import register_vector

    from aatm.vector_db import pg_connect_timeout

    timeout = pg_connect_timeout(dsn)
    kwargs = {} if timeout is None else {"connect_timeout": timeout}
    try:
        conn = psycopg.connect(dsn, autocommit=True, **kwargs)
    except psycopg.OperationalError as e:
        raise PgVectorConnectionError(connection_failure_message(dsn, e)) from e
    if register:
        try:
            register_vector(conn)
        except psycopg.ProgrammingError as e:
            conn.close()
            raise PgVectorConfigError(
                "The database has no pgvector extension. Run "
                "`CREATE EXTENSION vector;` in it (it needs pgvector installed on "
                "the server, e.g. the pgvector/pgvector Docker image)."
            ) from e
    return conn


def connection_failure_message(dsn: str, error: BaseException) -> str:
    """Explain a failed connection: where AATM tried, why, and how to fix it."""
    from aatm.vector_db import (
        PG_DOCKER_COMMAND,
        PG_DSN_ENV_VAR,
        PG_EXAMPLE_DSN,
        pg_target,
    )

    reason = " ".join(str(error).split()) or type(error).__name__
    lines = [f"Could not connect to PostgreSQL at {pg_target(dsn)}: {reason}"]
    if not dsn:
        lines.append(
            f"{PG_DSN_ENV_VAR} is not set, so libpq's defaults and the PG* "
            f"variables apply. Set it (in the environment or .env), e.g. "
            f"{PG_DSN_ENV_VAR}={PG_EXAMPLE_DSN}"
        )
    else:
        lines.append(f"Check {PG_DSN_ENV_VAR} and that the server is running.")
    lines.append(
        "To run PostgreSQL with pgvector on this machine, with Docker:\n  "
        + PG_DOCKER_COMMAND
    )
    return "\n".join(lines)


def pgvector_version(conn: Any) -> tuple[int, ...] | None:
    """Return the installed pgvector version, or ``None`` if it is not installed."""
    row = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    if row is None:
        return None
    return tuple(int(part) for part in re.findall(r"\d+", row[0])[:3])


def _index_kind(dimension: int) -> str | None:
    """``vector`` or ``halfvec``: how the HNSW index holds the embeddings, or ``None``."""
    if dimension <= HNSW_MAX_VECTOR_DIMENSIONS:
        return "vector"
    if dimension <= HNSW_MAX_HALFVEC_DIMENSIONS:
        return "halfvec"
    return None


def _ensure_metadata_table(conn: Any) -> None:
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {_ident(INDEX_METADATA_TABLE)} ("
        "name text PRIMARY KEY, dimension integer NOT NULL, "
        "metadata jsonb NOT NULL DEFAULT '{}'::jsonb)"
    )


def pg_table_exists(dsn: str, name: str) -> bool:
    """Return whether the index table ``name`` exists."""
    with connect(dsn, register=False) as conn:
        row = conn.execute("SELECT to_regclass(%s)", (_ident(name),)).fetchone()
        return row is not None and row[0] is not None


def pg_table_count(dsn: str, name: str) -> int | None:
    """Return about how many rows an index table holds, or ``None`` if it does not exist.

    Counting millions of rows takes a while, so a table that has been analysed
    (`create_pg_indexes` does it) reports PostgreSQL's estimate. An empty
    table always reports 0.
    """
    with connect(dsn, register=False) as conn:
        row = conn.execute(
            "SELECT reltuples FROM pg_class WHERE oid = to_regclass(%s)",
            (_ident(name),),
        ).fetchone()
        if row is None:
            return None
        if conn.execute(f"SELECT NOT EXISTS (SELECT 1 FROM {_ident(name)})").fetchone()[
            0
        ]:
            return 0
        if row[0] is not None and row[0] > 0:
            return int(row[0])
        return int(conn.execute(f"SELECT count(*) FROM {_ident(name)}").fetchone()[0])


def create_pg_table(
    dsn: str, name: str, dimension: int, metadata: Mapping[str, Any] | None = None
) -> None:
    """Create an empty index table, without its search indexes.

    Installs the ``vector`` extension if the database lacks it (which needs the
    right to create extensions). Call `create_pg_indexes` after loading the rows.

    Args:
        dsn: Connection string.
        name: Table name, from `pg_table_name`.
        dimension: Embedding dimension.
        metadata: AATM metadata of the index, e.g. its precision.

    Raises:
        PgVectorConfigError: If pgvector cannot be installed.
    """
    if dimension < 1:
        raise ValueError(f"dimension must be positive: {dimension}")
    psycopg = import_psycopg()
    from psycopg.types.json import Jsonb

    with connect(dsn, register=False) as conn:
        if pgvector_version(conn) is None:
            try:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except psycopg.Error as e:
                raise PgVectorConfigError(
                    "Could not install the pgvector extension "
                    f"({type(e).__name__}: {e}). Install pgvector on the server "
                    "(e.g. use the pgvector/pgvector Docker image) or ask an "
                    "administrator to run `CREATE EXTENSION vector;`."
                ) from e
        with conn.transaction():
            _ensure_metadata_table(conn)
            conn.execute(
                f"CREATE TABLE {_ident(name)} ("
                "id text PRIMARY KEY, "
                f"embedding vector({int(dimension)}) NOT NULL, "
                "document text, "
                "metadata jsonb NOT NULL DEFAULT '{}'::jsonb)"
            )
            conn.execute(
                f"INSERT INTO {_ident(INDEX_METADATA_TABLE)} (name, dimension, metadata) "
                "VALUES (%s, %s, %s) ON CONFLICT (name) DO UPDATE "
                "SET dimension = EXCLUDED.dimension, metadata = EXCLUDED.metadata",
                (name, int(dimension), Jsonb(to_pg_metadata(dict(metadata or {})))),
            )


def drop_pg_table(dsn: str, name: str) -> None:
    """Delete an index table and its metadata."""
    with connect(dsn, register=False) as conn, conn.transaction():
        conn.execute(f"DROP TABLE IF EXISTS {_ident(name)}")
        row = conn.execute(
            "SELECT to_regclass(%s)", (_ident(INDEX_METADATA_TABLE),)
        ).fetchone()
        if row and row[0]:
            conn.execute(
                f"DELETE FROM {_ident(INDEX_METADATA_TABLE)} WHERE name = %s", (name,)
            )


IndexProgress = Callable[[str, str, int, int], None]
"""Reports an index build: ``(step, phase, done, total)``.

``step`` is ``"HNSW index"``, ``"metadata index"`` or ``"statistics"``. ``done``
and ``total`` count blocks or rows, as PostgreSQL reports them; ``total`` is 0
while it is unknown. Each step ends with ``phase == "done"``.
"""

PROGRESS_INTERVAL = 1.0
"""Seconds between two readings of ``pg_stat_progress_create_index``."""


def hnsw_memory_estimate(rows: int, dimension: int, m: int = DEFAULT_HNSW_M) -> int:
    """Estimate the memory, in bytes, an HNSW build needs to hold its whole graph.

    One vector per row (4 bytes per value, or 2 as ``halfvec``) plus its
    neighbour lists. On 2.46 million 768-dimension rows the index came to 9.2 GB
    on disk with ``m`` 16 and 24 alike; this estimates 9.4 GB for ``m=16`` and
    10.5 GB for ``m=24``, erring on the safe side.
    """
    element = 4 if _index_kind(dimension) == "vector" else 2
    return int(rows) * (int(dimension) * element + 64 * int(m))


def create_pg_indexes(
    dsn: str,
    name: str,
    m: int = DEFAULT_HNSW_M,
    ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
    maintenance_work_mem: str | None = None,
    workers: int | None = None,
    progress: IndexProgress | None = None,
) -> None:
    """Create the HNSW and GIN indexes of an index table, if they are missing.

    Building them once after the bulk load is much faster than keeping them
    up to date row by row. When the HNSW graph does not fit in
    ``maintenance_work_mem``, PostgreSQL builds most of it on disk, which is
    much slower; a warning then says how much memory the build would need.

    Args:
        dsn: Connection string.
        name: Table name.
        m: HNSW links per node. Defaults to `DEFAULT_HNSW_M` (24).
        ef_construction: HNSW candidate list size while building. Defaults to
            `DEFAULT_HNSW_EF_CONSTRUCTION` (128).
        maintenance_work_mem: Memory for the build, e.g. ``8GB``. Defaults to
            the server's setting (PostgreSQL's default is 64 MB). A parallel
            build keeps it in shared memory, so the server needs at least as
            much (``--shm-size`` for Docker).
        workers: Parallel workers for the build
            (``max_parallel_maintenance_workers``). Defaults to the server's
            setting (2), and is capped by its ``max_parallel_workers``.
        progress: Called about every second while an index builds; see
            `IndexProgress`.
    """
    with connect(dsn) as conn:
        dimension = _dimension(conn, name)
        kind = _index_kind(dimension)
        if maintenance_work_mem:
            conn.execute(
                "SELECT set_config('maintenance_work_mem', %s, false)",
                (maintenance_work_mem,),
            )
        if workers is not None:
            conn.execute(
                "SELECT set_config('max_parallel_maintenance_workers', %s, false)",
                (str(int(workers)),),
            )
        table = _ident(name)
        hnsw = _index_name(name, "hnsw")
        started = time.perf_counter()
        if kind is None:
            logger.warning(
                f"{dimension}-dimensional embeddings are too large for an HNSW "
                f"index (at most {HNSW_MAX_HALFVEC_DIMENSIONS}); searches on "
                f"'{name}' will be exact and slow on large indexes."
            )
        else:
            if not _index_exists(conn, hnsw):
                _warn_if_memory_is_short(conn, name, dimension, m)
            target = (
                "embedding vector_l2_ops"
                if kind == "vector"
                else f"(embedding::halfvec({dimension})) halfvec_l2_ops"
            )
            _execute_watched(
                conn,
                dsn,
                f"CREATE INDEX IF NOT EXISTS {_ident(hnsw)} "
                f"ON {table} USING hnsw ({target}) "
                f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)})",
                "HNSW index",
                progress,
            )
        _execute_watched(
            conn,
            dsn,
            f"CREATE INDEX IF NOT EXISTS {_ident(_index_name(name, 'meta'))} "
            f"ON {table} USING gin (metadata jsonb_path_ops)",
            "metadata index",
            progress,
        )
        _execute_watched(conn, dsn, f"ANALYZE {table}", "statistics", progress)
        logger.info(
            f"Indexed '{name}' in {time.perf_counter() - started:.1f}s"
            + ("" if kind is None else f" (HNSW on {kind})")
        )


def _index_exists(conn: Any, index: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s)", (_ident(index),)).fetchone()
    return row is not None and row[0] is not None


def _warn_if_memory_is_short(conn: Any, name: str, dimension: int, m: int) -> None:
    """Warn when the HNSW graph will not fit in ``maintenance_work_mem``."""
    rows, memory = conn.execute(
        "SELECT (SELECT GREATEST(reltuples, 0)::bigint FROM pg_class "
        "WHERE oid = to_regclass(%s)), "
        "pg_size_bytes(current_setting('maintenance_work_mem'))",
        (_ident(name),),
    ).fetchone()
    if not rows:
        rows = conn.execute(f"SELECT count(*) FROM {_ident(name)}").fetchone()[0]
    needed = hnsw_memory_estimate(int(rows or 0), dimension, m)
    if needed > int(memory):
        logger.warning(
            f"The HNSW index of '{name}' needs about {_gb(needed)} to build in "
            f"memory, but maintenance_work_mem is {_gb(int(memory))}, so most of "
            "it is built on disk, which is much slower. Set "
            "AATM_PG_MAINTENANCE_WORK_MEM (e.g. to "
            f"{max(1, -(-needed // 2**30))}GB) if the server has the memory, and "
            "enough shared memory for a parallel build (Docker: --shm-size)."
        )


def _gb(size: int) -> str:
    return f"{size / 2**30:.1f} GB" if size >= 2**30 else f"{size // 2**20} MB"


def _execute_watched(
    conn: Any, dsn: str, sql: str, step: str, progress: IndexProgress | None
) -> None:
    """Run a statement, reporting its ``pg_stat_progress_create_index`` row meanwhile.

    The statement runs in a thread on ``conn``; this thread reads the progress
    on a second connection. Ctrl+C cancels the statement on the server.
    """
    if progress is None:
        conn.execute(sql)
        return
    pid = conn.info.backend_pid
    errors: list[BaseException] = []

    def work() -> None:
        try:
            conn.execute(sql)
        except BaseException as e:  # re-raised below, in the calling thread
            errors.append(e)

    progress(step, "starting", 0, 0)
    worker = threading.Thread(target=work, name=f"aatm-{step}", daemon=True)
    worker.start()
    try:
        with connect(dsn, register=False) as watcher:
            while worker.is_alive():
                worker.join(PROGRESS_INTERVAL)
                if not worker.is_alive():
                    break
                row = watcher.execute(
                    "SELECT phase, blocks_done, blocks_total, tuples_done, "
                    "tuples_total FROM pg_stat_progress_create_index WHERE pid = %s",
                    (pid,),
                ).fetchone()
                if row is None:
                    continue
                phase, blocks_done, blocks_total, tuples_done, tuples_total = row
                if blocks_total:
                    progress(step, phase, int(blocks_done), int(blocks_total))
                elif tuples_total:
                    progress(step, phase, int(tuples_done), int(tuples_total))
                else:
                    progress(step, phase, int(tuples_done or 0), 0)
    except BaseException:
        # e.g. Ctrl+C: stop the build on the server too, not just here.
        cancel = getattr(conn, "cancel_safe", None) or conn.cancel
        try:
            cancel()
        finally:
            worker.join()
        raise
    worker.join()
    if errors:
        raise errors[0]
    progress(step, "done", 1, 1)


def _index_name(table: str, suffix: str) -> str:
    name = f"{table}_{suffix}"
    if len(name) > PG_IDENTIFIER_MAX_LENGTH:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        name = (
            f"{table[: PG_IDENTIFIER_MAX_LENGTH - len(suffix) - 10]}_{digest}_{suffix}"
        )
    return name


def _dimension(conn: Any, name: str) -> int:
    row = conn.execute(
        f"SELECT dimension FROM {_ident(INDEX_METADATA_TABLE)} WHERE name = %s",
        (name,),
    ).fetchone()
    if row is None:
        raise PgVectorConfigError(f"No AATM index table '{name}' in this database.")
    return int(row[0])


# --- The store -------------------------------------------------------------------


class PgVectorStore(VectorStore):
    """One embedding model's index in a PostgreSQL table. See the module docstring.

    Each thread gets its own connection, so the build's lookups and writes,
    which run in two threads of the writer process, do not wait for each other.
    """

    filters_during_search = True

    def __init__(
        self,
        dsn: str,
        name: str,
        ef_search: int = DEFAULT_EF_SEARCH,
        iterative_scan: str = DEFAULT_ITERATIVE_SCAN,
        max_scan_tuples: int | None = None,
        location: str | None = None,
        exact: bool = False,
    ):
        """Open the index table ``name``.

        Args:
            dsn: Connection string.
            name: Table name, from `pg_table_name`.
            ef_search: ``hnsw.ef_search``: candidates kept during the search.
                Each query uses at least its ``top_k``, up to 1,000.
            iterative_scan: ``hnsw.iterative_scan``: ``off``,
                ``strict_order`` or ``relaxed_order``. Ignored before pgvector
                0.8.
            max_scan_tuples: ``hnsw.max_scan_tuples``: how far an iterative
                scan goes. Defaults to the server's setting (20,000).
            location: Server and database, without credentials, for the
                fingerprint.
            exact: Search exactly, without the HNSW index: every matching row
                is compared with the query. Slow on large tables; a reference
                for recall benchmarks.

        Raises:
            PgVectorConfigError: If the table, or pgvector, is missing.
        """
        if not 1 <= ef_search <= HNSW_MAX_EF_SEARCH:
            raise ValueError(f"ef_search must be between 1 and {HNSW_MAX_EF_SEARCH}.")
        if iterative_scan not in ITERATIVE_SCAN_MODES:
            raise ValueError(
                f"iterative_scan must be one of {', '.join(ITERATIVE_SCAN_MODES)}."
            )
        self.dsn = dsn
        self.name = name
        self.table = _ident(name)
        self.ef_search = ef_search
        self.max_scan_tuples = max_scan_tuples
        self.location = location
        self.exact = exact
        self._local = threading.local()

        conn = self._conn()
        self.version = pgvector_version(conn)
        self.dimension = _dimension(conn, name)
        self.index_kind = _index_kind(self.dimension)
        if iterative_scan != "off" and (
            self.version is None or self.version < ITERATIVE_SCAN_MIN_VERSION
        ):
            version = ".".join(str(part) for part in self.version or ())
            logger.warning(
                f"pgvector {version or '?'} has no iterative index scans (0.8+); "
                "filtered searches may return fewer than top_k results."
            )
            iterative_scan = "off"
        self.iterative_scan = iterative_scan

    def _conn(self) -> Any:
        conn = getattr(self._local, "conn", None)
        if conn is None or conn.closed:
            conn = connect(self.dsn)
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close this thread's connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _order_by(self) -> str:
        if self.index_kind == "halfvec" and not self.exact:
            cast = f"halfvec({self.dimension})"
            return f"embedding::{cast} <-> %(q)s::{cast}"
        return "embedding <-> %(q)s"

    def query(
        self,
        embeddings: Sequence[Sequence[float]],
        top_k: int,
        where: Mapping[str, Any] | None = None,
    ) -> list[list[Hit]]:
        if top_k < 1:
            return [[] for _ in embeddings]
        condition, filter_params = to_pg_filter(where) if where else ("TRUE", [])
        # psycopg takes either named or positional parameters, so the filter's
        # positional ones become named.
        named = {f"f{i}": value for i, value in enumerate(filter_params)}
        condition = _name_placeholders(condition, len(filter_params))
        sql = (
            "WITH hits AS MATERIALIZED ("
            f"SELECT metadata, embedding <-> %(q)s AS distance FROM {self.table} "
            f"WHERE {condition} ORDER BY {self._order_by()} LIMIT %(k)s"
            ") SELECT distance, metadata FROM hits ORDER BY distance"
        )
        conn = self._conn()
        results: list[list[Hit]] = []
        with conn.transaction():
            self._set_search_options(conn, top_k)
            for embedding in embeddings:
                vector = np.asarray(embedding, dtype=np.float32)
                rows = conn.execute(
                    sql, {"q": vector, "k": int(top_k), **named}
                ).fetchall()
                results.append(
                    [
                        (float(distance) ** 2, dict(metadata))
                        for distance, metadata in rows
                    ]
                )
        return results

    def _set_search_options(self, conn: Any, top_k: int) -> None:
        if self.exact:
            # Without index scans the planner reads the matching rows and sorts
            # them by distance: an exact search.
            conn.execute("SELECT set_config('enable_indexscan', 'off', true)")
            return
        settings = {
            "hnsw.ef_search": str(min(HNSW_MAX_EF_SEARCH, max(self.ef_search, top_k)))
        }
        if self.version is not None and self.version >= ITERATIVE_SCAN_MIN_VERSION:
            settings["hnsw.iterative_scan"] = self.iterative_scan
            if self.max_scan_tuples is not None:
                settings["hnsw.max_scan_tuples"] = str(int(self.max_scan_tuples))
        for key, value in settings.items():
            conn.execute("SELECT set_config(%s, %s, true)", (key, value))

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        if not ids:
            return set()
        rows = self._conn().execute(
            f"SELECT id FROM {self.table} WHERE id = ANY(%s)", (list(ids),)
        )
        return {row[0] for row in rows}

    def write(
        self,
        ids: Sequence[str],
        embeddings: Any,
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
        attempt: int = 0,
    ) -> None:
        """Copy a batch into a staging table, then insert it.

        The first attempt skips IDs that are already stored, like ChromaDB's
        ``add``; retries replace them, so a retry never duplicates rows.
        """
        from psycopg.types.json import Jsonb

        if not ids:
            return
        vectors = np.asarray(embeddings, dtype=np.float32)
        conflict = (
            "DO NOTHING"
            if attempt == 0
            else "DO UPDATE SET embedding = EXCLUDED.embedding, "
            "document = EXCLUDED.document, metadata = EXCLUDED.metadata"
        )
        conn = self._conn()
        with conn.transaction():
            conn.execute(
                "CREATE TEMP TABLE IF NOT EXISTS aatm_staging "
                "(id text, embedding vector, document text, metadata jsonb) "
                "ON COMMIT DELETE ROWS"
            )
            with conn.cursor() as cur:
                with cur.copy(
                    "COPY aatm_staging (id, embedding, document, metadata) "
                    "FROM STDIN (FORMAT BINARY)"
                ) as copy:
                    copy.set_types(["text", "vector", "text", "jsonb"])
                    for identifier, vector, document, metadata in zip(
                        ids, vectors, documents, metadatas
                    ):
                        copy.write_row(
                            (
                                str(identifier),
                                vector,
                                document,
                                Jsonb(to_pg_metadata(metadata or {})),
                            )
                        )
                # DISTINCT ON keeps one row per ID if a batch repeats one.
                cur.execute(
                    f"INSERT INTO {self.table} (id, embedding, document, metadata) "
                    "SELECT DISTINCT ON (id) id, embedding, document, metadata "
                    f"FROM aatm_staging ON CONFLICT (id) {conflict}"
                )

    def count(self) -> int:
        row = self._conn().execute(f"SELECT count(*) FROM {self.table}").fetchone()
        return int(row[0])

    def index_metadata(self) -> dict[str, Any]:
        row = (
            self._conn()
            .execute(
                f"SELECT metadata FROM {_ident(INDEX_METADATA_TABLE)} WHERE name = %s",
                (self.name,),
            )
            .fetchone()
        )
        return dict(row[0]) if row else {}

    def fingerprint(self) -> dict[str, Any]:
        return {
            "pg_table": self.name,
            "location": self.location,
            "dimension": self.dimension,
            "count": self.count(),
        }


def _name_placeholders(sql: str, n: int) -> str:
    """Turn ``n`` positional ``%s`` placeholders into ``%(f0)s``, ``%(f1)s``, ..."""
    parts = sql.split("%s")
    if len(parts) != n + 1:
        raise AssertionError("placeholder count does not match the parameters")
    out = parts[0]
    for i, part in enumerate(parts[1:]):
        out += f"%(f{i})s{part}"
    return out
