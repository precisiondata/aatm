"""Choose which vector database holds the expression index, and where.

AATM stores its expression index in a vector database. Four are supported:

- ``local`` (default): a persistent ChromaDB database on this machine, one per
  embedding model, under ``.aatm/chroma_vector_dbs/<model>``.
- ``cloud``: a Chroma Cloud database, so the index can be built once and
  shared, and machines without the disk space or GPU for a local index can
  still run retrieval.
- ``pinecone``: one Pinecone serverless index per embedding model. Needs the
  ``aatm[pinecone]`` extra.
- ``pgvector``: PostgreSQL with the pgvector extension, one table per embedding
  model (see `aatm.pgvector_store`). Needs the ``aatm[pgvector]`` extra.

Configuration comes from environment variables (a ``.env`` file works too):

- ``AATM_VECTOR_DB``: ``local`` (default), ``cloud``, ``pinecone`` or ``pgvector``.
- ``CHROMA_API_KEY``: Chroma Cloud API key. Required in cloud mode.
- ``CHROMA_TENANT`` and ``CHROMA_DATABASE``: optional. Chroma Cloud infers them
  when the API key is scoped to a single database.
- ``CHROMA_CLOUD_HOST``: optional, defaults to ``api.trychroma.com``.
- ``PINECONE_API_KEY``: Pinecone API key. Required in pinecone mode.
- ``PINECONE_INDEX_PREFIX``: optional, defaults to ``aatm-expressions``. Each
  model's index is named ``<prefix>-<model>`` (see `pinecone_index_name`).
- ``PINECONE_CLOUD`` and ``PINECONE_REGION``: where new serverless indexes are
  created. Default ``aws`` and ``us-east-1``.
- ``AATM_PG_DSN``: PostgreSQL connection string, e.g.
  ``postgresql://postgres@localhost:5432/aatm``. If unset, libpq's defaults and
  the standard ``PGHOST``, ``PGPORT``, ``PGDATABASE``, ``PGUSER`` and
  ``PGPASSWORD`` variables apply. Prefer ``PGPASSWORD`` or a ``.pgpass`` file to
  a password in the connection string.
- ``AATM_PG_TABLE_PREFIX``: optional, defaults to ``aatm_expressions``. Each
  model's table is named ``<prefix>_<model>`` (see
  `aatm.pgvector_store.pg_table_name`).
- ``AATM_PG_EF_SEARCH``: optional HNSW search breadth (``hnsw.ef_search``),
  1 to 1000, default 100. Higher finds more true neighbours, more slowly.
- ``AATM_PG_ITERATIVE_SCAN``: ``relaxed_order`` (default), ``strict_order`` or
  ``off``. How a filtered search keeps scanning until it has ``top_k`` matches.
- ``AATM_PG_MAINTENANCE_WORK_MEM``: optional memory for building the HNSW index,
  e.g. ``4GB``. Defaults to the server's setting.

In cloud mode one database holds the indexes of every embedding model, so each
model gets its own collection (see `ChromaRetrieverModelSpec.cloud_collection_name`).
A Pinecone index has a fixed dimension, so each model gets its own index, and
in PostgreSQL its own table.

Every place that opens a vector database goes through `connection_from_env`
(or the ``from_env`` of `ChromaConnection`, `PineconeConnection` and
`PgVectorConnection`), so switching
backends is configuration only.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from aatm.logs import get_logger

logger = get_logger(__name__)

VectorDbMode = Literal["local", "cloud", "pinecone", "pgvector"]
"""Which vector database to use, and where it lives."""

VECTOR_DB_MODES: tuple[VectorDbMode, ...] = (
    "local",
    "cloud",
    "pinecone",
    "pgvector",
)

VECTOR_DB_ENV_VAR = "AATM_VECTOR_DB"
CHROMA_API_KEY_ENV_VAR = "CHROMA_API_KEY"
CHROMA_TENANT_ENV_VAR = "CHROMA_TENANT"
CHROMA_DATABASE_ENV_VAR = "CHROMA_DATABASE"
CHROMA_CLOUD_HOST_ENV_VAR = "CHROMA_CLOUD_HOST"

PINECONE_API_KEY_ENV_VAR = "PINECONE_API_KEY"
PINECONE_INDEX_PREFIX_ENV_VAR = "PINECONE_INDEX_PREFIX"
PINECONE_CLOUD_ENV_VAR = "PINECONE_CLOUD"
PINECONE_REGION_ENV_VAR = "PINECONE_REGION"

PG_DSN_ENV_VAR = "AATM_PG_DSN"
PG_TABLE_PREFIX_ENV_VAR = "AATM_PG_TABLE_PREFIX"
PG_EF_SEARCH_ENV_VAR = "AATM_PG_EF_SEARCH"
PG_ITERATIVE_SCAN_ENV_VAR = "AATM_PG_ITERATIVE_SCAN"
PG_MAINTENANCE_WORK_MEM_ENV_VAR = "AATM_PG_MAINTENANCE_WORK_MEM"
PG_ITERATIVE_SCAN_MODES = ("off", "strict_order", "relaxed_order")
"""Values of pgvector's ``hnsw.iterative_scan``."""

DEFAULT_CLOUD_HOST = "api.trychroma.com"
DEFAULT_PINECONE_INDEX_PREFIX = "aatm-expressions"
DEFAULT_PINECONE_CLOUD = "aws"
DEFAULT_PINECONE_REGION = "us-east-1"
DEFAULT_PG_TABLE_PREFIX = "aatm_expressions"

PINECONE_INDEX_NAME_MAX_LENGTH = 45
"""Pinecone index names have at most 45 characters of ``a-z``, ``0-9`` and ``-``."""

T = TypeVar("T")


class VectorDatabaseConfigError(ValueError):
    """Raised when the vector database configuration is missing or invalid."""


def resolve_vector_db_mode(
    mode: str | None = None, env: Mapping[str, str] | None = None
) -> VectorDbMode:
    """Return the vector database mode to use.

    Args:
        mode: Mode requested by the caller. If ``None``, ``AATM_VECTOR_DB`` is
            used, and otherwise ``local``.
        env: Environment to read. Defaults to `os.environ`.

    Returns:
        ``local``, ``cloud``, ``pinecone`` or ``pgvector``.

    Raises:
        VectorDatabaseConfigError: If the mode is not one of `VECTOR_DB_MODES`.
    """
    env = os.environ if env is None else env
    value = mode if mode is not None else env.get(VECTOR_DB_ENV_VAR) or "local"
    value = value.strip().lower()
    if value not in VECTOR_DB_MODES:
        raise VectorDatabaseConfigError(
            f"Unknown vector database mode '{value}'. "
            f"Use one of: {', '.join(VECTOR_DB_MODES)} "
            f"(set with --vector-db or {VECTOR_DB_ENV_VAR})."
        )
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ChromaConnection:
    """Describe how to open a ChromaDB client, locally or in Chroma Cloud.

    Instances are small and picklable, so they can be sent to the writer
    process. The API key is kept out of ``repr`` so it does not end up in logs.
    """

    mode: VectorDbMode
    """``local`` for a persistent database on disk, ``cloud`` for Chroma Cloud."""

    path: str | None = None
    """Directory of the local database. Only used in local mode."""

    tenant: str | None = None
    """Chroma Cloud tenant. ``None`` lets Chroma Cloud infer it from the API key."""

    database: str | None = None
    """Chroma Cloud database. ``None`` lets Chroma Cloud infer it from the API key."""

    api_key: str | None = field(default=None, repr=False)
    """Chroma Cloud API key. Never logged."""

    cloud_host: str = DEFAULT_CLOUD_HOST
    """Chroma Cloud host name."""

    @property
    def is_cloud(self) -> bool:
        """Whether this connection points at Chroma Cloud."""
        return self.mode == "cloud"

    @classmethod
    def local(cls, path: str) -> ChromaConnection:
        """Return a connection to a local persistent database at ``path``."""
        return cls(mode="local", path=path)

    @classmethod
    def from_env(
        cls,
        local_path: str,
        mode: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ChromaConnection:
        """Build a connection from the environment.

        Args:
            local_path: Directory used in local mode.
            mode: ``local`` or ``cloud``. If ``None``, ``AATM_VECTOR_DB`` decides.
            env: Environment to read. Defaults to `os.environ`.

        Returns:
            The connection for the selected mode.

        Raises:
            VectorDatabaseConfigError: If the mode is invalid or not ChromaDB
                (use `connection_from_env` for any backend), or cloud mode is
                selected without ``CHROMA_API_KEY``.
        """
        env = os.environ if env is None else env
        resolved = resolve_vector_db_mode(mode, env)
        if resolved in ("pinecone", "pgvector"):
            raise VectorDatabaseConfigError(
                f"The vector database is set to '{resolved}', which is not ChromaDB. "
                "Use aatm.vector_db.connection_from_env to open any backend."
            )
        if resolved == "local":
            return cls.local(local_path)

        api_key = (env.get(CHROMA_API_KEY_ENV_VAR) or "").strip()
        if not api_key:
            raise VectorDatabaseConfigError(
                f"Chroma Cloud needs an API key. Set {CHROMA_API_KEY_ENV_VAR} "
                "in the environment or in a .env file."
            )
        return cls(
            mode="cloud",
            tenant=(env.get(CHROMA_TENANT_ENV_VAR) or "").strip() or None,
            database=(env.get(CHROMA_DATABASE_ENV_VAR) or "").strip() or None,
            api_key=api_key,
            cloud_host=(env.get(CHROMA_CLOUD_HOST_ENV_VAR) or "").strip()
            or DEFAULT_CLOUD_HOST,
        )

    def describe(self) -> str:
        """Return a human-readable location, without credentials."""
        if not self.is_cloud:
            return f"local database at '{self.path}'"
        parts = [f"host={self.cloud_host}"]
        if self.tenant:
            parts.append(f"tenant={self.tenant}")
        if self.database:
            parts.append(f"database={self.database}")
        return f"Chroma Cloud ({', '.join(parts)})"

    def client(self) -> Any:
        """Open a ChromaDB client for this connection.

        Returns:
            A ``chromadb.ClientAPI``.

        Raises:
            VectorDatabaseConfigError: If a local connection has no path.
        """
        import chromadb

        if not self.is_cloud:
            if self.path is None:
                raise VectorDatabaseConfigError("A local connection needs a path.")
            return chromadb.PersistentClient(path=self.path)

        return chromadb.CloudClient(
            tenant=self.tenant,
            database=self.database,
            api_key=self.api_key,
            cloud_host=self.cloud_host,
        )

    def open_store(self, name: str) -> Any:
        """Open an existing collection as a `aatm.vector_stores.ChromaVectorStore`.

        Raises:
            chromadb.errors.NotFoundError: If the collection does not exist.
        """
        from aatm.vector_stores import ChromaVectorStore

        return ChromaVectorStore(self.client().get_collection(name))


PINECONE_MISSING_MESSAGE = (
    "Pinecone support needs the 'pinecone' package. Install it with "
    "`pip install 'aatm[pinecone]'` (or `uv sync --extra pinecone`)."
)


def import_pinecone() -> Any:
    """Import the ``pinecone`` package, with a clear error if it is missing.

    Raises:
        VectorDatabaseConfigError: If ``pinecone`` is not installed.
    """
    try:
        import pinecone
    except ImportError as e:
        raise VectorDatabaseConfigError(PINECONE_MISSING_MESSAGE) from e
    return pinecone


_INVALID_INDEX_CHARS = re.compile(r"[^a-z0-9-]+")


def pinecone_index_name(prefix: str, model_name: str) -> str:
    """Return a valid Pinecone index name for one embedding model's index.

    Pinecone index names have at most 45 characters of lowercase letters,
    digits and ``-``, and cannot start or end with ``-``. Other characters
    become ``-``. A name that is too long is shortened and gets a short hash of
    the full name, so two models never share an index.

    Args:
        prefix: Common prefix, e.g. ``aatm-expressions``. May be empty.
        model_name: Registry name of the embedding model, e.g. ``qwen3-06B``.

    Returns:
        The index name, e.g. ``aatm-expressions-qwen3-06b``.

    Raises:
        VectorDatabaseConfigError: If nothing valid is left of the name.
    """
    raw = f"{prefix}-{model_name}" if prefix else model_name
    name = _INVALID_INDEX_CHARS.sub("-", raw.lower())
    name = re.sub(r"-{2,}", "-", name).strip("-")
    if not name:
        raise VectorDatabaseConfigError(
            f"Cannot build a Pinecone index name from {raw!r}."
        )
    if len(name) > PINECONE_INDEX_NAME_MAX_LENGTH:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6]
        keep = PINECONE_INDEX_NAME_MAX_LENGTH - len(digest) - 1
        name = f"{name[:keep].rstrip('-')}-{digest}"
    return name


@dataclass(frozen=True, slots=True)
class PineconeConnection:
    """Describe how to open a Pinecone client and where to create indexes.

    Instances are small and picklable, so they can be sent to the writer
    process. The API key is kept out of ``repr`` so it does not end up in logs.
    """

    api_key: str = field(repr=False)
    """Pinecone API key. Never logged."""

    index_prefix: str = DEFAULT_PINECONE_INDEX_PREFIX
    """Prefix of the per-model index names (see `pinecone_index_name`)."""

    cloud: str = DEFAULT_PINECONE_CLOUD
    """Cloud provider for new serverless indexes."""

    region: str = DEFAULT_PINECONE_REGION
    """Region for new serverless indexes."""

    mode: VectorDbMode = "pinecone"
    """Always ``pinecone``. Mirrors `ChromaConnection.mode`."""

    @property
    def is_cloud(self) -> bool:
        """Always ``True``: Pinecone is a remote service."""
        return True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> PineconeConnection:
        """Build a connection from the ``PINECONE_*`` environment variables.

        Raises:
            VectorDatabaseConfigError: If ``PINECONE_API_KEY`` is not set.
        """
        env = os.environ if env is None else env
        api_key = (env.get(PINECONE_API_KEY_ENV_VAR) or "").strip()
        if not api_key:
            raise VectorDatabaseConfigError(
                f"Pinecone needs an API key. Set {PINECONE_API_KEY_ENV_VAR} "
                "in the environment or in a .env file."
            )

        def setting(name: str, default: str) -> str:
            return (env.get(name) or "").strip() or default

        return cls(
            api_key=api_key,
            index_prefix=setting(
                PINECONE_INDEX_PREFIX_ENV_VAR, DEFAULT_PINECONE_INDEX_PREFIX
            ),
            cloud=setting(PINECONE_CLOUD_ENV_VAR, DEFAULT_PINECONE_CLOUD),
            region=setting(PINECONE_REGION_ENV_VAR, DEFAULT_PINECONE_REGION),
        )

    def index_name(self, model_name: str) -> str:
        """Return the name of one embedding model's index."""
        return pinecone_index_name(self.index_prefix, model_name)

    def describe(self) -> str:
        """Return a human-readable location, without credentials."""
        return (
            f"Pinecone (indexes '{self.index_prefix}-*', "
            f"created in {self.cloud}/{self.region})"
        )

    def client(self) -> Any:
        """Open a Pinecone client.

        Returns:
            A ``pinecone.Pinecone`` client.

        Raises:
            VectorDatabaseConfigError: If ``pinecone`` is not installed.
        """
        pinecone = import_pinecone()
        return pinecone.Pinecone(api_key=self.api_key)

    def open_store(self, name: str) -> Any:
        """Open an existing index as a `aatm.vector_stores.PineconeVectorStore`."""
        from aatm.vector_stores import PineconeVectorStore

        return PineconeVectorStore.open(self.client(), name)


_DSN_URI_PASSWORD = re.compile(r"(://[^:/@\s]*:)[^@\s]*@")
_DSN_KEY_PASSWORD = re.compile(
    r"(password\s*=\s*)('(?:[^'\\]|\\.)*'|\S+)", re.IGNORECASE
)


def redact_dsn(dsn: str) -> str:
    """Return a PostgreSQL connection string with its password hidden."""
    dsn = _DSN_URI_PASSWORD.sub(r"\1***@", dsn)
    return _DSN_KEY_PASSWORD.sub(r"\1***", dsn)


@dataclass(frozen=True, slots=True)
class PgVectorConnection:
    """Describe how to reach a PostgreSQL database with pgvector, and how to search it.

    Instances are small and picklable, so they can be sent to the writer
    process. The connection string may hold a password, so it is kept out of
    ``repr``, and `describe` never shows it.
    """

    dsn: str = field(default="", repr=False)
    """libpq connection string or URI. Empty uses libpq's defaults and ``PG*``."""

    table_prefix: str = DEFAULT_PG_TABLE_PREFIX
    """Prefix of the per-model table names (see `aatm.pgvector_store.pg_table_name`)."""

    ef_search: int = 100
    """``hnsw.ef_search`` for queries."""

    iterative_scan: str = "relaxed_order"
    """``hnsw.iterative_scan`` for filtered queries."""

    maintenance_work_mem: str | None = None
    """``maintenance_work_mem`` while building the HNSW index, e.g. ``4GB``."""

    mode: VectorDbMode = "pgvector"
    """Always ``pgvector``. Mirrors `ChromaConnection.mode`."""

    @property
    def is_cloud(self) -> bool:
        """Always ``False``: not Chroma Cloud. The server may be anywhere."""
        return False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> PgVectorConnection:
        """Build a connection from the ``AATM_PG_*`` environment variables.

        Raises:
            VectorDatabaseConfigError: If a setting is invalid.
        """
        env = os.environ if env is None else env

        def setting(name: str) -> str:
            return (env.get(name) or "").strip()

        ef_search_text = setting(PG_EF_SEARCH_ENV_VAR) or "100"
        try:
            ef_search = int(ef_search_text)
        except ValueError:
            ef_search = 0
        if not 1 <= ef_search <= 1000:
            raise VectorDatabaseConfigError(
                f"{PG_EF_SEARCH_ENV_VAR} must be a whole number from 1 to 1000; "
                f"got {ef_search_text!r}."
            )
        iterative_scan = setting(PG_ITERATIVE_SCAN_ENV_VAR).lower() or "relaxed_order"
        if iterative_scan not in PG_ITERATIVE_SCAN_MODES:
            raise VectorDatabaseConfigError(
                f"{PG_ITERATIVE_SCAN_ENV_VAR} must be one of "
                f"{', '.join(PG_ITERATIVE_SCAN_MODES)}; got {iterative_scan!r}."
            )
        return cls(
            dsn=setting(PG_DSN_ENV_VAR),
            table_prefix=setting(PG_TABLE_PREFIX_ENV_VAR) or DEFAULT_PG_TABLE_PREFIX,
            ef_search=ef_search,
            iterative_scan=iterative_scan,
            maintenance_work_mem=setting(PG_MAINTENANCE_WORK_MEM_ENV_VAR) or None,
        )

    def table_name(self, model_name: str) -> str:
        """Return the name of one embedding model's table."""
        from aatm.pgvector_store import pg_table_name

        return pg_table_name(self.table_prefix, model_name)

    def location(self) -> str:
        """Return the server and database, without credentials."""
        if not self.dsn:
            return "libpq defaults"
        try:
            from psycopg.conninfo import conninfo_to_dict

            info = conninfo_to_dict(self.dsn)
        except Exception:
            return redact_dsn(self.dsn)
        host = info.get("host") or "localhost"
        port = info.get("port") or "5432"
        database = info.get("dbname") or info.get("user") or "?"
        return f"{host}:{port}/{database}"

    def describe(self) -> str:
        """Return a human-readable location, without credentials."""
        return f"PostgreSQL ({self.location()}, tables '{self.table_prefix}_*')"

    def client(self) -> Any:
        """Open an autocommit ``psycopg`` connection to the database.

        Raises:
            VectorDatabaseConfigError: If ``psycopg`` or ``pgvector`` is not
                installed.
        """
        from aatm.pgvector_store import connect

        return connect(self.dsn, register=False)

    def open_store(self, name: str) -> Any:
        """Open an existing table as a `aatm.pgvector_store.PgVectorStore`."""
        from aatm.pgvector_store import PgVectorStore

        return PgVectorStore(
            self.dsn,
            name,
            ef_search=self.ef_search,
            iterative_scan=self.iterative_scan,
            location=self.location(),
        )


VectorDbConnection = ChromaConnection | PineconeConnection | PgVectorConnection
"""A connection to any supported vector database."""


def connection_from_env(
    local_path: str,
    mode: str | None = None,
    env: Mapping[str, str] | None = None,
) -> VectorDbConnection:
    """Build the connection for the configured vector database.

    Args:
        local_path: Directory used in local mode.
        mode: ``local``, ``cloud``, ``pinecone`` or ``pgvector``. If ``None``,
            ``AATM_VECTOR_DB`` decides, and otherwise ``local``.
        env: Environment to read. Defaults to `os.environ`.

    Raises:
        VectorDatabaseConfigError: If the configuration is invalid.
    """
    env = os.environ if env is None else env
    resolved = resolve_vector_db_mode(mode, env)
    if resolved == "pinecone":
        return PineconeConnection.from_env(env)
    if resolved == "pgvector":
        return PgVectorConnection.from_env(env)
    return ChromaConnection.from_env(local_path=local_path, mode=mode, env=env)


def _is_transient_pinecone_error(exc: BaseException) -> bool:
    try:
        from pinecone.errors import exceptions as errors
    except ImportError:
        return False
    transient = (
        errors.RateLimitError,
        errors.ServiceError,
        errors.PineconeConnectionError,
        errors.PineconeTimeoutError,
    )
    return isinstance(exc, transient)


def is_transient_error(exc: BaseException) -> bool:
    """Return whether a vector database call that raised ``exc`` is worth retrying.

    Rate limits, server errors and network failures are transient, for both
    ChromaDB and Pinecone. Everything else (bad input, missing collections or
    indexes, auth and quota errors) is not.
    """
    import httpx
    from chromadb.errors import InternalError, RateLimitError

    if isinstance(exc, (RateLimitError, InternalError, httpx.TransportError)):
        return True
    return _is_transient_pinecone_error(exc)


def call_with_retries(
    func: Callable[[int], T],
    *,
    attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    is_retryable: Callable[[BaseException], bool] = is_transient_error,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``func`` and retry it with exponential backoff on transient errors.

    Args:
        func: Called with the attempt number, starting at 0, so a retry can
            choose an idempotent operation (e.g. ``upsert`` after ``add``).
        attempts: Maximum number of calls.
        base_delay: Delay before the first retry, in seconds. It doubles after
            each attempt, with jitter.
        max_delay: Upper bound for a single delay, in seconds.
        is_retryable: Decides whether an exception is transient.
        sleep: Sleep function, replaceable in tests.

    Returns:
        What ``func`` returned.

    Raises:
        Exception: The last error, when it is not transient or no attempts are left.
    """
    for attempt in range(attempts):
        try:
            return func(attempt)
        except Exception as exc:
            if attempt == attempts - 1 or not is_retryable(exc):
                raise
            delay = min(max_delay, base_delay * 2**attempt)
            delay *= 0.5 + random.random() / 2
            logger.warning(
                f"Vector database call failed ({type(exc).__name__}: {exc}); "
                f"retrying in {delay:.1f}s ({attempt + 1}/{attempts - 1})"
            )
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
