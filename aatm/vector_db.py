"""Choose where the ChromaDB vector database lives: on this machine or in Chroma Cloud.

AATM stores its expression index in ChromaDB. By default each embedding model
gets a local persistent database under ``.aatm/chroma_vector_dbs/<model>``.
With ``AATM_VECTOR_DB=cloud`` the same index lives in a Chroma Cloud database
instead, so it can be built once and shared, and machines without the disk space
or GPU for a local index can still run retrieval.

Configuration comes from environment variables (a ``.env`` file works too):

- ``AATM_VECTOR_DB``: ``local`` (default) or ``cloud``.
- ``CHROMA_API_KEY``: Chroma Cloud API key. Required in cloud mode.
- ``CHROMA_TENANT`` and ``CHROMA_DATABASE``: optional. Chroma Cloud infers them
  when the API key is scoped to a single database.
- ``CHROMA_CLOUD_HOST``: optional, defaults to ``api.trychroma.com``.

In cloud mode one database holds the indexes of every embedding model, so each
model gets its own collection (see `ChromaRetrieverModelSpec.cloud_collection_name`).

Every place that opens a ChromaDB client goes through `ChromaConnection.client`,
so switching between local and cloud is configuration only.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from aatm.logs import get_logger

logger = get_logger(__name__)

VectorDbMode = Literal["local", "cloud"]
"""Where the vector database lives."""

VECTOR_DB_MODES: tuple[VectorDbMode, ...] = ("local", "cloud")

VECTOR_DB_ENV_VAR = "AATM_VECTOR_DB"
CHROMA_API_KEY_ENV_VAR = "CHROMA_API_KEY"
CHROMA_TENANT_ENV_VAR = "CHROMA_TENANT"
CHROMA_DATABASE_ENV_VAR = "CHROMA_DATABASE"
CHROMA_CLOUD_HOST_ENV_VAR = "CHROMA_CLOUD_HOST"

DEFAULT_CLOUD_HOST = "api.trychroma.com"

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
        ``local`` or ``cloud``.

    Raises:
        VectorDatabaseConfigError: If the mode is not ``local`` or ``cloud``.
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
            VectorDatabaseConfigError: If the mode is invalid, or cloud mode is
                selected without ``CHROMA_API_KEY``.
        """
        env = os.environ if env is None else env
        resolved = resolve_vector_db_mode(mode, env)
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


def is_transient_error(exc: BaseException) -> bool:
    """Return whether a ChromaDB call that raised ``exc`` is worth retrying.

    Rate limits, server errors and network failures are transient. Everything
    else (bad input, missing collections, auth and quota errors) is not.
    """
    import httpx
    from chromadb.errors import InternalError, RateLimitError

    return isinstance(exc, (RateLimitError, InternalError, httpx.TransportError))


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
