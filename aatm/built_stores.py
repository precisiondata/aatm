"""Which vector databases are built, and which one a retriever defaults to.

``aatm init`` builds the vector database of one embedding model, and
``aatm vector-db upload`` copies one to Chroma Cloud, Pinecone or PostgreSQL.
Each records
what it built in ``.aatm/config.yaml``:

```yaml
vector_databases:
- embedding_model: text-embedding-3-small
  vector_db: local
  location: .aatm/chroma_vector_dbs/text-embedding-3-small
  built_at: '2026-11-12T10:31:05+00:00'
```

When no retriever is given (``TerminologyMapper()``, ``AgenticCoder(model)``,
``aatm map``), `default_retriever_name` picks the most recently built model for
the configured vector database (``AATM_VECTOR_DB``). Without a record, it looks
for built databases (as ``aatm vector-db check`` does) and uses one only if
there is exactly one.

A retriever only opens a database that is built (`check_built`): naming a model
whose database does not exist or is empty raises
`VectorDatabaseNotBuiltError`, which lists the databases that are built and how
to build the one asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import yaml

from aatm.logs import get_logger
from aatm.vector_db import (
    VECTOR_DB_MODES,
    VectorDatabaseConfigError,
    VectorDbMode,
    resolve_vector_db_mode,
)

logger = get_logger(__name__)

CONFIG_PATH = Path(".aatm/config.yaml")
"""Where ``aatm init`` records the vector databases it built."""


class VectorDatabaseNotBuiltError(ValueError):
    """The vector database of the requested embedding model is not built."""


class NoVectorDatabaseError(VectorDatabaseNotBuiltError):
    """No vector database is built, so there is no retriever to default to."""


@dataclass(frozen=True)
class BuiltStore:
    """A built vector database: an embedding model's index, and where it lives."""

    embedding_model: str
    """Registry name of the embedding model, e.g. ``text-embedding-3-small``."""

    vector_db: VectorDbMode
    """``local``, ``cloud``, ``pinecone`` or ``pgvector``."""

    location: str
    """The local directory, Chroma Cloud collection, Pinecone index or PostgreSQL table."""

    count: int | None = None
    """Number of expressions, if it was counted."""

    built_at: str | None = None
    """When it was built or uploaded (ISO 8601), if recorded."""

    def describe(self) -> str:
        """One line for messages: model, vector database, location and size."""
        kind = {"cloud": "collection ", "pinecone": "index ", "pgvector": "table "}.get(
            self.vector_db, ""
        )
        line = f"{self.embedding_model}  {self.vector_db}  {kind}{self.location}"
        if self.count is not None:
            line += f"  ({self.count:,} expressions)"
        return line


# --- the record in .aatm/config.yaml ---------------------------------------------------


def _read_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        logger.warning(f"Ignoring {CONFIG_PATH}: it is not valid YAML ({e})")
        return {}
    return data if isinstance(data, dict) else {}


def record_vector_database(
    embedding_model: str,
    vector_db: VectorDbMode,
    location: str | Path,
    *,
    keep_date: bool = False,
) -> None:
    """Record in ``.aatm/config.yaml`` that a vector database is built.

    A record of the same model and vector database is replaced.

    Args:
        embedding_model: Registry name of the embedding model.
        vector_db: ``local``, ``cloud``, ``pinecone`` or ``pgvector``.
        location: The local directory, or the collection or index name.
        keep_date: Keep the build date of an existing record (an existing
            database kept as it was), and only date a new one.
    """
    config = _read_config()
    entries = [e for e in config.get("vector_databases") or [] if isinstance(e, dict)]
    previous = next(
        (
            e
            for e in entries
            if e.get("embedding_model") == embedding_model
            and e.get("vector_db") == vector_db
        ),
        None,
    )
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if keep_date and previous is not None and previous.get("built_at"):
        built_at = str(previous["built_at"])
    entries = [e for e in entries if e is not previous]
    entries.append(
        {
            "embedding_model": embedding_model,
            "vector_db": vector_db,
            "location": Path(location).as_posix()
            if vector_db == "local"
            else str(location),
            "built_at": built_at,
        }
    )
    config["vector_databases"] = entries
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def recorded_vector_databases(vector_db: str | None = None) -> list[BuiltStore]:
    """The vector databases recorded in ``.aatm/config.yaml``, newest first.

    Args:
        vector_db: Only those in this vector database (``local``, ``cloud``,
            ``pinecone`` or ``pgvector``). ``None`` returns all of them.
    """
    stores = []
    for entry in _read_config().get("vector_databases") or []:
        if not isinstance(entry, dict):
            continue
        model, mode = entry.get("embedding_model"), entry.get("vector_db")
        if not model or mode not in VECTOR_DB_MODES:
            continue
        if vector_db is not None and mode != vector_db:
            continue
        stores.append(
            BuiltStore(
                embedding_model=str(model),
                vector_db=cast(VectorDbMode, mode),
                location=str(entry.get("location", "")),
                built_at=str(entry["built_at"]) if entry.get("built_at") else None,
            )
        )
    # Newest first; of two built in the same second, the one recorded last.
    order = {id(store): i for i, store in enumerate(stores)}
    return sorted(stores, key=lambda s: (s.built_at or "", order[id(s)]), reverse=True)


# --- what is built -----------------------------------------------------------------


def _local_count(path: str, collection_name: str) -> int | None:
    """Count a local collection's records from ``chroma.sqlite3``, read-only.

    Opening a ChromaDB client loads the collection's vector index, which takes
    memory and time for each of several multi-million-row databases; the
    record count is in the SQLite file. Returns ``None`` if the file cannot be
    read this way (e.g. another ChromaDB version), so the caller opens a client.

    Raises:
        chromadb.errors.NotFoundError: If there is no such collection.
    """
    import sqlite3

    from chromadb.errors import NotFoundError

    file = Path(path) / "chroma.sqlite3"
    if not file.exists():
        return None
    try:
        db = sqlite3.connect(f"{file.resolve().as_uri()}?mode=ro", uri=True)
        try:
            row = db.execute(
                "SELECT id FROM collections WHERE name = ?", (collection_name,)
            ).fetchone()
            if row is None:
                raise NotFoundError(collection_name)
            (count,) = db.execute(
                "SELECT COUNT(*) FROM embeddings WHERE segment_id IN "
                "(SELECT id FROM segments WHERE collection = ? AND scope = 'METADATA')",
                (row[0],),
            ).fetchone()
        finally:
            db.close()
    except sqlite3.Error as e:
        logger.debug(f"Could not count {file} with SQLite: {e!r}")
        return None
    return int(count)


def _store_status(spec: Any, mode: VectorDbMode, local_path: str | None = None):
    """Return ``(location, count, reason)`` for one model's database.

    ``count`` is ``None`` and ``reason`` says why when it is not built (it does
    not exist or is empty).
    """
    if mode == "pinecone":
        from aatm.vector_stores import PineconeVectorStore, pinecone_index_exists

        connection, index_name = spec.pinecone_location()
        client = connection.client()
        if not pinecone_index_exists(client, index_name):
            return index_name, None, f"Pinecone has no index '{index_name}'"
        count = PineconeVectorStore.open(client, index_name).count()
        if count == 0:
            return index_name, None, f"Pinecone index '{index_name}' is empty"
        return index_name, count, None

    if mode == "pgvector":
        from aatm.pgvector_store import pg_table_count

        connection, table_name = spec.pgvector_location()
        count = pg_table_count(connection.dsn, table_name)
        if count is None:
            return table_name, None, f"PostgreSQL has no table '{table_name}'"
        if count == 0:
            return table_name, None, f"PostgreSQL table '{table_name}' is empty"
        return table_name, count, None

    from chromadb.errors import NotFoundError

    connection, collection_name = spec.chroma_location(mode, local_path)
    if mode == "local":
        # Opening a local client would create the directory: check it first.
        location = Path(local_path or spec.chromadb_path).as_posix()
        if not Path(location).exists():
            return location, None, f"{location} not found"
        where = f"{location} has"
    else:
        location = collection_name
        where = "Chroma Cloud has"
    try:
        local_count = (
            _local_count(location, collection_name) if mode == "local" else None
        )
        if local_count is not None:
            count = local_count
        else:
            collection = connection.client().get_collection(collection_name)
            count = collection.count()
    except NotFoundError:
        return location, None, f"{where} no collection '{collection_name}'"
    except VectorDatabaseConfigError:
        raise
    except Exception as e:  # e.g. an interrupted build left a broken index
        logger.debug(f"Could not open {location}: {e!r}")
        return location, None, f"{location} could not be opened: {e}"
    if count == 0:
        return location, None, f"{where} an empty collection '{collection_name}'"
    return location, count, None


def built_vector_databases(vector_db: str | None = None) -> list[BuiltStore]:
    """Find the embedding models whose vector database is built and not empty.

    Args:
        vector_db: ``local``, ``cloud``, ``pinecone`` or ``pgvector``. If ``None``,
            ``AATM_VECTOR_DB`` decides, and otherwise ``local``.

    Raises:
        VectorDatabaseConfigError: If the vector database configuration is
            invalid, e.g. cloud mode without ``CHROMA_API_KEY``.
    """
    from aatm.registries.retrievers import CHROMADB_RETRIEVER_MODEL_REGISTRY

    mode = resolve_vector_db_mode(vector_db)
    recorded = {s.embedding_model: s for s in recorded_vector_databases(mode)}
    stores = []
    for name, spec in CHROMADB_RETRIEVER_MODEL_REGISTRY.items():
        location, count, reason = _store_status(spec, mode)
        if reason is None:
            built_at = recorded[name].built_at if name in recorded else None
            stores.append(BuiltStore(name, mode, location, count, built_at))
    return stores


def _build_command(embedding_model: str, mode: VectorDbMode) -> str:
    command = f"aatm init --embedding-model {embedding_model}"
    if mode != "local":
        command += f" --vector-db {mode}"
        command += (
            f", or copy a local one with: aatm vector-db upload --to {mode} "
            f"-e {embedding_model}"
        )
    return command


def _list_built(stores: list[BuiltStore]) -> str:
    return "\n".join(f"  - {store.describe()}" for store in stores)


def check_built(
    embedding_model: str, vector_db: str | None = None, local_path: str | None = None
) -> None:
    """Raise unless the vector database of ``embedding_model`` is built.

    Args:
        embedding_model: Registry name of the embedding model.
        vector_db: ``local``, ``cloud``, ``pinecone`` or ``pgvector``. If ``None``,
            ``AATM_VECTOR_DB`` decides, and otherwise ``local``.
        local_path: Directory of the local database, in local mode. Defaults
            to the model's directory under ``.aatm/chroma_vector_dbs``.

    Raises:
        VectorDatabaseNotBuiltError: If the database does not exist or is
            empty. The message lists the databases that are built and how to
            build this one.
    """
    from aatm.registries.retrievers import CHROMADB_RETRIEVER_MODEL_REGISTRY

    mode = resolve_vector_db_mode(vector_db)
    spec = CHROMADB_RETRIEVER_MODEL_REGISTRY[embedding_model]
    _, _, reason = _store_status(spec, mode, local_path)
    if reason is None:
        return
    built = built_vector_databases(mode)
    lines = [f"No vector database for '{embedding_model}' ({reason})."]
    if built:
        lines += ["Built and ready:", _list_built(built)]
        lines.append(
            f'Use retriever="{built[0].embedding_model}", or build it with: '
            + _build_command(embedding_model, mode)
        )
    else:
        lines.append(
            f"No vector database is built in {mode} mode. Build it with: "
            + _build_command(embedding_model, mode)
        )
    raise VectorDatabaseNotBuiltError("\n".join(lines))


def default_retriever_name(vector_db: str | None = None) -> str:
    """The embedding model a retriever uses when none is given.

    See `default_vector_database`.
    """
    return default_vector_database(vector_db).embedding_model


def default_vector_database(vector_db: str | None = None) -> BuiltStore:
    """The vector database a retriever uses when none is given.

    The most recently built one recorded for the vector database, or, with no
    record, the only built one.

    Args:
        vector_db: ``local``, ``cloud``, ``pinecone`` or ``pgvector``. If ``None``,
            ``AATM_VECTOR_DB`` decides, and otherwise ``local``.

    Raises:
        NoVectorDatabaseError: If no vector database is built.
        VectorDatabaseNotBuiltError: If several are built and none is recorded,
            so the choice is the user's.
    """
    from aatm.registries.retrievers import CHROMADB_RETRIEVER_MODEL_REGISTRY

    mode = resolve_vector_db_mode(vector_db)
    recorded = [
        s
        for s in recorded_vector_databases(mode)
        if s.embedding_model in CHROMADB_RETRIEVER_MODEL_REGISTRY
    ]
    if recorded:
        name = recorded[0].embedding_model
        if len(recorded) > 1:
            others = ", ".join(s.embedding_model for s in recorded[1:])
            logger.info(
                f"Using retriever '{name}', the most recently built "
                f"{mode} vector database (also built: {others})"
            )
        else:
            logger.info(f"Using retriever '{name}', built by aatm init")
        return recorded[0]

    built = built_vector_databases(mode)
    if len(built) == 1:
        name = built[0].embedding_model
        logger.info(f"Using retriever '{name}', the only built {mode} vector database")
        return built[0]
    if not built:
        raise NoVectorDatabaseError(
            f"No vector database is built in {mode} mode, so there is no retriever "
            "to default to. Build one with: aatm init"
            + ("" if mode == "local" else f" --vector-db {mode}")
        )
    raise VectorDatabaseNotBuiltError(
        "Several vector databases are built and none is recorded in "
        f"{CONFIG_PATH.as_posix()}:\n{_list_built(built)}\n"
        f'Choose one, e.g. retriever="{built[0].embedding_model}".'
    )
