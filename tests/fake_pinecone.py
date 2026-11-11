"""A stand-in for the Pinecone client, backed by a SQLite file.

Data lives on disk so the index builder's writer process, which unpickles the
connection, sees the same indexes as the test. Like Pinecone, the fake:

- scores with the squared Euclidean distance (the ``euclidean`` metric), exactly;
- applies filters during the search, and only accepts explicit operators
  (``{"field": {"$eq": value}}``), so untranslated ChromaDB filters fail;
- rejects null metadata values, more than 1,000 records per upsert or fetch,
  and ``top_k`` above 10,000.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from aatm.vector_db import PineconeConnection


class FakePineconeError(RuntimeError):
    """Raised for requests real Pinecone would reject."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS indexes (
    name TEXT PRIMARY KEY, dimension INTEGER, metric TEXT, tags TEXT
);
CREATE TABLE IF NOT EXISTS records (
    index_name TEXT, id TEXT, vals BLOB, metadata TEXT,
    PRIMARY KEY (index_name, id)
);
"""


class _Db:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        with self.connect() as con:
            con.executescript(_SCHEMA)

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30, check_same_thread=False)

    def run(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock, closing(self.connect()) as con, con:
            return con.execute(sql, params).fetchall()


def _check_filter(where: Any) -> None:
    if not isinstance(where, dict) or not where:
        raise FakePineconeError(f"Bad filter: {where!r}")
    for key, value in where.items():
        if key in ("$and", "$or"):
            for clause in value:
                _check_filter(clause)
        elif key.startswith("$"):
            raise FakePineconeError(f"Unknown operator {key!r}")
        elif not isinstance(value, dict):
            raise FakePineconeError(
                f"The fake needs explicit operators; got {key!r}: {value!r}"
            )


def _matches(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    for key, value in where.items():
        if key == "$and":
            if not all(_matches(metadata, clause) for clause in value):
                return False
            continue
        if key == "$or":
            if not any(_matches(metadata, clause) for clause in value):
                return False
            continue
        for op, operand in value.items():
            present = key in metadata
            field_value = metadata.get(key)
            if op == "$eq":
                ok = present and field_value == operand
            elif op == "$ne":
                ok = not present or field_value != operand
            elif op == "$in":
                ok = present and field_value in operand
            elif op == "$nin":
                ok = not present or field_value not in operand
            elif op in ("$gt", "$gte", "$lt", "$lte"):
                ok = (
                    present
                    and {
                        "$gt": field_value > operand,
                        "$gte": field_value >= operand,
                        "$lt": field_value < operand,
                        "$lte": field_value <= operand,
                    }[op]
                )
            else:
                raise FakePineconeError(f"Unknown operator {op!r}")
            if not ok:
                return False
    return True


class FakeIndex:
    def __init__(self, db: _Db, name: str):
        self._db = db
        self.name = name
        self.queries: list[dict[str, Any]] = []

    def _rows(self) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
        rows = self._db.run(
            "SELECT id, vals, metadata FROM records WHERE index_name = ?", (self.name,)
        )
        return [
            (identifier, np.frombuffer(vals, dtype=np.float64), json.loads(metadata))
            for identifier, vals, metadata in rows
        ]

    def upsert(self, vectors: list[dict[str, Any]], show_progress: bool = True) -> Any:
        if len(vectors) > 1000:
            raise FakePineconeError(f"{len(vectors)} records in one upsert")
        ((dimension,),) = self._db.run(
            "SELECT dimension FROM indexes WHERE name = ?", (self.name,)
        )
        for record in vectors:
            metadata = record.get("metadata") or {}
            if any(value is None for value in metadata.values()):
                raise FakePineconeError("Metadata values cannot be null")
            if len(record["values"]) != dimension:
                raise FakePineconeError("Vector dimension does not match the index")
            # Pinecone stores every number as a float.
            stored = {
                key: float(value)
                if isinstance(value, int) and not isinstance(value, bool)
                else value
                for key, value in metadata.items()
            }
            self._db.run(
                "INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?)",
                (
                    self.name,
                    record["id"],
                    np.asarray(record["values"], dtype=np.float64).tobytes(),
                    json.dumps(stored),
                ),
            )
        return {"upserted_count": len(vectors)}

    def fetch(self, ids: list[str]) -> Any:
        if len(ids) > 1000:
            raise FakePineconeError(f"{len(ids)} IDs in one fetch")
        wanted = set(ids)
        return {
            "vectors": {
                identifier: {"id": identifier}
                for identifier, _, _ in self._rows()
                if identifier in wanted
            }
        }

    def query(
        self,
        vector: list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = False,
    ) -> Any:
        if top_k > 10_000:
            raise FakePineconeError(f"top_k {top_k} is above 10,000")
        if filter is not None:
            _check_filter(filter)
        self.queries.append({"top_k": top_k, "filter": filter})
        query = np.asarray(vector, dtype=np.float64)
        scored = [
            (float(np.sum((values - query) ** 2)), identifier, metadata)
            for identifier, values, metadata in self._rows()
            if filter is None or _matches(metadata, filter)
        ]
        scored.sort(key=lambda item: (item[0], item[1]))
        return SimpleNamespace(
            matches=[
                SimpleNamespace(
                    id=identifier,
                    score=score,
                    metadata=metadata if include_metadata else None,
                )
                for score, identifier, metadata in scored[:top_k]
            ]
        )

    def describe_index_stats(self) -> Any:
        ((count,),) = self._db.run(
            "SELECT COUNT(*) FROM records WHERE index_name = ?", (self.name,)
        )
        return {"total_vector_count": count}


class FakePineconeClient:
    def __init__(self, path: str):
        self._db = _Db(path)
        self.indexes: dict[str, FakeIndex] = {}

    def has_index(self, name: str) -> bool:
        return bool(self._db.run("SELECT 1 FROM indexes WHERE name = ?", (name,)))

    def create_index(
        self,
        name: str,
        dimension: int,
        metric: str,
        spec: Any = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        if self.has_index(name):
            raise FakePineconeError(f"Index {name!r} already exists")
        if len(name) > 45:
            raise FakePineconeError(f"Index name {name!r} is too long")
        if tags and not all(isinstance(value, str) for value in tags.values()):
            raise FakePineconeError("Tag values must be strings")
        self._db.run(
            "INSERT INTO indexes VALUES (?, ?, ?, ?)",
            (name, dimension, metric, json.dumps(tags or {})),
        )

    def describe_index(self, name: str) -> Any:
        rows = self._db.run(
            "SELECT dimension, metric, tags FROM indexes WHERE name = ?", (name,)
        )
        if not rows:
            raise FakePineconeError(f"No index {name!r}")
        ((dimension, metric, tags),) = rows
        return SimpleNamespace(
            name=name,
            dimension=dimension,
            metric=metric,
            tags=json.loads(tags) or None,
            host=f"{name}.fake.pinecone.io",
        )

    def delete_index(self, name: str) -> None:
        self._db.run("DELETE FROM records WHERE index_name = ?", (name,))
        self._db.run("DELETE FROM indexes WHERE name = ?", (name,))

    def list_indexes(self) -> Any:
        names = [name for (name,) in self._db.run("SELECT name FROM indexes")]
        return SimpleNamespace(names=lambda: names)

    def index(self, name: str = "", host: str = "") -> FakeIndex:
        if not self.has_index(name):
            raise FakePineconeError(f"No index {name!r}")
        return self.indexes.setdefault(name, FakeIndex(self._db, name))


@dataclass(frozen=True, slots=True)
class FakePineconeConnection(PineconeConnection):
    """Behaves like a Pinecone connection but stores data in ``path``."""

    path: str = field(default="")

    def client(self) -> FakePineconeClient:
        return FakePineconeClient(self.path)


def fake_pinecone(path) -> FakePineconeConnection:
    return FakePineconeConnection(api_key="secret-key", path=str(path))
