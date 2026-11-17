"""Concept lookups in the local OMOP vocabulary database built by ``aatm init``."""

import sqlite3
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol

from pydantic import BaseModel

from aatm.omop.mapping import ConceptRef

DEFAULT_VOCABULARY_DB = Path(".aatm/omop.db")

VOCABULARY_INDEXES: dict[str, tuple[str, str]] = {
    "idx_aatm_concept_id": ("concept", "concept_id"),
    "idx_aatm_concept_vocabulary_code": ("concept", "vocabulary_id, concept_code"),
    "idx_aatm_concept_code_lower": ("concept", "lower(concept_code)"),
    "idx_aatm_relationship_id1": (
        "concept_relationship",
        "concept_id_1, relationship_id",
    ),
    "idx_aatm_synonym_id": ("concept_synonym", "concept_id"),
    "idx_aatm_ancestor_descendant": ("concept_ancestor", "descendant_concept_id"),
    "idx_aatm_ancestor_ancestor": ("concept_ancestor", "ancestor_concept_id"),
}
"""Indexes of the vocabulary database: ``name -> (table, columns)``.

They serve the lookups by id and code, "Maps to" and the hierarchy, which
otherwise scan tables of tens of millions of rows (seconds per lookup, under a
millisecond with them). Searches by name (``LIKE '%word%'``) still scan.
"""

_warned_unindexed: set[Path] = set()


def _tables(con: sqlite3.Connection) -> set[str]:
    return {
        name
        for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def missing_vocabulary_indexes(path: str | Path = DEFAULT_VOCABULARY_DB) -> list[str]:
    """`VOCABULARY_INDEXES` that a vocabulary database lacks, for its tables."""
    uri = f"file:{Path(path).as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        tables = _tables(con)
        existing = {
            name
            for (name,) in con.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
    return [
        name
        for name, (table, _) in VOCABULARY_INDEXES.items()
        if table in tables and name not in existing
    ]


def index_vocabulary_database(
    path: str | Path = DEFAULT_VOCABULARY_DB,
    progress: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """Create the `VOCABULARY_INDEXES` a vocabulary database lacks.

    Indexes of tables the database doesn't have are skipped. When any is created,
    ``ANALYZE`` updates the statistics the query planner uses. A full Athena
    download takes a few minutes, mostly ``concept_relationship`` and
    ``concept_ancestor``.

    Args:
        path: The database, by default the one ``aatm init`` builds.
        progress: Called with each index's name before it is created.

    Returns:
        The names of the indexes created.
    """
    missing = missing_vocabulary_indexes(path)
    if not missing:
        return []
    con = sqlite3.connect(path)
    try:
        for name in missing:
            table, columns = VOCABULARY_INDEXES[name]
            if progress is not None:
                progress(name)
            con.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({columns})")
            con.commit()
        con.execute("ANALYZE")
        con.commit()
    finally:
        con.close()
    _warned_unindexed.discard(Path(path))
    return missing


class Concept(BaseModel):
    """A row of the OMOP ``concept`` table."""

    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_code: str
    standard_concept: Optional[str] = None
    invalid_reason: Optional[str] = None

    @property
    def is_standard(self) -> bool:
        return self.standard_concept == "S"

    def __str__(self) -> str:
        return (
            f"{self.vocabulary_id}:{self.concept_code} {self.concept_name!r} "
            f"(concept_id {self.concept_id}, domain {self.domain_id})"
        )


class ConceptLookup(Protocol):
    """Looks concepts up by code or id, and follows "Maps to" relationships."""

    def by_code(self, refs: Iterable[ConceptRef]) -> dict[ConceptRef, Concept]: ...

    def by_id(self, concept_ids: Iterable[int]) -> dict[int, Concept]: ...

    def maps_to(self, concept_ids: Iterable[int]) -> dict[int, list[int]]: ...

    def search_codes(
        self, vocabulary_id: str, codes: Iterable[str]
    ) -> dict[str, list[Concept]]:
        """Concepts of a vocabulary whose code matches, ignoring case, keyed by lowercase code."""
        ...


_CONCEPT_COLUMNS = (
    "concept_id, concept_name, domain_id, vocabulary_id, concept_code, "
    "standard_concept, invalid_reason"
)


class SqliteConceptLookup:
    """`ConceptLookup` over the SQLite vocabulary database (``.aatm/omop.db``).

    The database is opened read-only. Lookups are batched, so a database without
    `VOCABULARY_INDEXES` scans each table once per call, whatever the number of
    codes; a warning says how to add them.
    """

    def __init__(self, path: str | Path = DEFAULT_VOCABULARY_DB):
        self.path = Path(path)
        if not self.path.exists():
            raise ValueError(
                f"OMOP vocabulary database not found: {self.path}. "
                "Run `aatm init` to build it."
            )
        self._warn_if_unindexed()

    def _warn_if_unindexed(self) -> None:
        if self.path in _warned_unindexed:
            return
        _warned_unindexed.add(self.path)
        try:
            missing = missing_vocabulary_indexes(self.path)
        except sqlite3.Error:
            return  # Not a vocabulary database we can read: lookups will say so.
        if missing:
            from aatm.logs import get_logger

            get_logger(__name__).warning(
                f"The vocabulary database {self.path} lacks {len(missing)} index(es), "
                "so lookups take seconds each instead of milliseconds. Add them "
                f"with `aatm index-vocab --path {self.path.as_posix()}` (a few minutes)."
            )

    def _query(self, sql: str, params: list) -> list[tuple]:
        uri = f"file:{self.path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as con:
            return con.execute(sql, params).fetchall()

    @staticmethod
    def _concept(row: tuple) -> Concept:
        return Concept(
            concept_id=int(row[0]),
            concept_name=row[1],
            domain_id=row[2],
            vocabulary_id=row[3],
            concept_code=row[4],
            standard_concept=row[5] or None,
            invalid_reason=row[6] or None,
        )

    def by_code(self, refs: Iterable[ConceptRef]) -> dict[ConceptRef, Concept]:
        refs = set(refs)
        if not refs:
            return {}
        codes = sorted({r.concept_code for r in refs})
        rows = self._query(
            f"SELECT {_CONCEPT_COLUMNS} FROM concept "
            f"WHERE concept_code IN ({','.join('?' * len(codes))})",
            codes,
        )
        found: dict[ConceptRef, Concept] = {}
        for row in rows:
            ref = ConceptRef(vocabulary_id=row[3], concept_code=row[4])
            if ref in refs:
                found[ref] = self._concept(row)
        return found

    def by_id(self, concept_ids: Iterable[int]) -> dict[int, Concept]:
        ids = sorted({str(i) for i in concept_ids})
        if not ids:
            return {}
        rows = self._query(
            f"SELECT {_CONCEPT_COLUMNS} FROM concept "
            f"WHERE concept_id IN ({','.join('?' * len(ids))})",
            ids,
        )
        return {int(row[0]): self._concept(row) for row in rows}

    def maps_to(self, concept_ids: Iterable[int]) -> dict[int, list[int]]:
        ids = sorted({str(i) for i in concept_ids})
        if not ids:
            return {}
        rows = self._query(
            "SELECT concept_id_1, concept_id_2 FROM concept_relationship "
            f"WHERE concept_id_1 IN ({','.join('?' * len(ids))}) "
            "AND relationship_id = 'Maps to' AND invalid_reason IS NULL",
            ids,
        )
        targets: dict[int, list[int]] = {}
        for source, target in rows:
            targets.setdefault(int(source), []).append(int(target))
        return targets

    def search_codes(
        self, vocabulary_id: str, codes: Iterable[str]
    ) -> dict[str, list[Concept]]:
        wanted = sorted({c.strip().lower() for c in codes if c.strip()})
        if not wanted:
            return {}
        rows = self._query(
            f"SELECT {_CONCEPT_COLUMNS} FROM concept WHERE vocabulary_id = ? "
            f"AND lower(concept_code) IN ({','.join('?' * len(wanted))})",
            [vocabulary_id, *wanted],
        )
        found: dict[str, list[Concept]] = {}
        for row in rows:
            found.setdefault(row[4].lower(), []).append(self._concept(row))
        return found
