"""Concept lookups in the local OMOP vocabulary database built by ``aatm init``."""

import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Protocol

from pydantic import BaseModel

from aatm.omop.mapping import ConceptRef

DEFAULT_VOCABULARY_DB = Path(".aatm/omop.db")


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

    The database is opened read-only. Lookups are batched, since the tables are not
    indexed: each call scans a table once, whatever the number of codes.
    """

    def __init__(self, path: str | Path = DEFAULT_VOCABULARY_DB):
        self.path = Path(path)
        if not self.path.exists():
            raise ValueError(
                f"OMOP vocabulary database not found: {self.path}. "
                "Run `aatm init` to build it."
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
