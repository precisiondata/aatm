"""The tools the concept search agent calls to explore the vocabularies.

- **Semantic search** over the vector index (ChromaDB or Pinecone) of a retriever, filtered by domain and
  vocabulary.
- **Lexical search**: concept names and synonyms containing every word of a query, in
  the local vocabulary database (``.aatm/omop.db``).
- **Code lookup**: a code in any vocabulary, with the standard concepts it maps to.
- **Concept details**: domain, standard status, synonyms, "Maps to" and, for LOINC,
  the axes of the code (component, property, time, system, scale, method).
- **Hierarchy**: the direct parents or children of a standard concept.

Every tool except concept details only returns standard concepts within the search
scope (domain and vocabularies), so whatever the agent picks from its results is a
valid answer.
"""

from pathlib import Path
from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel

from aatm.omop.coding import retriever_filter
from aatm.omop.vocabulary import DEFAULT_VOCABULARY_DB, Concept, SqliteConceptLookup

# LOINC "Has <axis>" relationships, in the order of a LOINC name.
LOINC_AXES = {
    "Has component": "component",
    "Has property": "property",
    "Has time aspect": "time",
    "Has system": "system",
    "Has scale type": "scale",
    "Has method": "method",
}


class Candidate(BaseModel):
    """A concept returned by a tool.

    Attributes:
        concept_id: The concept's id.
        concept_name: Its name.
        vocabulary_id: Its vocabulary.
        concept_code: Its code in that vocabulary.
        domain_id: Its domain.
        standard: Whether it is a standard concept.
        matched: The indexed text that matched a semantic search, when it is not the
            concept name (a synonym, or the name of a concept that maps to it).
    """

    concept_id: int
    concept_name: str
    vocabulary_id: str
    concept_code: str
    domain_id: str
    standard: Optional[bool] = None
    matched: Optional[str] = None

    @classmethod
    def from_concept(cls, concept: Concept) -> "Candidate":
        return cls(
            concept_id=concept.concept_id,
            concept_name=concept.concept_name,
            vocabulary_id=concept.vocabulary_id,
            concept_code=concept.concept_code,
            domain_id=concept.domain_id,
            standard=concept.is_standard,
        )

    def line(self) -> str:
        """One line for the agent's prompt."""
        text = (
            f"{self.concept_id} | {self.vocabulary_id} {self.concept_code} | "
            f"{self.concept_name} | {self.domain_id}"
        )
        if self.standard is False:
            text += " | non-standard"
        if self.matched and self.matched.casefold() != self.concept_name.casefold():
            text += f" | matched: {self.matched!r}"
        return text

    def short(self) -> str:
        """``id name``, for traces."""
        return f"{self.concept_id} {self.concept_name}"


class VocabularySearch(SqliteConceptLookup):
    """Searches in the SQLite vocabulary database built by ``aatm init``.

    The tables are not indexed, so each search scans a table once (well under a
    second for the concept table of a typical download).
    """

    def __init__(self, path: str | Path = DEFAULT_VOCABULARY_DB):
        super().__init__(path)

    @staticmethod
    def _scope_sql(
        domain: Optional[str], vocabularies: Optional[Sequence[str]], alias: str = "c"
    ) -> tuple[str, list]:
        sql = f" AND {alias}.standard_concept = 'S' AND {alias}.invalid_reason IS NULL"
        params: list = []
        if domain is not None:
            sql += f" AND {alias}.domain_id = ?"
            params.append(domain)
        if vocabularies:
            sql += (
                f" AND {alias}.vocabulary_id IN ({','.join('?' * len(vocabularies))})"
            )
            params.extend(vocabularies)
        return sql, params

    def search_names(
        self,
        query: str,
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
        limit: int = 20,
    ) -> list[Candidate]:
        """Standard concepts whose name or a synonym contains every word of `query`.

        Ranked by: exact name, name starting with the query, then shorter names, which
        are usually the more general concepts.
        """
        words = [w for w in query.casefold().split() if w]
        if not words:
            return []
        scope, scope_params = self._scope_sql(domain, vocabularies)
        like = " AND ".join("lower({col}) LIKE ?" for _ in words)
        like_params = [f"%{w}%" for w in words]
        rows = self._query(
            "SELECT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id, "
            "c.concept_code, c.standard_concept, c.invalid_reason, c.concept_name "
            f"FROM concept c WHERE {like.format(col='c.concept_name')}{scope} "
            "UNION "
            "SELECT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id, "
            "c.concept_code, c.standard_concept, c.invalid_reason, s.concept_synonym_name "
            "FROM concept_synonym s JOIN concept c ON c.concept_id = s.concept_id "
            f"WHERE {like.format(col='s.concept_synonym_name')}{scope}",
            [*like_params, *scope_params, *like_params, *scope_params],
        )
        q = query.casefold().strip()
        best: dict[int, tuple[tuple, Candidate]] = {}
        for row in rows:
            candidate = Candidate.from_concept(self._concept(row[:7]))
            matched = row[7] or candidate.concept_name
            candidate.matched = matched
            m = matched.casefold()
            rank = (0 if m == q else 1 if m.startswith(q) else 2, len(m), m)
            current = best.get(candidate.concept_id)
            if current is None or rank < current[0]:
                best[candidate.concept_id] = (rank, candidate)
        ordered = sorted(best.values(), key=lambda item: item[0])
        return [candidate for _, candidate in ordered[:limit]]

    def search_code(self, code: str) -> list[Candidate]:
        """Concepts of any vocabulary whose code is `code`, ignoring case."""
        code = code.strip()
        if not code:
            return []
        rows = self._query(
            "SELECT concept_id, concept_name, domain_id, vocabulary_id, concept_code, "
            "standard_concept, invalid_reason FROM concept "
            "WHERE lower(concept_code) = ?",
            [code.lower()],
        )
        return [Candidate.from_concept(self._concept(row)) for row in rows]

    def relationships(self, concept_id: int) -> list[tuple[str, int]]:
        """Valid ``(relationship_id, concept_id_2)`` pairs of a concept."""
        rows = self._query(
            "SELECT relationship_id, concept_id_2 FROM concept_relationship "
            "WHERE concept_id_1 = ? AND invalid_reason IS NULL",
            [str(concept_id)],
        )
        return [(rel, int(target)) for rel, target in rows]

    def synonyms(self, concept_id: int, limit: int = 10) -> list[str]:
        rows = self._query(
            "SELECT concept_synonym_name FROM concept_synonym WHERE concept_id = ?",
            [str(concept_id)],
        )
        return sorted({r[0] for r in rows if r[0]})[:limit]

    def neighbours(
        self,
        concept_id: int,
        direction: Literal["parents", "children"],
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
        limit: int = 25,
    ) -> list[Candidate]:
        """Direct parents or children of a standard concept (``concept_ancestor``)."""
        this, other = (
            ("descendant", "ancestor")
            if direction == "parents"
            else ("ancestor", "descendant")
        )
        scope, scope_params = self._scope_sql(domain, vocabularies)
        rows = self._query(
            "SELECT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id, "
            "c.concept_code, c.standard_concept, c.invalid_reason "
            f"FROM concept_ancestor a JOIN concept c ON c.concept_id = a.{other}_concept_id "
            f"WHERE a.{this}_concept_id = ? AND a.min_levels_of_separation = '1'{scope}",
            [str(concept_id), *scope_params],
        )
        candidates = sorted(
            (Candidate.from_concept(self._concept(row)) for row in rows),
            key=lambda c: (len(c.concept_name), c.concept_name),
        )
        return candidates[:limit]


def _expression_candidate(expression: Any) -> Optional[Candidate]:
    """A `Candidate` from a retrieved expression (`RetrievedExpressionMetadata`)."""
    if not expression.std_concept_id:
        return None
    vocabulary = getattr(
        expression.std_vocabulary_id, "value", expression.std_vocabulary_id
    )
    return Candidate(
        concept_id=int(expression.std_concept_id),
        concept_name=expression.std_concept_name or "",
        vocabulary_id=vocabulary or "",
        concept_code=expression.std_vocabulary_code or "",
        domain_id=expression.std_domain_id or "",
        standard=True,
        matched=expression.expression,
    )


def candidates_from_expressions(expressions: Sequence[Any]) -> list[Candidate]:
    """Retrieved expressions as candidates, one per standard concept, in rank order."""
    seen: dict[int, Candidate] = {}
    for expression in expressions:
        candidate = _expression_candidate(expression)
        if candidate is not None and candidate.concept_id not in seen:
            seen[candidate.concept_id] = candidate
    return list(seen.values())


class ConceptSearchTools:
    """The agent's tools over a retriever and a vocabulary database.

    Args:
        retriever: A retriever with a ``retrieve(queries, where=..., top_k=...)``
            method, such as `aatm.retrievers.ChromaDBRetriever` or
            `aatm.retrievers.PineconeRetriever`, for semantic search.
        vocabulary: The vocabulary database, as a path or a `VocabularySearch`, for
            lexical search, code lookup, concept details and hierarchy. ``None``
            leaves only semantic search.
        top_k: Number of results per search.
    """

    def __init__(
        self,
        retriever: Any = None,
        vocabulary: VocabularySearch | str | Path | None = None,
        top_k: int = 10,
    ):
        if retriever is None and vocabulary is None:
            raise ValueError(
                "The concept search needs a retriever, a vocabulary, or both"
            )
        self.retriever = retriever
        self.vocabulary = (
            VocabularySearch(vocabulary)
            if isinstance(vocabulary, (str, Path))
            else vocabulary
        )
        self.top_k = top_k

    @property
    def actions(self) -> list[str]:
        """The tools available with this configuration."""
        actions = []
        if self.retriever is not None:
            actions.append("semantic_search")
        if self.vocabulary is not None:
            actions += ["lexical_search", "code_lookup", "concept_details", "hierarchy"]
        return actions

    def semantic_search(
        self,
        query: str,
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[Candidate]:
        if self.retriever is None:
            raise ValueError("semantic_search needs a retriever")
        k = top_k or self.top_k
        # Several indexed expressions (synonyms) can point to the same concept:
        # retrieve more to return about k distinct concepts.
        results = self.retriever.retrieve(
            [query], where=retriever_filter(domain, vocabularies), top_k=k * 2
        )
        return candidates_from_expressions(results.results[0])[:k]

    def lexical_search(
        self,
        query: str,
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[Candidate]:
        assert self.vocabulary is not None
        return self.vocabulary.search_names(
            query, domain, vocabularies, limit=top_k or self.top_k
        )

    def code_lookup(
        self,
        code: str,
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
    ) -> list[Candidate]:
        """The concepts with this code, and the in-scope standard concepts they map to."""
        assert self.vocabulary is not None
        found = self.vocabulary.search_code(code)
        maps_to = self.vocabulary.maps_to(c.concept_id for c in found if not c.standard)
        targets = self.vocabulary.by_id({t for ts in maps_to.values() for t in ts})
        results: list[Candidate] = []
        for candidate in found:
            results.append(candidate)
            for target_id in maps_to.get(candidate.concept_id, []):
                target = targets.get(target_id)
                if target is not None:
                    mapped = Candidate.from_concept(target)
                    mapped.matched = (
                        f"maps from {candidate.vocabulary_id} {candidate.concept_code}"
                    )
                    results.append(mapped)
        return [
            c for c in results if not c.standard or in_scope(c, domain, vocabularies)
        ]

    def concept_details(self, concept_id: int) -> str:
        assert self.vocabulary is not None
        concept = self.vocabulary.by_id([concept_id]).get(concept_id)
        if concept is None:
            return f"No concept with id {concept_id}."
        lines = [
            f"concept_id: {concept.concept_id}",
            f"name: {concept.concept_name}",
            f"vocabulary: {concept.vocabulary_id} {concept.concept_code}",
            f"domain: {concept.domain_id}",
            f"standard: {'yes' if concept.is_standard else 'no'}",
        ]
        if concept.invalid_reason:
            lines.append(f"invalid_reason: {concept.invalid_reason}")
        synonyms = self.vocabulary.synonyms(concept_id)
        if synonyms:
            lines.append(f"synonyms: {'; '.join(synonyms)}")
        relationships = self.vocabulary.relationships(concept_id)
        related_ids = {
            target
            for rel, target in relationships
            if rel in LOINC_AXES or rel == "Maps to"
        }
        names = self.vocabulary.by_id(related_ids)
        axes = {
            LOINC_AXES[rel]: names[target].concept_name
            for rel, target in relationships
            if rel in LOINC_AXES and target in names
        }
        if axes:
            lines.append(
                "LOINC axes: "
                + "; ".join(
                    f"{axis}={axes[axis]}"
                    for axis in LOINC_AXES.values()
                    if axis in axes
                )
            )
        maps_to = [
            names[target]
            for rel, target in relationships
            if rel == "Maps to" and target in names and target != concept_id
        ]
        for target in maps_to:
            lines.append(f"maps to: {Candidate.from_concept(target).line()}")
        return "\n".join(lines)

    def hierarchy(
        self,
        concept_id: int,
        direction: Literal["parents", "children"],
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[Candidate]:
        assert self.vocabulary is not None
        return self.vocabulary.neighbours(
            concept_id, direction, domain, vocabularies, limit=(top_k or self.top_k) * 2
        )


def in_scope(
    candidate: Candidate | Concept,
    domain: Optional[str],
    vocabularies: Optional[Sequence[str]],
) -> bool:
    """Whether a concept is a valid standard concept of the domain and vocabularies."""
    standard = (
        candidate.is_standard
        if isinstance(candidate, Concept)
        else bool(candidate.standard)
    )
    if not standard:
        return False
    if isinstance(candidate, Concept) and candidate.invalid_reason:
        return False
    if domain is not None and candidate.domain_id != domain:
        return False
    if vocabularies and candidate.vocabulary_id not in vocabularies:
        return False
    return True
