"""Coding extracted text with the terminology mapper.

`OmopTableBuilder` codes ``{from: ..., map: ...}`` sources through a `TextCoder`.
`TerminologyMapperCoder` adapts a configured `aatm.terminology_mapper.TerminologyMapper`
(translator, retriever, reranker and selector), so extraction and terminology mapping
share one pipeline:

```python
from aatm.terminology_mapper import TerminologyMapper
from aatm.omop import OmopTableBuilder

mapper = TerminologyMapper(retriever="text-embedding-3-small")
builder = OmopTableBuilder(definitions, mapper=mapper)  # wrapped automatically
```
"""

from typing import Any, Optional, Protocol, Sequence

from pydantic import BaseModel


class CodedText(BaseModel):
    """The standard concept a text was mapped to.

    Attributes:
        text: The text that was mapped.
        concept_id: The selected standard concept, or ``None`` when nothing was
            selected.
        concept_name: Its name.
        vocabulary_id: Its vocabulary, e.g. ``"LOINC"``.
        concept_code: Its code in that vocabulary.
        domain_id: Its domain, e.g. ``"Measurement"``.
        score: The mapper's confidence, from 0 to 1 (``1 - distance`` for the default
            selector), when available.
    """

    text: str
    concept_id: Optional[int] = None
    concept_name: Optional[str] = None
    vocabulary_id: Optional[str] = None
    concept_code: Optional[str] = None
    domain_id: Optional[str] = None
    score: Optional[float] = None


class TextCoder(Protocol):
    """Maps texts to standard concepts, optionally restricted to a domain and vocabularies."""

    def code(
        self,
        texts: Sequence[str],
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
    ) -> list[CodedText]:
        """Return one `CodedText` per text, in order."""
        ...


def retriever_filter(
    domain: Optional[str], vocabularies: Optional[Sequence[str]]
) -> Optional[dict[str, Any]]:
    """Build the ChromaDB metadata filter that restricts retrieval to a domain and vocabularies."""
    conditions: list[dict[str, Any]] = []
    if domain is not None:
        conditions.append({"std_domain_id": domain})
    if vocabularies:
        conditions.append({"std_vocabulary_id": {"$in": list(vocabularies)}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class TerminologyMapperCoder:
    """`TextCoder` that runs texts through a `TerminologyMapper`.

    Domain and vocabularies become a filter on the mapper's retriever, so only
    matching standard concepts are candidates.
    """

    def __init__(self, mapper: Any):
        self.mapper = mapper

    def code(
        self,
        texts: Sequence[str],
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
    ) -> list[CodedText]:
        if not texts:
            return []
        mapped = self.mapper.map(
            expressions=list(texts),
            return_as="mapped_source_concepts",
            save_to_disk=False,
            where=retriever_filter(domain, vocabularies),
        )
        return [
            CodedText(
                text=text,
                concept_id=int(m.target_concept_id) if m.target_concept_id else None,
                # MappedSourceConcept has no concept name; the builder adds it from
                # the vocabulary database.
                vocabulary_id=getattr(
                    m.target_vocabulary_id, "value", m.target_vocabulary_id
                ),
                concept_code=m.target_vocabulary_code,
                domain_id=m.domain_id,
                score=m.confidence_score,
            )
            for text, m in zip(texts, mapped)
        ]
