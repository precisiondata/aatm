"""Agentic concept search: an LLM agent that searches the vocabularies for the right code.

One retrieval on a short span often misses the right concept. `ConceptSearchAgent`
gets the text and its context and searches with tools (semantic search, lexical
search, code lookup, concept details, hierarchy) until it finds the standard concept
or concludes there is none. It is available in both pipelines:

- **Notes to OMOP tables**: ``OmopTableBuilder(definitions, mapper=AgenticCoder(...))``
  (`aatm.omop.AgenticCoder`). The builder passes each text's related values and note
  line.
- **Terminology mapping**: ``TerminologyMapper(selector=AgenticSelector(...))``. The
  agent starts from the mapper's retrieved candidates, with each source concept's code
  and vocabulary as context.

See `aatm.concept_search.agent` for the confidence scale of the scores.
"""

from aatm.concept_search.agent import (
    CONFIDENCE_SCORES,
    AgentStep,
    CodingContext,
    ConceptSearchAgent,
    ConceptSearchResult,
)
from aatm.concept_search.tools import Candidate, ConceptSearchTools, VocabularySearch

__all__ = [
    "CONFIDENCE_SCORES",
    "AgentStep",
    "AgenticSelector",
    "Candidate",
    "CodingContext",
    "ConceptSearchAgent",
    "ConceptSearchResult",
    "ConceptSearchTools",
    "VocabularySearch",
]


def __getattr__(name: str):
    # The selector imports the selector stack (OpenAI, Gemini clients): load it on use.
    if name == "AgenticSelector":
        from aatm.concept_search.selector import AgenticSelector

        return AgenticSelector
    raise AttributeError(f"module 'aatm.concept_search' has no attribute {name!r}")
