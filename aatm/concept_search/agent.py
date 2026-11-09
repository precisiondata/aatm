"""An LLM agent that searches the vocabularies for the standard concept of a text.

A single retrieval on a short span often misses the right concept: "creatinine"
says nothing of the specimen, and "Hb" or "potássio" need expanding or translating
before an embedding search finds them. The agent gets the text **and its context**
(related values such as the unit, the sentence of the note, the target column) and
searches iteratively with tools until it finds the concept or concludes there is none:

```python
from aatm.concept_search import CodingContext, ConceptSearchAgent, ConceptSearchTools
from aatm.registries.retrievers import load_retriever

agent = ConceptSearchAgent(
    "openai/gpt-5-mini",
    ConceptSearchTools(load_retriever("text-embedding-3-small"), vocabulary=".aatm/omop.db"),
)
context = CodingContext(text="creatinine", related={"unit": "mg/dl", "value": "1.4"})
result = await agent.search(context, domain="Measurement")
result.concept, result.score, result.rationale, result.trace
```

The agent is used by `aatm.omop.AgenticCoder` (extraction to OMOP tables) and by
`aatm.concept_search.AgenticSelector` (terminology mapping with `TerminologyMapper`).

The loop is provider-agnostic: at each step the model answers with one structured
`AgentStep` (an action and its arguments), so any `aatm.llm` model works, without
native tool calling.

## Confidence scale

The agent reports a confidence level, stored as a score:

| Level | Score | Meaning |
|---|---|---|
| ``high`` | 0.9 | The concept matches the text and everything the context states (e.g. specimen and unit). |
| ``medium`` | 0.6 | The right concept, but a detail the context does not state was assumed (e.g. serum for a chemistry test without specimen). |
| ``low`` | 0.3 | The closest concept available, with a known mismatch or real ambiguity. |

When the tool budget runs out, the agent must answer with what it has seen, and the
score is capped at ``low``. ``min_score: 0.5`` therefore keeps ``high`` and ``medium``.
"""

import asyncio
import json
import time
from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from aatm.concept_search.tools import Candidate, ConceptSearchTools, in_scope
from aatm.llm import LLMSettings, StructuredLLM, create_llm
from aatm.llm.usage import track_usage
from aatm.logs import get_logger

logger = get_logger(__name__)

Confidence = Literal["high", "medium", "low"]

CONFIDENCE_SCORES: dict[str, float] = {"high": 0.9, "medium": 0.6, "low": 0.3}
"""Score stored for each confidence level. See the module docstring."""

Action = Literal[
    "semantic_search",
    "lexical_search",
    "code_lookup",
    "concept_details",
    "hierarchy",
    "finish",
]


class CodingContext(BaseModel):
    """A text to code and what is known around it.

    Attributes:
        text: The text to code, e.g. ``"creatinine"``.
        related: Related values, e.g. ``{"lab_value": "1.4", "lab_unit": "mg/dl"}``
            from the children of an extracted mention, or ``{"source_code": "I63"}``
            for a source concept.
        sentence: The sentence or line of the note the text comes from.
        target: What the concept is for, e.g. ``"measurement.measurement_concept_id"``.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    related: dict[str, str] = Field(default_factory=dict)
    sentence: Optional[str] = None
    target: Optional[str] = None

    def cache_key(self) -> str:
        """Identical contexts share a key, so they are searched once."""
        return json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False)

    def describe(self) -> str:
        lines = [f"Text to code: {self.text!r}"]
        if self.related:
            lines.append(
                "Related values: "
                + "; ".join(f"{k}={v!r}" for k, v in self.related.items())
            )
        if self.sentence and self.sentence.strip() != self.text:
            lines.append(f"Sentence: {self.sentence.strip()!r}")
        if self.target:
            lines.append(f"Target: {self.target}")
        return "\n".join(lines)


class AgentStep(BaseModel):
    """One step of the agent: the next action and its arguments.

    Every field is required (``null`` when unused) so the schema is accepted by
    every provider's structured-output mode.
    """

    reasoning: str = Field(
        description="One or two sentences: what the last results showed and why "
        "this action is next."
    )
    action: Action
    query: Optional[str] = Field(
        description="Search text for semantic_search and lexical_search, or the "
        "code for code_lookup. null otherwise."
    )
    concept_id: Optional[int] = Field(
        description="Concept id for concept_details, hierarchy and finish. null "
        "otherwise, or for finish when no concept fits."
    )
    direction: Optional[Literal["parents", "children"]] = Field(
        description="For hierarchy: parents (more general) or children (more "
        "specific). null otherwise."
    )
    confidence: Optional[Confidence] = Field(
        description="For finish: high, medium or low (see the scale). null otherwise."
    )


class ConceptSearchResult(BaseModel):
    """What the agent found for one text.

    Attributes:
        concept: The chosen standard concept, or ``None``.
        confidence: The agent's confidence level.
        score: The score of that level (`CONFIDENCE_SCORES`).
        rationale: The agent's reason for its answer.
        trace: The actions taken, with their arguments and results, for review.
        tool_calls: Number of tool calls, the initial search included.
        llm_calls: Number of LLM calls.
        input_tokens: Prompt tokens of those calls, when the provider reports them.
        output_tokens: Output tokens of those calls.
        seconds: Wall-clock time of the search.
        budget_exhausted: Whether the tool budget ran out before the agent finished.
        error: The error that stopped the search, if any.
    """

    concept: Optional[Candidate] = None
    confidence: Optional[Confidence] = None
    score: Optional[float] = None
    rationale: Optional[str] = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    budget_exhausted: bool = False
    error: Optional[str] = None

    def usage(self) -> dict[str, Any]:
        """The cost of the search, for reports."""
        return {
            "tool_calls": self.tool_calls,
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "seconds": round(self.seconds, 2),
            "budget_exhausted": self.budget_exhausted,
        }


SYSTEM_PROMPT = """\
You code clinical text to a standard concept of the OMOP vocabularies. You search the \
vocabularies with tools, one action per turn, until you find the standard concept that \
matches the text in its context, or conclude that there is none.

Answer: a standard concept {scope}. Only finish with a concept id that a tool returned \
in this conversation; tool results only list concepts that qualify, unless marked \
non-standard.

How to search:
- Write your own queries: expand abbreviations ("Hb" -> "hemoglobin", "K" -> \
"potassium"), translate non-English terms ("potássio" -> "potassium"), rephrase, and \
add what the context says (specimen, unit, timing), e.g. "creatinine serum mass volume".
- Embedding (semantic) search can miss exact names; lexical search finds names and \
synonyms containing every word of the query. Use few, distinctive words for it.
- Use the context to tell variants apart: the specimen (serum or plasma, blood, urine, \
CSF), the property implied by the unit (mg/dL: mass/volume; mmol/L: moles/volume; %: \
fraction or ratio; a ratio unit: ratio), the timing (fasting, 24 hour) and the method.
- When the context does not say, prefer the usual clinical variant (serum or plasma \
for chemistry, blood for hematology) and say so in your reasoning.
- Check concept_details (LOINC axes: component, property, time, system, scale, \
method) when variants look alike. Use hierarchy to move to a more general or more \
specific concept.
- Prefer a specific concept that matches the context over a general one, but do not \
invent details the context does not support.

Actions (fill only the fields an action uses, null for the others):
{actions}
- finish: concept_id (or null if no concept fits) and confidence.

Confidence scale for finish:
- high: the concept matches the text and everything the context states.
- medium: the right concept, but a detail the context does not state was assumed.
- low: the closest concept available, with a known mismatch or real ambiguity.
"""

ACTION_HELP = {
    "semantic_search": "- semantic_search: query. Embedding search over concept names and synonyms.",
    "lexical_search": "- lexical_search: query. Names and synonyms containing every word of the query.",
    "code_lookup": "- code_lookup: query = a code, e.g. '2160-0' or 'I63'. Finds it in every vocabulary, with the standard concepts it maps to.",
    "concept_details": "- concept_details: concept_id. Domain, synonyms, LOINC axes, 'Maps to'.",
    "hierarchy": "- hierarchy: concept_id and direction (parents or children).",
}


def _scope_text(domain: Optional[str], vocabularies: Optional[Sequence[str]]) -> str:
    parts = []
    if domain:
        parts.append(f"of the {domain} domain")
    if vocabularies:
        parts.append(f"from {', '.join(vocabularies)}")
    return " ".join(parts) if parts else "of any domain and vocabulary"


class ConceptSearchAgent:
    """Searches the vocabularies for the standard concept of a text, with tools.

    Args:
        model: An `aatm.llm` model id such as ``"openai/gpt-5-mini"``, or a
            `aatm.llm.StructuredLLM`.
        tools: The search tools.
        max_tool_calls: Tool calls the agent may make per text, after the initial
            semantic search. When they run out, the agent must answer with what it
            has seen, and its confidence is capped at ``low``.
        settings: Generation settings for a model given by id. For more
            reproducible runs, set ``temperature=0`` on models that accept it
            (reasoning models such as GPT-5 reject it); the trace makes runs
            comparable either way.
        top_k: Results per search.
    """

    def __init__(
        self,
        model: str | StructuredLLM,
        tools: ConceptSearchTools,
        max_tool_calls: int = 8,
        settings: Optional[LLMSettings | dict[str, Any]] = None,
        top_k: int = 10,
    ):
        if max_tool_calls < 0:
            raise ValueError("max_tool_calls must be >= 0")
        if isinstance(model, str):
            model = create_llm(model, settings=settings)
        self.llm: StructuredLLM = model
        self.tools = tools
        self.max_tool_calls = max_tool_calls
        self.top_k = top_k

    async def search(
        self,
        context: CodingContext,
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
        initial_candidates: Optional[Sequence[Candidate]] = None,
    ) -> ConceptSearchResult:
        """Find the standard concept of `context.text` within a domain and vocabularies.

        Args:
            context: The text and its context.
            domain: Only concepts of this domain qualify.
            vocabularies: Only concepts of these vocabularies qualify.
            initial_candidates: Candidates already retrieved for the text, e.g. by a
                `TerminologyMapper`. When ``None``, the agent starts with a semantic
                search on the text (or a lexical search without a retriever).

        Returns:
            The result. Errors from the model are caught and reported in
            ``error``, with no concept.
        """
        start = time.perf_counter()
        result = ConceptSearchResult()
        with track_usage() as usage:
            try:
                await self._run(
                    context, domain, vocabularies, initial_candidates, result
                )
            except Exception as e:  # noqa: BLE001 - one text must not stop a batch
                logger.warning("Concept search failed for %r: %s", context.text, e)
                result.concept = None
                result.confidence = None
                result.score = None
                result.error = f"{type(e).__name__}: {e}"
                result.rationale = f"Search failed: {result.error}"
        result.llm_calls = usage.calls
        result.input_tokens = usage.input_tokens
        result.output_tokens = usage.output_tokens
        result.seconds = time.perf_counter() - start
        return result

    async def _run(
        self,
        context: CodingContext,
        domain: Optional[str],
        vocabularies: Optional[Sequence[str]],
        initial_candidates: Optional[Sequence[Candidate]],
        result: ConceptSearchResult,
    ) -> None:
        seen: dict[int, Candidate] = {}
        history: list[str] = []

        def observe(candidates: Sequence[Candidate]) -> str:
            for c in candidates:
                if c.standard is not False and in_scope(c, domain, vocabularies):
                    seen.setdefault(c.concept_id, c)
            if not candidates:
                return "No results."
            return "\n".join(c.line() for c in candidates)

        # Step 0: the candidates of a plain retrieval, so easy texts need one LLM call.
        if initial_candidates is not None:
            first = list(initial_candidates)
            label = f"retrieved candidates for {context.text!r}"
            result.trace.append(
                {"action": "retrieved", "results": [c.short() for c in first]}
            )
        else:
            action = (
                "semantic_search"
                if self.tools.retriever is not None
                else "lexical_search"
            )
            first = await self._tool(action, context.text, domain, vocabularies)
            result.tool_calls += 1
            label = f"{action} {context.text!r}"
            result.trace.append(
                {
                    "action": action,
                    "query": context.text,
                    "results": [c.short() for c in first],
                }
            )
        history.append(f"[{label}]\n{observe(first)}")

        system = SYSTEM_PROMPT.format(
            scope=_scope_text(domain, vocabularies),
            actions="\n".join(ACTION_HELP[a] for a in self.tools.actions),
        )
        calls_left = self.max_tool_calls
        retries_left = 2  # invalid finishes or unavailable actions
        while True:
            forced = calls_left <= 0
            user = self._prompt(context, history, calls_left)
            step = await self.llm.generate(
                system=system, user=user, output_type=AgentStep
            )
            entry: dict[str, Any] = {"action": step.action, "reasoning": step.reasoning}

            if step.action == "finish" or forced:
                if step.action != "finish":
                    # The model ignored the instruction to finish: take nothing.
                    entry["error"] = "budget exhausted"
                    result.trace.append(entry)
                    result.budget_exhausted = True
                    result.rationale = "Tool budget exhausted before an answer."
                    return
                entry.update(concept_id=step.concept_id, confidence=step.confidence)
                problem = self._finish(step, seen, domain, vocabularies, result)
                if problem is None:
                    if forced:
                        result.budget_exhausted = True
                        if result.concept is not None:
                            result.confidence = "low"
                            result.score = CONFIDENCE_SCORES["low"]
                    result.trace.append(entry)
                    return
                entry["error"] = problem
                result.trace.append(entry)
                if forced or retries_left == 0:
                    result.concept = None
                    result.confidence = result.score = None
                    result.budget_exhausted = forced
                    result.rationale = f"No valid answer: {problem}"
                    return
                retries_left -= 1
                history.append(f"[finish {step.concept_id}]\nRejected: {problem}")
                continue

            if step.action not in self.tools.actions:
                problem = f"{step.action} is not available. Use one of {self.tools.actions} or finish."
            else:
                problem = self._missing_argument(step)
            if problem is not None:
                entry["error"] = problem
                result.trace.append(entry)
                if retries_left == 0:
                    result.rationale = f"No valid answer: {problem}"
                    return
                retries_left -= 1
                history.append(f"[{step.action}]\nError: {problem}")
                continue

            calls_left -= 1
            result.tool_calls += 1
            label, observation, summary = await self._execute(
                step, domain, vocabularies, observe
            )
            entry.update(summary)
            result.trace.append(entry)
            history.append(f"[{label}]\n{observation}")

    @staticmethod
    def _missing_argument(step: AgentStep) -> Optional[str]:
        if step.action in ("semantic_search", "lexical_search", "code_lookup"):
            if not (step.query or "").strip():
                return f"{step.action} needs a query."
        if step.action in ("concept_details", "hierarchy") and step.concept_id is None:
            return f"{step.action} needs a concept_id."
        if step.action == "hierarchy" and step.direction is None:
            return "hierarchy needs a direction: parents or children."
        return None

    async def _tool(
        self,
        action: str,
        query: str,
        domain: Optional[str],
        vocabularies: Optional[Sequence[str]],
    ) -> list[Candidate]:
        method = getattr(self.tools, action)
        # The tools block (SQLite, ChromaDB): run them off the event loop.
        return await asyncio.to_thread(method, query, domain, vocabularies, self.top_k)

    async def _execute(
        self,
        step: AgentStep,
        domain: Optional[str],
        vocabularies: Optional[Sequence[str]],
        observe: Any,
    ) -> tuple[str, str, dict[str, Any]]:
        if step.action in ("semantic_search", "lexical_search"):
            query = step.query.strip()  # type: ignore[union-attr]
            found = await self._tool(step.action, query, domain, vocabularies)
            return (
                f"{step.action} {query!r}",
                observe(found),
                {"query": query, "results": [c.short() for c in found]},
            )
        if step.action == "code_lookup":
            code = step.query.strip()  # type: ignore[union-attr]
            found = await asyncio.to_thread(
                self.tools.code_lookup, code, domain, vocabularies
            )
            return (
                f"code_lookup {code!r}",
                observe(found),
                {"query": code, "results": [c.short() for c in found]},
            )
        if step.action == "concept_details":
            assert step.concept_id is not None  # checked by _missing_argument
            details = await asyncio.to_thread(
                self.tools.concept_details, step.concept_id
            )
            return (
                f"concept_details {step.concept_id}",
                details,
                {"concept_id": step.concept_id},
            )
        assert step.action == "hierarchy"
        # Checked by _missing_argument.
        assert step.concept_id is not None and step.direction is not None
        found = await asyncio.to_thread(
            self.tools.hierarchy,
            step.concept_id,
            step.direction,
            domain,
            vocabularies,
            self.top_k,
        )
        return (
            f"hierarchy {step.direction} of {step.concept_id}",
            observe(found),
            {
                "concept_id": step.concept_id,
                "direction": step.direction,
                "results": [c.short() for c in found],
            },
        )

    def _finish(
        self,
        step: AgentStep,
        seen: dict[int, Candidate],
        domain: Optional[str],
        vocabularies: Optional[Sequence[str]],
        result: ConceptSearchResult,
    ) -> Optional[str]:
        """Record a finish step; return why it is rejected, or ``None``."""
        result.rationale = step.reasoning
        if step.concept_id is None:
            result.concept = None
            result.confidence = result.score = None
            return None
        if step.confidence is None:
            return "finish with a concept needs a confidence: high, medium or low."
        candidate = seen.get(step.concept_id)
        if candidate is None:
            # Not returned in scope by a tool: accept it only if the vocabulary
            # confirms it qualifies.
            vocabulary = self.tools.vocabulary
            concept = (
                vocabulary.by_id([step.concept_id]).get(step.concept_id)
                if vocabulary is not None
                else None
            )
            if concept is None or not in_scope(concept, domain, vocabularies):
                return (
                    f"{step.concept_id} is not a standard concept "
                    f"{_scope_text(domain, vocabularies)} returned by a tool."
                )
            candidate = Candidate.from_concept(concept)
        result.concept = candidate
        result.confidence = step.confidence
        result.score = CONFIDENCE_SCORES[step.confidence]
        return None

    @staticmethod
    def _prompt(context: CodingContext, history: list[str], calls_left: int) -> str:
        budget = (
            f"Tool calls left: {calls_left}."
            if calls_left > 0
            else "No tool calls left: you must finish now, with a concept from the "
            "results above or null."
        )
        return (
            f"{context.describe()}\n\n"
            "Results so far:\n\n"
            + "\n\n".join(history)
            + f"\n\n{budget} Choose the next action."
        )

    async def search_many(
        self,
        items: Sequence[tuple[CodingContext, Optional[Sequence[Candidate]]]],
        domain: Optional[str] = None,
        vocabularies: Optional[Sequence[str]] = None,
        max_concurrency: int = 8,
    ) -> list[ConceptSearchResult]:
        """Search several texts concurrently; identical contexts are searched once.

        Args:
            items: ``(context, initial_candidates)`` pairs.
            domain: See `search`.
            vocabularies: See `search`.
            max_concurrency: Searches running at the same time.

        Returns:
            One result per item, in order.
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        unique: dict[str, tuple[CodingContext, Optional[Sequence[Candidate]]]] = {}
        for context, initial in items:
            unique.setdefault(context.cache_key(), (context, initial))

        async def run(context: CodingContext, initial: Any) -> ConceptSearchResult:
            async with semaphore:
                return await self.search(context, domain, vocabularies, initial)

        keys = list(unique)
        results = await asyncio.gather(*(run(*unique[k]) for k in keys))
        by_key = dict(zip(keys, results))
        return [by_key[context.cache_key()] for context, _ in items]


def run_sync(coroutine: Any) -> Any:
    """Run a coroutine from synchronous code, even inside a running event loop.

    In a thread without a loop (a script, or ``asyncio.to_thread``) it uses
    ``asyncio.run``; inside a running loop (e.g. Jupyter) it runs in a new thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()
