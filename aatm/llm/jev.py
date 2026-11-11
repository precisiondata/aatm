"""Decision endpoints with Jev's request shape: a ``state`` and named ``questions``.

TypeSafe's Jev introduced the shape, and OpenRouter serves it at
``POST https://openrouter.ai/api/alpha/decisions`` (an alpha route: it may change).
Eden AI and self-hosted Jev-compatible models use the same body:

```json
{
  "model": "typesafe/jev-1.13",
  "state": {"text": "creatinine", "unit": "mg/dl"},
  "questions": {
    "is_lab": {"type": "noul", "instructions": "Is the text a lab test?"},
    "specimen": {"type": "choice", "instructions": "...",
                 "criteria": {"serum": "Serum or plasma", "urine": "Urine"}},
    "certainty": {"type": "score", "instructions": "...",
                  "criteria": ["low", "medium", "high"]}
  }
}
```

The answers are keyed by question name: ``noul`` (yes/no) gives a probability,
``choice`` a choice with ``probabilities`` and ``confidence``, ``score`` a
probability-weighted position with ``legend`` and ``probabilities``. ``usage`` has
the tokens and the ``cost`` in USD, which `aatm.llm.track_usage` records.
"""

import logging
from typing import Any, Mapping, Optional, Sequence

import httpx

from aatm.llm.decisions import (
    Answer,
    Decision,
    DecisionError,
    DecisionSettings,
    Question,
    State,
    check_questions,
    predicate_answer,
    refusal,
    score_label,
    state_text,
    warn_if_unpinned,
)
from aatm.llm.limits import estimate_tokens, get_rate_limiter
from aatm.llm.usage import record_usage

logger = logging.getLogger(__name__)

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

_KINDS = {"choice": "choice", "score": "score", "predicate": "noul"}

_WARNED_ALPHA: set[str] = set()


def warn_if_alpha(url: str) -> None:
    """Warn, once per URL, that an alpha endpoint may change without notice."""
    if "/alpha/" not in url or url in _WARNED_ALPHA:
        return
    _WARNED_ALPHA.add(url)
    logger.warning(
        "%s is an alpha endpoint: its request and response shapes, model ids and "
        "pricing may change without notice or deprecation period, and aatm may fail "
        "to parse its answers. Pin the aatm version and the model version for "
        "production runs.",
        url,
    )


class DecisionHTTPError(RuntimeError):
    """An HTTP error from a decision endpoint.

    It has ``status_code`` and ``response`` so the shared limiter
    (`aatm.llm.limits`) retries 429 and 5xx and honours ``Retry-After``.
    """

    def __init__(self, message: str, response: httpx.Response):
        super().__init__(message)
        self.response = response
        self.status_code = response.status_code


def _level_text(question: Question) -> list[str]:
    return [
        f"{o.value}: {o.description}" if o.description else o.value
        for o in question.options
    ]


def jev_question(question: Question) -> dict[str, Any]:
    """A question in Jev's shape (also the Gemini Decisions API's)."""
    body: dict[str, Any] = {
        "type": _KINDS[question.kind],
        "instructions": question.instructions,
    }
    if question.kind == "choice":
        body["criteria"] = {o.value: o.description or o.value for o in question.options}
    elif question.kind == "score":
        body["criteria"] = _level_text(question)
    return body


class JevDecisionModel:
    """A decision endpoint with Jev's request shape (OpenRouter, TypeSafe, Eden AI).

    Args:
        model: The model, e.g. ``"typesafe/jev-1.13"``. Pin a version: aliases such
            as ``jev-latest`` move.
        url: The full URL of the endpoint, e.g. `OPENROUTER_DECISIONS_URL`.
        api_key: Sent as a bearer token, when given.
        provider: The provider prefix of `model_id`, e.g. ``"openrouter"``.
        settings: Timeout and extra body fields.
        http_client: An ``httpx.AsyncClient`` to use instead of one per call.
    """

    def __init__(
        self,
        model: str,
        *,
        url: str,
        api_key: Optional[str] = None,
        provider: str = "jev-compatible",
        settings: Optional[DecisionSettings] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.model = model
        self.model_id = f"{provider}/{model}"
        self.url = url
        self.api_key = api_key
        self.settings = settings or DecisionSettings()
        self.http_client = http_client
        warn_if_unpinned(self.model_id)
        warn_if_alpha(url)

    def request_body(
        self, state: State, questions: Sequence[Question]
    ) -> dict[str, Any]:
        """The JSON body of the request."""
        return {
            "model": self.model,
            "state": state,
            "questions": {q.name: jev_question(q) for q in questions},
            **self.settings.provider_kwargs,
        }

    async def decide(self, state: State, questions: Sequence[Question]) -> Decision:
        check_questions(questions)
        body = self.request_body(state, questions)
        return await get_rate_limiter(self.model_id).run(
            lambda: self._decide(body, questions),
            estimated_tokens=estimate_tokens(
                state_text(state), *(q.instructions for q in questions)
            ),
        )

    async def _post(self, client: httpx.AsyncClient, body: dict[str, Any]) -> Any:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = await client.post(
            self.url, json=body, headers=headers, timeout=self.settings.timeout
        )
        if response.status_code >= 400:
            raise DecisionHTTPError(
                f"{self.model_id} decision request failed "
                f"({response.status_code}): {response.text[:500]}",
                response,
            )
        return response.json()

    async def _decide(
        self, body: dict[str, Any], questions: Sequence[Question]
    ) -> Decision:
        if self.http_client is not None:
            response = await self._post(self.http_client, body)
        else:
            async with httpx.AsyncClient() as client:
                response = await self._post(client, body)
        if not isinstance(response, dict):
            raise DecisionError(f"{self.model_id} returned no decision: {response!r}")
        usage = response.get("usage") or {}
        cost = usage.get("cost")
        record_usage(
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            model_id=self.model_id,
            cost=float(cost) if isinstance(cost, (int, float)) else None,
        )
        answers = response.get("answers")
        if not isinstance(answers, Mapping):
            raise DecisionError(f"{self.model_id} returned no answers: {response!r}")
        return Decision(
            model=response.get("model") or self.model,
            answers=[
                jev_answer(self.model_id, q, answers.get(q.name)) for q in questions
            ],
        )


def _level_of(question: Question, key: str) -> Optional[str]:
    """The level label of a probability key: an index, a label or a criterion."""
    levels = question.values
    if key in levels:
        return key
    texts = _level_text(question)
    if key in texts:
        return levels[texts.index(key)]
    if key.isdigit() and int(key) < len(levels):
        return levels[int(key)]
    return None


def jev_answer(model_id: str, question: Question, raw: Any) -> Answer:
    """Parse an answer in Jev's shape (also the Gemini Decisions API's).

    Raises:
        DecisionError: If the answer is missing, of the wrong type or malformed.
    """
    if not isinstance(raw, Mapping):
        raise DecisionError(f"{model_id} returned no answer for {question.name!r}.")
    kind = raw.get("type")
    if kind == "refusal":
        return refusal(question)
    if kind != _KINDS[question.kind]:
        raise DecisionError(
            f"{model_id} answered question {question.name!r} ({question.kind}) "
            f"with {dict(raw)!r}."
        )
    try:
        confidence = raw.get("confidence")
        if question.kind == "predicate":
            probability = raw["noul"] if "noul" in raw else raw["probability"]
            return predicate_answer(question, float(probability), confidence)
        if question.kind == "choice":
            return Answer(
                name=question.name,
                kind="choice",
                value=str(raw["choice"]),
                probabilities={
                    str(k): float(v)
                    for k, v in (raw.get("probabilities") or {}).items()
                },
                confidence=confidence,
            )
        probabilities: dict[str, float] = {}
        for key, value in (raw.get("probabilities") or {}).items():
            level = _level_of(question, str(key))
            if level is None:
                raise ValueError(f"unknown level {key!r}")
            probabilities[level] = float(value)
        score = float(raw["score"])
        return Answer(
            name=question.name,
            kind="score",
            value=score,
            label=score_label(question, probabilities, score),
            probabilities=probabilities,
            confidence=confidence,
        )
    except (KeyError, TypeError, ValueError) as e:
        raise DecisionError(
            f"{model_id} returned a malformed answer for {question.name!r}: {dict(raw)!r}"
        ) from e
