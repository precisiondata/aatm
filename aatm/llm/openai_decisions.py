"""The OpenAI Decisions API (``POST /v1/decisions``) as a `DecisionModel`.

Requests go through ``client.decisions.create`` (``openai`` 3.x). Its errors carry
the HTTP status, so the shared limiter retries 429 and 5xx.
"""

from typing import Any, Optional, Sequence

from openai import AsyncOpenAI

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


class OpenAIDecisionModel:
    """OpenAI models through the Decisions API, e.g. ``gpt-6-luna``.

    Answers come with probabilities per option and a confidence. Text states are
    sent as text; JSON objects, and arrays of objects, as ``input_json``.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        settings: Optional[DecisionSettings] = None,
        client: Optional[AsyncOpenAI] = None,
    ):
        self.model = model
        self.model_id = f"openai/{model}"
        self.settings = settings or DecisionSettings()
        # Retries are handled by the shared limiter (`aatm.llm.limits`).
        self.client = client or AsyncOpenAI(
            api_key=api_key, base_url=base_url, max_retries=0
        )
        warn_if_unpinned(self.model_id)

    def request_body(
        self, state: State, questions: Sequence[Question]
    ) -> dict[str, Any]:
        """The JSON body of the request."""
        input_: Any
        if isinstance(state, dict) or (
            isinstance(state, list)
            and state
            and all(isinstance(s, dict) for s in state)
        ):
            content = [{"type": "input_json", "data": state}]
            input_ = [{"role": "user", "content": content}]
        else:
            input_ = state_text(state)
        return {
            "model": self.model,
            "input": input_,
            "questions": [_question(q) for q in questions],
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

    async def _post(self, body: dict[str, Any]) -> Any:
        """Send the request with ``client.decisions.create``; return a dict."""
        kwargs = {k: body[k] for k in _SDK_PARAMS if k in body}
        extra = {k: v for k, v in body.items() if k not in _SDK_PARAMS}
        if extra:
            kwargs["extra_body"] = extra
        if self.settings.timeout is not None:
            kwargs["timeout"] = self.settings.timeout
        decision = await self.client.decisions.create(**kwargs)
        return decision.model_dump()

    async def _decide(
        self, body: dict[str, Any], questions: Sequence[Question]
    ) -> Decision:
        response = await self._post(body)
        if not isinstance(response, dict):
            raise DecisionError(f"{self.model_id} returned no decision: {response!r}")
        usage = response.get("usage") or {}
        record_usage(
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            model_id=self.model_id,
        )
        answers = response.get("answers") or []
        if len(answers) != len(questions):
            raise DecisionError(
                f"{self.model_id} returned {len(answers)} answers for "
                f"{len(questions)} questions."
            )
        return Decision(
            model=response.get("model") or self.model,
            answers=[_answer(self.model_id, q, a) for q, a in zip(questions, answers)],
        )


#: Keyword arguments of ``client.decisions.create`` (openai 3.x); other body fields
#: are sent as ``extra_body``.
_SDK_PARAMS = (
    "model",
    "input",
    "questions",
    "prompt_cache_key",
    "prompt_cache_options",
    "safety_identifier",
)


def _described(key: str, value: str, description: Optional[str]) -> dict[str, Any]:
    return {key: value, **({"description": description} if description else {})}


def _question(question: Question) -> dict[str, Any]:
    body: dict[str, Any] = {
        "type": question.kind,
        "name": question.name,
        "instructions": question.instructions,
    }
    if question.kind == "choice":
        body["choices"] = [
            _described("value", o.value, o.description) for o in question.options
        ]
    elif question.kind == "score":
        body["levels"] = [
            _described("label", o.value, o.description) for o in question.options
        ]
    return body


def _answer(model_id: str, question: Question, raw: Any) -> Answer:
    kind = raw.get("type") if isinstance(raw, dict) else None
    if kind == "refusal":
        return refusal(question)
    if kind != question.kind:
        raise DecisionError(
            f"{model_id} answered question {question.name!r} ({question.kind}) "
            f"with {raw!r}."
        )
    try:
        if kind == "predicate":
            return predicate_answer(question, float(raw["probability"]))
        if kind == "choice":
            return Answer(
                name=question.name,
                kind="choice",
                value=str(raw["choice"]),
                probabilities={
                    str(p["value"]): float(p["probability"])
                    for p in raw.get("probabilities") or []
                },
                confidence=raw.get("confidence"),
            )
        levels = question.values
        probabilities: dict[str, float] = {}
        for p in raw.get("probabilities") or []:
            label = p.get("label")
            if label not in levels:
                label = levels[int(p["value"])]
            probabilities[str(label)] = float(p["probability"])
        score = float(raw["score"])
        return Answer(
            name=question.name,
            kind="score",
            value=score,
            label=score_label(question, probabilities, score),
            probabilities=probabilities,
            confidence=raw.get("confidence"),
        )
    except (KeyError, TypeError, ValueError, IndexError) as e:
        raise DecisionError(
            f"{model_id} returned a malformed answer for {question.name!r}: {raw!r}"
        ) from e
