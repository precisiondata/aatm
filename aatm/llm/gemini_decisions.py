"""The Gemini Decisions API as a `DecisionModel`.

Google's Decisions API (``POST /v1beta/decisions``, model ``gemini-decision-1``)
takes the same questions as Jev (``noul``, ``choice``, ``score``, keyed by name) and
returns answers in the same shape, so it reuses `aatm.llm.jev`'s question and answer
helpers. The state goes in ``input``, as text (JSON objects are sent as JSON text).

Requests go through ``client.aio.decisions.create`` of ``google-genai`` 3.x, on the
``v1beta`` API. Its errors carry the HTTP code, so the shared limiter retries 429
and 5xx.

A prompt blocked by Gemini's safety filters (``prompt_feedback.block_reason``) has
no answers: every question is answered as refused, and a warning is logged.
"""

import logging
from typing import Any, Mapping, Optional, Sequence

from aatm.llm.base import PerEventLoop
from aatm.llm.decisions import (
    Decision,
    DecisionError,
    DecisionSettings,
    Question,
    State,
    check_questions,
    refusal,
    state_text,
    warn_if_unpinned,
)
from aatm.llm.gemini import gemini_client
from aatm.llm.jev import jev_answer, jev_question
from aatm.llm.limits import estimate_tokens, get_rate_limiter
from aatm.llm.usage import record_usage

logger = logging.getLogger(__name__)


class GeminiDecisionModel:
    """Gemini decision models, e.g. ``gemini-decision-1``.

    The API key is read from ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``) unless passed.

    Args:
        model: The model, e.g. ``"gemini-decision-1"``.
        api_key: API key.
        settings: Timeout and extra request fields.
        client: A ``google.genai.Client`` to use, on every event loop, instead of
            one per event loop.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        settings: Optional[DecisionSettings] = None,
        client: Any = None,
    ):
        self.model = model
        self.model_id = f"gemini/{model}"
        self.settings = settings or DecisionSettings()
        # One client per event loop: the SDK binds its connection pool to the first.
        self._clients: PerEventLoop[Any] = PerEventLoop(
            lambda: gemini_client(api_key, timeout=self.settings.timeout)
        )
        if client is not None:
            self._clients.set(client)
        warn_if_unpinned(self.model_id)

    @property
    def client(self) -> Any:
        """The client of the running event loop (or the one given)."""
        return self._clients.get()

    def request_body(
        self, state: State, questions: Sequence[Question]
    ) -> dict[str, Any]:
        """The arguments of ``decisions.create``."""
        return {
            "model": self.model,
            "input": state_text(state),
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

    async def _decide(
        self, body: dict[str, Any], questions: Sequence[Question]
    ) -> Decision:
        response = await self.client.aio.decisions.create(**body)
        data = response.model_dump() if hasattr(response, "model_dump") else response
        if not isinstance(data, Mapping):
            raise DecisionError(f"{self.model_id} returned no decision: {data!r}")
        usage = data.get("usage") or {}
        record_usage(
            usage.get("total_input_tokens"),
            usage.get("total_output_tokens"),
            model_id=self.model_id,
        )
        answers = data.get("answers")
        if not answers:
            feedback = data.get("prompt_feedback") or {}
            reason = feedback.get("block_reason")
            if reason is None:
                raise DecisionError(f"{self.model_id} returned no answers: {data!r}")
            logger.warning(
                "%s blocked the prompt (%s%s): every question is answered as refused.",
                self.model_id,
                reason,
                f": {feedback['block_reason_message']}"
                if feedback.get("block_reason_message")
                else "",
            )
            return Decision(model=self.model, answers=[refusal(q) for q in questions])
        return Decision(
            model=self.model,
            answers=[
                jev_answer(self.model_id, q, answers.get(q.name)) for q in questions
            ],
        )
