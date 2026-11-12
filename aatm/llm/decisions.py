"""Decision models: closed-set questions answered with probabilities.

A decision model does not write text. It reads a *state* (a text, or JSON) and
answers declared *questions*, each with a closed set of answers, and returns the
answer with a probability per option. Many pipeline steps are such decisions:
picking a candidate concept, labelling an attribute, checking that a mention is not
negated.

```python
from aatm.llm import Question, create_decision_model

model = create_decision_model("openrouter/typesafe/jev-1.13")
decision = await model.decide(
    {"text": "creatinine", "unit": "mg/dl"},
    [
        Question.choice(
            "specimen",
            "Which specimen was the test most likely done on?",
            {"serum": "Serum or plasma", "urine": "Urine", "unknown": "Not stated"},
        ),
        Question.predicate("is_lab", "Is the text the name of a lab test?"),
    ],
)
decision["specimen"].value, decision["specimen"].probabilities
decision["is_lab"].probability
```

Question kinds:

| Kind | Answers | `Answer.value` |
|---|---|---|
| ``choice`` | one of 2–255 named options | the option |
| ``score`` | an ordered scale of 2–10 levels, lowest first | the probability-weighted position, from 0 (first level) to n - 1; `Answer.label` is the most likely level |
| ``predicate`` | yes or no | ``True`` or ``False``; `Answer.probability` is the probability of yes |

Backends (see `create_decision_model`):

- ``openai/gpt-6-luna``: the OpenAI Decisions API.
- ``gemini/gemini-decision-1``: the Gemini Decisions API.
- ``openrouter/<model>``: OpenRouter's decisions endpoint, e.g. TypeSafe's Jev
  (``openrouter/typesafe/jev-1.13``).
- ``jev-compatible/<model>``: any endpoint with the same request shape as Jev's
  (TypeSafe directly, Eden AI, a self-hosted model), with ``base_url``.
- ``llm/<llm id>``: any `StructuredLLM` (e.g. ``llm/anthropic/claude-haiku-5-5``),
  through structured output. Its answers have **no probabilities**, so pipeline
  steps work offline or with any provider, but thresholds on probabilities do not
  apply.

Pin versions (``jev-1.13``, not ``jev-latest``): a moving alias changes the answers
behind cached results.

Calls go through the shared rate limiter and retries (`aatm.llm.limits`) and are
counted by `aatm.llm.track_usage`.

**Data governance:** decision backends send the state, e.g. a sentence of a clinical
note, to the provider. Check the provider's data processing terms and retention
before sending protected health information.
"""

import json
import logging
import re
from typing import (
    Any,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from aatm.llm.base import LLMResponseError, StructuredLLM

logger = logging.getLogger(__name__)

QuestionKind = Literal["choice", "score", "predicate"]

State = Union[str, Mapping[str, Any], Sequence[Any]]
"""What a decision is about: a text, a JSON object, or an array."""

#: Limits accepted by every backend (the OpenAI Decisions API is the strictest).
MAX_CHOICES = 255
MAX_LEVELS = 10
MAX_QUESTIONS = 400

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class DecisionError(LLMResponseError):
    """Raised when a decision backend returns no usable answer for a question."""


class DecisionSettings(BaseModel):
    """Request settings of decision backends.

    Attributes:
        timeout: Request timeout in seconds.
        provider_kwargs: Extra top-level fields sent as-is in the request body, e.g.
            ``{"prompt_cache_key": "..."}`` (OpenAI) or ``{"session_id": "..."}``
            (OpenRouter).
    """

    model_config = ConfigDict(extra="forbid")

    timeout: Optional[float] = Field(default=None, gt=0)
    provider_kwargs: dict[str, Any] = Field(default_factory=dict)


class Option(BaseModel):
    """One answer of a choice question, or one level of a score question.

    Attributes:
        value: The option, e.g. ``"serum"``, returned as `Answer.value` (choice) or
            `Answer.label` (score).
        description: What the option means, shown to the model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(min_length=1)
    description: Optional[str] = None


def _options(
    options: Union[Mapping[str, Optional[str]], Sequence[Union[str, Option]]],
) -> list[Option]:
    if isinstance(options, Mapping):
        return [Option(value=k, description=v) for k, v in options.items()]
    return [o if isinstance(o, Option) else Option(value=o) for o in options]


class Question(BaseModel):
    """A question with a closed set of answers.

    Build one with `Question.choice`, `Question.score` or `Question.predicate`.

    Attributes:
        name: Identifies the answer: a letter, then letters, digits or ``_``.
        kind: ``choice``, ``score`` or ``predicate``.
        instructions: The question, e.g. ``"Is the condition negated?"``.
        options: The options (choice) or the levels, lowest first (score). Empty for
            a predicate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: QuestionKind
    instructions: str = Field(min_length=1)
    options: tuple[Option, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> "Question":
        if not _NAME.match(self.name):
            raise ValueError(
                f"Invalid question name {self.name!r}: use a letter, then up to 63 "
                "letters, digits or '_'."
            )
        values = [o.value for o in self.options]
        if len(set(values)) != len(values):
            raise ValueError(f"Question {self.name!r} has duplicate options.")
        n = len(self.options)
        if self.kind == "choice" and not 2 <= n <= MAX_CHOICES:
            raise ValueError(
                f"Choice question {self.name!r} needs 2 to {MAX_CHOICES} options, "
                f"got {n}."
            )
        if self.kind == "score" and not 2 <= n <= MAX_LEVELS:
            raise ValueError(
                f"Score question {self.name!r} needs 2 to {MAX_LEVELS} levels, got {n}."
            )
        if self.kind == "predicate" and n:
            raise ValueError(f"Predicate question {self.name!r} takes no options.")
        return self

    @classmethod
    def choice(
        cls,
        name: str,
        instructions: str,
        options: Union[Mapping[str, Optional[str]], Sequence[Union[str, Option]]],
    ) -> "Question":
        """A question answered by one of `options`.

        Args:
            options: ``{value: description}``, or values, or `Option` objects.
        """
        return cls(
            name=name,
            kind="choice",
            instructions=instructions,
            options=tuple(_options(options)),
        )

    @classmethod
    def score(
        cls,
        name: str,
        instructions: str,
        levels: Union[Mapping[str, Optional[str]], Sequence[Union[str, Option]]],
    ) -> "Question":
        """A question answered on an ordered scale.

        Args:
            levels: The levels, **lowest first**, as for `choice`.
        """
        return cls(
            name=name,
            kind="score",
            instructions=instructions,
            options=tuple(_options(levels)),
        )

    @classmethod
    def predicate(cls, name: str, instructions: str) -> "Question":
        """A yes/no question."""
        return cls(name=name, kind="predicate", instructions=instructions)

    @property
    def values(self) -> list[str]:
        """The option values, or level labels, in order."""
        return [o.value for o in self.options]


class Answer(BaseModel):
    """The answer to one `Question`.

    Attributes:
        name: The question's name.
        kind: The question's kind.
        value: The option (choice), the probability-weighted position on the scale
            from 0 to n - 1 (score), or whether the statement holds (predicate).
            ``None`` when the model refused to answer.
        label: Score only: the most likely level.
        probabilities: Probability of each option or level (keys are the option
            values), or of ``"true"`` and ``"false"`` (predicate). Empty when the
            backend gives none (``llm/`` models).
        confidence: The backend's confidence in the answer, between 0 and 1, when it
            reports one.
        refused: Whether the model declined to answer.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: QuestionKind
    value: Union[bool, float, str, None]
    label: Optional[str] = None
    probabilities: dict[str, float] = {}
    confidence: Optional[float] = None
    refused: bool = False

    @property
    def probability(self) -> Optional[float]:
        """Predicate: the probability of yes. Choice and score: of the answer."""
        if self.kind == "predicate":
            return self.probabilities.get("true")
        key = self.label if self.kind == "score" else self.value
        return self.probabilities.get(str(key)) if key is not None else None


class Decision(BaseModel):
    """The answers to the questions of one `DecisionModel.decide` call.

    Attributes:
        model: The model that answered, as the provider reports it (the exact
            version when the provider gives one).
        answers: One answer per question, in the order of the questions.
    """

    model: str
    answers: list[Answer]

    def __getitem__(self, name: str) -> Answer:
        for answer in self.answers:
            if answer.name == name:
                return answer
        raise KeyError(name)


@runtime_checkable
class DecisionModel(Protocol):
    """A model that answers closed-set questions about a state."""

    model_id: str

    async def decide(self, state: State, questions: Sequence[Question]) -> Decision:
        """Answer `questions` about `state`.

        Raises:
            ValueError: If the questions are empty, too many or share a name.
            DecisionError: If the backend returns no usable answer for a question.
        """
        ...


def check_questions(questions: Sequence[Question]) -> None:
    """Raise `ValueError` unless there are 1 to `MAX_QUESTIONS` uniquely named."""
    if not questions:
        raise ValueError("A decision needs at least one question.")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(
            f"A decision takes at most {MAX_QUESTIONS} questions, got {len(questions)}."
        )
    names = [q.name for q in questions]
    if len(set(names)) != len(names):
        raise ValueError(f"Question names must be unique, got {names}.")


def state_text(state: State) -> str:
    """The state as text: strings as they are, anything else as JSON."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, default=str)


def score_label(
    question: Question, probabilities: Mapping[str, float], score: float
) -> str:
    """The most likely level, or the level nearest `score` without probabilities."""
    if probabilities:
        return max(question.values, key=lambda v: probabilities.get(v, 0.0))
    index = min(max(round(score), 0), len(question.options) - 1)
    return question.values[index]


def predicate_answer(
    question: Question, probability: float, confidence: Optional[float] = None
) -> Answer:
    """The answer to a predicate from the probability of yes."""
    return Answer(
        name=question.name,
        kind="predicate",
        value=probability >= 0.5,
        probabilities={"true": probability, "false": 1.0 - probability},
        confidence=confidence
        if confidence is not None
        else max(probability, 1.0 - probability),
    )


def refusal(question: Question) -> Answer:
    return Answer(name=question.name, kind=question.kind, value=None, refused=True)


def has_probabilities(model: DecisionModel) -> bool:
    """Whether the model's answers have probabilities (all but ``llm/`` models)."""
    return not model.model_id.startswith("llm/")


def model_identity(model: DecisionModel) -> dict[str, Any]:
    """What makes a decision model's answers differ, besides the questions.

    The model id, the server of jev-compatible and OpenAI-compatible endpoints, and
    extra request fields. The timeout is left out: it does not change the answers.
    """
    identity: dict[str, Any] = {"model_id": model.model_id}
    for name in ("url", "base_url"):
        value = getattr(model, name, None)
        if value is not None:
            identity[name] = str(value)
    client_url = getattr(getattr(model, "client", None), "base_url", None)
    if client_url is not None and "base_url" not in identity:
        identity["base_url"] = str(client_url)
    settings = getattr(model, "settings", None)
    extra = getattr(settings, "provider_kwargs", None)
    if extra:
        identity["provider_kwargs"] = {
            k: v if isinstance(v, (str, int, float, bool)) or v is None else repr(v)
            for k, v in sorted(extra.items())
        }
    return identity


def warn_if_unpinned(model_id: str) -> None:
    """Warn about moving aliases such as ``jev-latest``."""
    if re.search(r"(latest|preview)$", model_id):
        logger.warning(
            "%s is a moving alias: its answers can change between calls, and cached "
            "results no longer match them. Pin a version for reproducible runs.",
            model_id,
        )


# --- Any StructuredLLM as a decision model ---------------------------------------------

_LLM_SYSTEM = (
    "You answer questions about a state, e.g. a text or a JSON record. Answer every "
    "question with exactly one of its allowed answers, judging only from the state."
)


def _question_text(question: Question) -> str:
    lines = [f"## {question.name} ({question.kind})", question.instructions]
    if question.kind == "predicate":
        lines.append("Answer true or false.")
    else:
        if question.kind == "score":
            lines.append("Levels, from lowest to highest:")
        else:
            lines.append("Options:")
        for option in question.options:
            text = f"- {option.value}"
            if option.description:
                text += f": {option.description}"
            lines.append(text)
    return "\n".join(lines)


def _output_type(questions: Sequence[Question]) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for i, question in enumerate(questions):
        annotation: Any
        if question.kind == "predicate":
            annotation = bool
        else:
            annotation = Literal[tuple(question.values)]  # type: ignore[valid-type]
        fields[f"q{i}"] = (
            annotation,
            Field(alias=question.name, description=question.instructions),
        )
    return create_model(  # type: ignore[call-overload,no-any-return]
        "DecisionAnswers",
        __config__=ConfigDict(extra="forbid", populate_by_name=True),
        **fields,
    )


class LLMDecisionModel:
    """Any `StructuredLLM` used as a decision model, through structured output.

    The answers have the chosen value but **no probabilities and no confidence**: an
    LLM's verbal answer is not a probability. Use it to run pipeline steps without a
    decision provider, not to calibrate thresholds.
    """

    def __init__(self, llm: StructuredLLM):
        self.llm = llm
        self.model_id = f"llm/{llm.model_id}"

    async def decide(self, state: State, questions: Sequence[Question]) -> Decision:
        check_questions(questions)
        output_type = _output_type(questions)
        user = (
            "# State\n"
            + state_text(state)
            + "\n\n# Questions\n\n"
            + "\n\n".join(_question_text(q) for q in questions)
        )
        output = await self.llm.generate(
            system=_LLM_SYSTEM, user=user, output_type=output_type
        )
        answers: list[Answer] = []
        for i, question in enumerate(questions):
            value = getattr(output, f"q{i}")
            if question.kind == "score":
                position = question.values.index(value)
                answers.append(
                    Answer(
                        name=question.name,
                        kind="score",
                        value=float(position),
                        label=value,
                    )
                )
            else:
                answers.append(
                    Answer(name=question.name, kind=question.kind, value=value)
                )
        return Decision(model=self.llm.model_id, answers=answers)
