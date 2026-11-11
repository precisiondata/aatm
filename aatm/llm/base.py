"""Provider-neutral interface for structured-output LLM calls."""

import asyncio
import weakref
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound=BaseModel)

ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh", "max"]


class LLMSettings(BaseModel):
    """Generation settings shared by every provider.

    Settings left as ``None`` are not sent, so each provider's own default applies.
    Not every provider accepts every setting; see the notes on each field.

    Attributes:
        temperature: Sampling temperature. Ignored, with a warning, for Anthropic,
            whose current models do not accept it.
        max_tokens: Maximum number of output tokens. Anthropic requires a value and
            defaults to 16000 when this is unset.
        reasoning_effort: Reasoning effort for models that support it. Mapped to
            ``reasoning.effort`` (OpenAI), ``output_config.effort`` (Anthropic) and
            ``thinking_level`` (Gemini 3 and later). Not every model accepts every level.
        timeout: Request timeout in seconds.
        provider_kwargs: Extra keyword arguments passed as-is to the provider's
            request call, for options not covered above.
    """

    model_config = ConfigDict(extra="forbid")

    temperature: Optional[float] = None
    max_tokens: Optional[int] = Field(default=None, gt=0)
    reasoning_effort: Optional[ReasoningEffort] = None
    timeout: Optional[float] = Field(default=None, gt=0)
    provider_kwargs: dict[str, Any] = Field(default_factory=dict)


C = TypeVar("C")


class PerEventLoop(Generic[C]):
    """One object per asyncio event loop, e.g. an async SDK client.

    Async SDK clients (``openai`` 3.x, ``google-genai``'s Decisions API) keep a
    connection pool bound to the event loop they first ran on, and fail with
    "Event loop is closed" on the next one. aatm calls models from successive
    ``asyncio.run`` calls (synchronous APIs, notebooks), so adapters build one
    client per loop.

    Args:
        factory: Builds a new object.
    """

    def __init__(self, factory: Callable[[], C]):
        self._factory = factory
        self._by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, C] = (
            weakref.WeakKeyDictionary()
        )
        self._outside_loop: Optional[C] = None
        self._fixed: Optional[C] = None

    def get(self) -> C:
        """The object of the running loop; a shared one outside any loop."""
        if self._fixed is not None:
            return self._fixed
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if self._outside_loop is None:
                self._outside_loop = self._factory()
            return self._outside_loop
        value = self._by_loop.get(loop)
        if value is None:
            value = self._by_loop[loop] = self._factory()
        return value

    def set(self, value: C) -> None:
        """Use `value` on every loop from now on (e.g. a fake client in tests)."""
        self._fixed = value


class LLMResponseError(RuntimeError):
    """Raised when a provider returns no usable structured output."""


@runtime_checkable
class StructuredLLM(Protocol):
    """An LLM that answers one prompt with an instance of a Pydantic model."""

    model_id: str

    async def generate(self, *, system: str, user: str, output_type: type[T]) -> T:
        """Send one system prompt and one user message; return the parsed output.

        Raises:
            LLMResponseError: If the response is empty, refused or cannot be parsed.
        """
        ...
