"""Provider-neutral interface for structured-output LLM calls."""

from typing import Any, Literal, Optional, Protocol, TypeVar, runtime_checkable

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
