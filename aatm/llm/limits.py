"""Shared rate limits and retries for every LLM call.

Every call made through the adapters in `aatm.llm` goes through the `RateLimiter` of
its model, so all the components of a pipeline (extractor, concept search agent,
selectors...) share one budget per model. Limits are set in one place:

```python
from aatm.llm import set_rate_limits

set_rate_limits("*", max_concurrency=16)  # default for every model
set_rate_limits("anthropic", requests_per_minute=1000, tokens_per_minute=400_000)
set_rate_limits("openai/gpt-5-mini", max_concurrency=64, max_retries=8)
```

Limits are looked up by model id (``"provider/model"``), then provider, then ``"*"``.
Each model has its own limiter, even when its limits come from the provider entry.

Rate-limit errors (429), server errors (5xx), timeouts and connection errors are
retried with exponential backoff and full jitter, honouring ``Retry-After`` when the
provider sends it. A 429 also pauses every call to that model for the wait, so
concurrent calls don't keep hitting the limit. Other errors are raised at once.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import threading
import time
import weakref
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from aatm.llm.usage import track_usage

logger = logging.getLogger(__name__)

R = TypeVar("R")

WINDOW_SECONDS = 60.0
MAX_RETRY_AFTER_SECONDS = 600.0
_RETRYABLE_STATUS = {408, 409, 429}
_CONNECTION_ERROR_NAMES = {
    "APIConnectionError",  # openai, anthropic (APITimeoutError subclasses it)
    "TransportError",  # httpx (timeouts, network errors)
    "ClientConnectionError",  # aiohttp
    "ClientPayloadError",
    "ServerTimeoutError",
}


class RateLimits(BaseModel):
    """Concurrency, rate limits and retries for the calls to one model.

    Attributes:
        max_concurrency: Calls in flight at the same time. ``None`` for no cap.
        requests_per_minute: Calls started in any 60-second window. ``None`` for no
            limit.
        tokens_per_minute: Tokens (input and output) in any 60-second window. A call
            reserves an estimate of its input tokens before it starts; the provider's
            count replaces the estimate when it returns. ``None`` for no limit.
        max_retries: Retries of a call after a rate-limit or transient error.
        initial_backoff: Upper bound in seconds of the first retry's random wait;
            it doubles with each retry.
        max_backoff: Cap in seconds of the random wait between retries.
    """

    model_config = ConfigDict(extra="forbid")

    max_concurrency: Optional[int] = Field(default=16, gt=0)
    requests_per_minute: Optional[float] = Field(default=None, gt=0)
    tokens_per_minute: Optional[float] = Field(default=None, gt=0)
    max_retries: int = Field(default=6, ge=0)
    initial_backoff: float = Field(default=1.0, gt=0)
    max_backoff: float = Field(default=60.0, gt=0)


def estimate_tokens(*texts: str) -> int:
    """A rough token count of prompt texts (about 4 characters per token)."""
    return sum(len(t) for t in texts) // 4 + 1


class _Slot:
    """A started call: its token reservation, corrected once usage is known."""

    def __init__(self, limiter: RateLimiter, entry: Optional[list[float]]):
        self._limiter = limiter
        self._entry = entry

    def record_tokens(self, tokens: Optional[int]) -> None:
        if self._entry is not None and tokens:
            with self._limiter._lock:
                self._entry[1] = float(tokens)


class RateLimiter:
    """Concurrency cap, sliding-window rate limits and retries for one model.

    Get one with `get_rate_limiter`; its limits follow `set_rate_limits`. The windows
    are shared by every event loop; the concurrency cap applies per event loop.
    """

    def __init__(self, model_id: str, limits: RateLimits):
        self.model_id = model_id
        self.limits = limits
        self._lock = threading.Lock()
        self._requests: deque[float] = deque()
        self._tokens: deque[list[float]] = deque()  # [start time, tokens]
        self._paused_until = 0.0
        self._loops: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, _LoopState
        ] = weakref.WeakKeyDictionary()

    # --- waiting ------------------------------------------------------------------

    def pause(self, seconds: float) -> None:
        """Hold every new call to this model for `seconds`, e.g. after a 429."""
        with self._lock:
            self._paused_until = max(self._paused_until, time.monotonic() + seconds)

    def _wait_seconds(self, now: float, tokens: int) -> float:
        """Seconds to wait before a call may start; 0 reserves it. Lock held."""
        limits = self.limits
        wait = self._paused_until - now
        horizon = now - WINDOW_SECONDS
        while self._requests and self._requests[0] <= horizon:
            self._requests.popleft()
        while self._tokens and self._tokens[0][0] <= horizon:
            self._tokens.popleft()
        rpm = limits.requests_per_minute
        if rpm is not None and len(self._requests) >= rpm:
            index = int(len(self._requests) - rpm)
            wait = max(wait, self._requests[index] + WINDOW_SECONDS - now)
        tpm = limits.tokens_per_minute
        if tpm is not None and self._tokens:
            excess = sum(t for _, t in self._tokens) + tokens - tpm
            if excess > 0:
                # Wait until enough of the oldest reservations leave the window.
                freed = 0.0
                for started, used in self._tokens:
                    freed += used
                    if freed >= excess:
                        break
                wait = max(wait, started + WINDOW_SECONDS - now)
        return wait

    async def _reserve(self, tokens: int) -> Optional[list[float]]:
        while True:
            with self._lock:
                now = time.monotonic()
                wait = self._wait_seconds(now, tokens)
                if wait <= 0:
                    if self.limits.requests_per_minute is not None:
                        self._requests.append(now)
                    if self.limits.tokens_per_minute is None:
                        return None
                    entry = [now, float(tokens)]
                    self._tokens.append(entry)
                    return entry
            await asyncio.sleep(wait)

    def _loop_state(self) -> _LoopState:
        loop = asyncio.get_running_loop()
        state = self._loops.get(loop)
        if state is None:
            state = self._loops[loop] = _LoopState()
        return state

    @asynccontextmanager
    async def slot(self, estimated_tokens: int = 0) -> AsyncIterator[_Slot]:
        """Wait for a concurrency slot and for room in the rate windows."""
        state = self._loop_state()
        async with state.condition:
            await state.condition.wait_for(
                lambda: (
                    self.limits.max_concurrency is None
                    or state.active < self.limits.max_concurrency
                )
            )
            state.active += 1
        try:
            entry = await self._reserve(estimated_tokens)
            yield _Slot(self, entry)
        finally:
            async with state.condition:
                state.active -= 1
                state.condition.notify_all()

    # --- calling ------------------------------------------------------------------

    def backoff(self, attempt: int) -> float:
        """Random wait before retry number `attempt` (0-based): full jitter."""
        cap = min(self.limits.max_backoff, self.limits.initial_backoff * 2**attempt)
        return random.uniform(0, cap)

    async def run(
        self, call: Callable[[], Awaitable[R]], *, estimated_tokens: int = 0
    ) -> R:
        """Run `call` within the limits, retrying rate-limit and transient errors.

        Raises:
            Exception: The last error, when it is not retryable or retries ran out.
        """
        attempt = 0
        while True:
            async with self.slot(estimated_tokens) as slot:
                try:
                    with track_usage() as usage:
                        result = await call()
                    slot.record_tokens(usage.input_tokens + usage.output_tokens)
                    return result
                except Exception as e:
                    retryable, status, retry_after = retry_info(e)
                    if not retryable or attempt >= self.limits.max_retries:
                        raise
                    error = e
            delay = retry_after if retry_after is not None else self.backoff(attempt)
            if status == 429:
                self.pause(delay)
            attempt += 1
            logger.warning(
                "%s: %s (attempt %d of %d); retrying in %.1f s",
                self.model_id,
                _short(error),
                attempt,
                self.limits.max_retries + 1,
                delay,
            )
            await asyncio.sleep(delay)


class _LoopState:
    def __init__(self) -> None:
        self.condition = asyncio.Condition()
        self.active = 0


def _short(error: Exception) -> str:
    text = str(error).splitlines()[0] if str(error) else ""
    return f"{type(error).__name__}: {text[:200]}"


# --- error classification ----------------------------------------------------------


def _status(error: Exception) -> Optional[int]:
    """The HTTP status of a provider error (openai, anthropic and google-genai)."""
    for name in ("status_code", "code", "status"):
        value = getattr(error, name, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    return None


def retry_after_seconds(error: Exception) -> Optional[float]:
    """The wait the provider asked for in ``Retry-After``, in seconds, if any."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        millis = headers.get("retry-after-ms")
        if millis is not None:
            return min(max(float(millis) / 1000, 0.0), MAX_RETRY_AFTER_SECONDS)
        value = headers.get("retry-after")
    except Exception:  # noqa: BLE001 - unusual header containers
        return None
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            date = email.utils.parsedate_to_datetime(str(value))
        except (TypeError, ValueError):
            return None
        seconds = date.timestamp() - time.time()
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)


def retry_info(error: Exception) -> tuple[bool, Optional[int], Optional[float]]:
    """Whether `error` is worth retrying, its HTTP status and ``Retry-After``.

    Retried: 408, 409, 429 and 5xx responses, timeouts and connection errors.
    """
    status = _status(error)
    if status is not None:
        retryable = status in _RETRYABLE_STATUS or status >= 500
        return retryable, status, retry_after_seconds(error) if retryable else None
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True, None, None
    names = {cls.__name__ for cls in type(error).__mro__}
    if names & _CONNECTION_ERROR_NAMES:
        return True, None, None
    return False, None, None


# --- configuration -------------------------------------------------------------------

_CONFIG_LOCK = threading.Lock()
_LIMITS: dict[str, RateLimits] = {}
_LIMITERS: dict[str, RateLimiter] = {}


def _resolve(model_id: str) -> RateLimits:
    provider = model_id.split("/", 1)[0]
    for key in (model_id, provider, "*"):
        if key in _LIMITS:
            return _LIMITS[key]
    return RateLimits()


def set_rate_limits(
    key: str = "*", limits: Optional[RateLimits] = None, **fields: Any
) -> RateLimits:
    """Set the limits of a model (``"provider/model"``), a provider, or ``"*"``.

    Pass a `RateLimits` or its fields; fields left out take their defaults. Models
    already in use pick up the new limits on their next call.

    Returns:
        The limits set.
    """
    if limits is None:
        limits = RateLimits(**fields)
    elif fields:
        limits = limits.model_copy(
            update=RateLimits(**fields).model_dump(exclude_unset=True)
        )
    with _CONFIG_LOCK:
        _LIMITS[key] = limits
        for model_id, limiter in _LIMITERS.items():
            limiter.limits = _resolve(model_id)
    return limits


def get_rate_limits(model_id: str) -> RateLimits:
    """The limits that apply to a model."""
    with _CONFIG_LOCK:
        return _resolve(model_id)


def get_rate_limiter(model_id: str) -> RateLimiter:
    """The limiter shared by every call to `model_id`."""
    limiter = _LIMITERS.get(model_id)
    if limiter is None:
        with _CONFIG_LOCK:
            limiter = _LIMITERS.get(model_id)
            if limiter is None:
                limiter = _LIMITERS[model_id] = RateLimiter(
                    model_id, _resolve(model_id)
                )
    return limiter


def reset_rate_limits() -> None:
    """Forget every configured limit and limiter (mainly for tests)."""
    with _CONFIG_LOCK:
        _LIMITS.clear()
        _LIMITERS.clear()
