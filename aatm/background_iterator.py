"""Run a producer in a background thread while the caller consumes its items.

This is the building block for overlapping CPU-side preparation with GPU work.
The producer is any iterable, usually a generator. Its body runs in a worker
thread, at most ``maxsize`` items ahead of the consumer, so memory stays bounded
however fast the producer is.

```python
from aatm.background_iterator import BackgroundIterator

with BackgroundIterator(prepare_chunks(), maxsize=2) as chunks:
    for chunk in chunks:  # waits only if the producer has fallen behind
        encode_on_gpu(chunk)
```

Python threads share one interpreter lock (the GIL), so this only helps when the
two sides spend most of their time in code that releases it, such as GPU
kernels, file and network I/O, or native libraries that drop the lock.

Guarantees:

- An exception raised by the producer is re-raised in the consumer, at the
  position in the stream where it happened.
- Leaving the ``with`` block early, including on an exception in the consumer,
  stops the producer at its next item and joins its thread.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterable, Iterator
from types import TracebackType
from typing import Generic, TypeVar

T = TypeVar("T")

_DONE = object()
"""Sentinel the producer sends after its last item."""


class _Failure:
    """Wrap an exception so it cannot be mistaken for a produced item."""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class BackgroundIterator(Generic[T]):
    """Iterate over ``source`` in a background thread, a bounded number of items ahead.

    Attributes:
        wait_seconds: Total time the consumer spent waiting for the producer.
            A large value means the producer is the bottleneck.
    """

    def __init__(
        self,
        source: Iterable[T],
        maxsize: int = 2,
        poll_interval: float = 0.1,
        name: str = "aatm-background-iterator",
    ) -> None:
        """Prepare the background producer. It starts when the ``with`` block opens.

        Args:
            source: Iterable whose items are produced in the background thread.
            maxsize: Maximum number of produced items waiting for the consumer.
            poll_interval: How often, in seconds, blocked calls wake up to check
                whether they should stop. Keeps Ctrl+C responsive.
            name: Name of the background thread, shown in debuggers and logs.

        Raises:
            ValueError: If ``maxsize`` is smaller than 1.
        """
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")

        self._source = source
        self._queue: queue.Queue[object] = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._poll_interval = poll_interval
        self._thread = threading.Thread(target=self._produce, name=name, daemon=True)
        self.wait_seconds = 0.0

    def __enter__(self) -> BackgroundIterator[T]:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Stop the producer and wait for its thread to finish."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def __iter__(self) -> Iterator[T]:
        while True:
            started = time.perf_counter()
            item = self._get()
            self.wait_seconds += time.perf_counter() - started

            if item is _DONE:
                return
            if isinstance(item, _Failure):
                raise item.exc

            yield item  # type: ignore[misc]

    def _get(self) -> object:
        # A timeout keeps the wait interruptible by Ctrl+C on every platform.
        while True:
            try:
                return self._queue.get(timeout=self._poll_interval)
            except queue.Empty:
                if not self._thread.is_alive() and self._queue.empty():
                    raise RuntimeError(
                        "Background producer stopped without finishing"
                    ) from None

    def _put(self, item: object) -> bool:
        """Put an item, giving up if the consumer asked to stop."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=self._poll_interval)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self) -> None:
        iterator = iter(self._source)
        try:
            for item in iterator:
                if not self._put(item):
                    return
            self._put(_DONE)
        except BaseException as exc:
            self._put(_Failure(exc))
        finally:
            # Run the generator's own cleanup (its `finally` blocks) in this
            # thread, where it was running, instead of whenever it is collected.
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
