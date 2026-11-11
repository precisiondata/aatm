"""Read from and write to a vector database index from a separate process.

ChromaDB's ``Collection.add`` runs in native code that keeps Python's global
interpreter lock (GIL) for its whole duration, which is about a second per few
thousand rows. In a thread of the same process that would also freeze the
thread driving the GPU, so the GPU would sit idle during every write.

`VectorStoreWriterProcess` moves all vector database access into a child process, which has
its own interpreter and its own GIL. The parent sends work over
`multiprocessing` queues:

- ``existing_ids`` asks which IDs are already stored. A dedicated thread in the
  child answers it, so lookups are not stuck behind slow writes.
  ``Collection.get`` releases the GIL, so lookups and writes overlap in the
  child too.
- ``add`` queues a batch for writing and returns at once, unless
  ``max_pending_writes`` batches are already waiting. That bound keeps memory
  flat when the writer is slower than the encoder.
- ``poll`` returns acknowledgements for batches that have been written.

Duplicate IDs are harmless: ChromaDB ignores an ``add`` for an ID that already
exists and keeps the first version.

The parent must not keep its own ChromaDB client open on the same directory
while the writer runs. For Chroma Cloud and Pinecone (see `aatm.vector_db`)
that restriction does not apply, and lookups and writes are retried on
transient errors. Every backend is reached through
`aatm.vector_stores.VectorStore`, opened with the connection's ``open_store``.
"""

from __future__ import annotations

import multiprocessing
import queue
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from types import TracebackType
from typing import Any

from aatm.vector_db import ChromaConnection, VectorDbConnection, call_with_retries

_STOP = None
"""Message telling a child loop to finish."""


class VectorDatabaseWriterError(RuntimeError):
    """Raised in the parent when the writer process fails."""


@dataclass(frozen=True, slots=True)
class WriteAck:
    """Acknowledgement that one batch was written.

    Attributes:
        seq: Sequence number the parent gave the batch.
        n_added: Number of rows in the batch.
        add_seconds: Time ``Collection.add`` took in the writer process.
    """

    seq: int
    n_added: int
    add_seconds: float


def _writer_main(
    connection: VectorDbConnection,
    collection_name: str,
    check_requests: Any,
    check_results: Any,
    add_requests: Any,
    events: Any,
) -> None:
    """Entry point of the writer process. Runs in the child."""
    try:
        store = connection.open_store(collection_name)

        # Network calls to Chroma Cloud can fail transiently; local calls do not.
        # A Pinecone store retries its own calls, request by request.
        def retrying(func: Callable[[int], Any]) -> Any:
            if isinstance(connection, ChromaConnection) and connection.is_cloud:
                return call_with_retries(func)
            return func(0)

        def write(attempt: int, **batch: Any) -> None:
            # A retried ``add`` may have reached the server before it failed, so
            # retries upsert the same rows instead, which is idempotent.
            store.write(attempt=attempt, **batch)

        def serve_lookups() -> None:
            try:
                while (request := check_requests.get()) is not _STOP:
                    seq, ids = request
                    found = retrying(lambda _: store.existing_ids(ids))
                    check_results.put((seq, list(found)))
            except BaseException:
                check_results.put(("error", traceback.format_exc()))

        lookup_thread = threading.Thread(
            target=serve_lookups, name="aatm-vector-db-lookups", daemon=True
        )
        lookup_thread.start()

        while (request := add_requests.get()) is not _STOP:
            seq, ids, embeddings, documents, metadatas = request
            started = time.perf_counter()
            retrying(
                partial(
                    write,
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
            )
            events.put(("added", seq, len(ids), time.perf_counter() - started))

        lookup_thread.join()
        events.put(("closed",))
    except KeyboardInterrupt:
        # Ctrl+C reaches the whole console group; the parent reports it.
        events.put(("closed",))
    except BaseException:
        events.put(("error", traceback.format_exc()))


class VectorStoreWriterProcess:
    """Own a vector database index in a child process. See the module docstring."""

    def __init__(
        self,
        target: str | VectorDbConnection,
        collection_name: str,
        max_pending_writes: int = 2,
        poll_interval: float = 0.2,
        shutdown_timeout: float = 60.0,
    ) -> None:
        """Prepare the writer. The process starts when the ``with`` block opens.

        Args:
            target: Directory of a local persistent ChromaDB database, or a
                connection: a `ChromaConnection` (local or Chroma Cloud) or a
                `aatm.vector_db.PineconeConnection`. Remote calls are retried
                on transient errors.
            collection_name: Name of an existing collection or index.
            max_pending_writes: Maximum number of batches queued for writing
                before ``add`` blocks.
            poll_interval: How often, in seconds, blocked calls wake up to check
                that the writer is still alive.
            shutdown_timeout: Seconds to wait for the writer to finish after an
                error before terminating it.
        """
        connection = (
            ChromaConnection.local(target) if isinstance(target, str) else target
        )
        context = multiprocessing.get_context("spawn")
        self._check_requests = context.Queue()
        self._check_results = context.Queue()
        self._add_requests = context.Queue(maxsize=max_pending_writes)
        self._events = context.Queue()
        self._process = context.Process(
            target=_writer_main,
            args=(
                connection,
                collection_name,
                self._check_requests,
                self._check_results,
                self._add_requests,
                self._events,
            ),
            name="aatm-vector-db-writer",
            daemon=True,
        )
        self._poll_interval = poll_interval
        self._shutdown_timeout = shutdown_timeout
        self._check_seq = 0
        self._closed = False

    def __enter__(self) -> VectorStoreWriterProcess:
        self._process.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def existing_ids(self, ids: list[str]) -> set[str]:
        """Return which of ``ids`` are already stored.

        Call from one thread at a time.

        Args:
            ids: IDs to look up.

        Returns:
            The subset of ``ids`` present in the collection.

        Raises:
            VectorDatabaseWriterError: If the writer process failed.
        """
        self._check_seq += 1
        self._check_requests.put((self._check_seq, ids))

        while True:
            try:
                result = self._check_results.get(timeout=self._poll_interval)
            except queue.Empty:
                self._raise_if_dead()
                continue

            if result[0] == "error":
                raise VectorDatabaseWriterError(
                    f"Vector database lookup failed in the writer process:\n{result[1]}"
                )

            seq, found = result
            if seq != self._check_seq:
                raise VectorDatabaseWriterError(
                    f"Lookup answers arrived out of order ({seq} != {self._check_seq})"
                )
            return set(found)

    def add(
        self,
        seq: int,
        ids: list[str],
        embeddings: Any,
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Queue a batch for writing. Blocks while too many batches are pending.

        Args:
            seq: Caller-chosen number returned in the batch's `WriteAck`.
            ids: Row IDs.
            embeddings: One vector per row, preferably a float32 numpy array.
            documents: Row documents.
            metadatas: Row metadata.

        Raises:
            VectorDatabaseWriterError: If the writer process failed.
        """
        item = (seq, ids, embeddings, documents, metadatas)
        while True:
            try:
                self._add_requests.put(item, timeout=self._poll_interval)
                return
            except queue.Full:
                self._raise_if_dead()

    def poll(self) -> list[WriteAck]:
        """Return acknowledgements for batches written since the last call.

        Raises:
            VectorDatabaseWriterError: If the writer process failed.
        """
        acks: list[WriteAck] = []
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return acks
            acks.extend(self._handle_event(event))

    def close(self) -> list[WriteAck]:
        """Finish all queued writes, stop the writer and return the last acks.

        Raises:
            VectorDatabaseWriterError: If the writer process failed.
        """
        if self._closed:
            return []
        self._closed = True

        self._check_requests.put(_STOP)
        self._put_stop_for_writes()

        acks: list[WriteAck] = []
        while True:
            try:
                event = self._events.get(timeout=self._poll_interval)
            except queue.Empty:
                if not self._process.is_alive():
                    break
                continue
            if event[0] == "closed":
                break
            acks.extend(self._handle_event(event))

        self._process.join()
        if self._process.exitcode not in (0, None):
            raise VectorDatabaseWriterError(
                f"Writer process exited with code {self._process.exitcode}"
            )
        return acks

    def abort(self) -> None:
        """Stop the writer after an error in the parent.

        Batches already queued are still written, so a later repair has less to
        redo. The process is terminated if it does not finish in time.
        """
        if self._closed:
            return
        self._closed = True

        if self._process.is_alive():
            self._check_requests.put(_STOP)
            self._put_stop_for_writes()
            self._process.join(self._shutdown_timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join()

    def _put_stop_for_writes(self) -> None:
        while self._process.is_alive():
            try:
                self._add_requests.put(_STOP, timeout=self._poll_interval)
                return
            except queue.Full:
                continue

    def _handle_event(self, event: tuple[Any, ...]) -> list[WriteAck]:
        if event[0] == "added":
            _, seq, n_added, add_seconds = event
            return [WriteAck(seq=seq, n_added=n_added, add_seconds=add_seconds)]
        if event[0] == "error":
            raise VectorDatabaseWriterError(
                f"Vector database write failed in the writer process:\n{event[1]}"
            )
        return []

    def _raise_if_dead(self) -> None:
        if self._process.is_alive():
            return
        # Surface the child's own error message if it sent one.
        self.poll()
        raise VectorDatabaseWriterError(
            f"Writer process exited unexpectedly with code {self._process.exitcode}"
        )


ChromaWriterProcess = VectorStoreWriterProcess
"""The writer's original name, from when it only wrote to ChromaDB."""
