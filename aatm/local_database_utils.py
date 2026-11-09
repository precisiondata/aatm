"""Utilities for building and managing local terminology-mapping data assets.

This module provides helper functions for constructing the local resources
required by the terminology-mapping workflow. These resources include a local
SQLite vocabulary database, derived CSV datasets used for terminology mapping,
and a local vector database for semantic retrieval.

The module also includes a simple rate-limiting helper for embedding generation
workloads and interactive prompts to prevent accidental overwrites of existing
artifacts.
"""

import gc
import shutil
import sqlite3
import time
from pathlib import Path
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import questionary
import torch
from chromadb import Embeddings
from rich.console import Console
from rich.progress import Progress, TaskID, track

from aatm.background_iterator import BackgroundIterator
from aatm.device import get_device
from aatm.logs import get_logger
from aatm.precision import (
    COLLECTION_DTYPE_KEY,
    DEFAULT_EMBEDDING_DTYPE,
    EmbeddingDtype,
    check_dtype_supported,
    dtype_for_existing_collection,
    requested_embedding_dtype,
)
from aatm.data_models import ExpressionMetadata
from aatm.timing_profiler import TimingProfiler
from aatm.vector_db_writer import ChromaWriterProcess, WriteAck

logger = get_logger(__name__)
console = Console()


def log_cuda_memory(label: str) -> None:
    """Log CUDA memory usage.

    This helps distinguish between:
    - allocated memory: memory currently held by live tensors;
    - reserved memory: memory kept by PyTorch's CUDA caching allocator;
    - driver free memory: memory visible as free to the CUDA driver.

    Args:
        label: Human-readable label identifying where the measurement was taken.
    """
    if not torch.cuda.is_available():
        logger.debug(f"[CUDA] {label} | CUDA not available")
        return

    torch.cuda.synchronize()

    allocated_gib = torch.cuda.memory_allocated() / 1024**3
    reserved_gib = torch.cuda.memory_reserved() / 1024**3
    max_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
    max_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
    free_bytes, total_bytes = torch.cuda.mem_get_info()

    logger.debug(
        f"[CUDA] {label} | "
        f"allocated={allocated_gib:.3f} GiB | "
        f"reserved={reserved_gib:.3f} GiB | "
        f"max_allocated={max_allocated_gib:.3f} GiB | "
        f"max_reserved={max_reserved_gib:.3f} GiB | "
        f"driver_free={free_bytes / 1024**3:.3f}/{total_bytes / 1024**3:.3f} GiB"
    )


def gpu_memory_cleanup(reset_peak: bool = False, synchronize: bool = True) -> None:
    """Clean Python references and release unused PyTorch CUDA cache.

    Important:
        ``torch.cuda.empty_cache()`` does not free live tensors. It only releases
        unused cached memory held by PyTorch. If ``memory_allocated`` remains high,
        some CUDA tensor is still alive.

    Args:
        reset_peak: Whether to reset CUDA peak memory statistics.
    """
    gc.collect()

    if torch.cuda.is_available():
        if synchronize:
            torch.cuda.synchronize()

        torch.cuda.empty_cache()

        if reset_peak:
            torch.cuda.reset_peak_memory_stats()


def list_live_cuda_tensors(limit: int = 20) -> None:
    """Log live CUDA tensors currently reachable by Python's garbage collector.

    This is useful for debugging suspected GPU memory leaks.

    Args:
        limit: Maximum number of tensor summaries to log.
    """
    if not torch.cuda.is_available():
        logger.debug("[CUDA] CUDA not available")
        return

    cuda_tensors: list[torch.Tensor] = []

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                cuda_tensors.append(obj)
        except Exception:
            continue

    logger.debug(f"[CUDA] Live CUDA tensors: {len(cuda_tensors)}")

    for tensor in cuda_tensors[:limit]:
        size_mib = tensor.numel() * tensor.element_size() / 1024**2
        logger.debug(
            f"[CUDA] tensor | "
            f"shape={tuple(tensor.shape)} | "
            f"dtype={tensor.dtype} | "
            f"device={tensor.device} | "
            f"size={size_mib:.2f} MiB"
        )


def ensure_cpu_embeddings(embeddings: Embeddings) -> Embeddings:
    """Ensure embeddings are converted to CPU-side Python lists.

    ChromaDB should receive CPU objects, not CUDA tensors. If the embedding
    function returns a CUDA tensor, passing it forward can keep GPU memory alive
    longer than expected.

    Args:
        embeddings: Embeddings returned by an embedding function.

    Returns:
        Embeddings as a list of lists of floats.
    """
    if torch.is_tensor(embeddings):
        return embeddings.detach().cpu().float().numpy().tolist()

    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()

    return embeddings


def rate_limiter(
    n_docs: int, rate_limit: int, next_allowed_time: float
) -> tuple[float, float]:
    """Apply a document-based rate limit and return the next allowed execution time.

    Returns:
        Tuple containing:
            - updated next_allowed_time
            - elapsed sleep time
    """
    now = time.monotonic()
    slept_seconds = 0.0

    if now < next_allowed_time:
        slept_seconds = next_allowed_time - now
        time.sleep(slept_seconds)
        now = time.monotonic()

    next_allowed_time = max(next_allowed_time, now) + n_docs * 60.0 / rate_limit

    return next_allowed_time, slept_seconds


def build_local_sqlite_vocab_database(vocab_dir: Path) -> None:
    """Build a local SQLite vocabulary database from OMOP vocabulary files.

    This function reads vocabulary files from a directory, converts each file
    into a SQLite table, and stores the result in a local database file. If a
    database already exists, the user is prompted to either skip rebuilding it
    or overwrite the existing database.

    CSV parsing behavior depends on the table being imported. The
    ``source_to_concept_map`` file is read as comma-separated, while the other
    vocabulary files are read as tab-separated.

    Args:
        vocab_dir: Path to the directory containing the vocabulary files to be
            imported into the local SQLite database.

    Returns:
        None.
    """
    if Path(".aatm/omop.db").exists():
        user_preference = questionary.select(
            "A vocabulary database already exists. What would you like to do?",
            choices=[
                "Skip",
                "Overwrite",
            ],
        ).ask()

        if user_preference == "Skip":
            return

    con = sqlite3.connect(".aatm/omop.db")

    for file_path in track(
        list(vocab_dir.glob("*.csv")),
        description="Building vocabulary database...",
    ):
        table_name = file_path.stem.lower()

        if table_name == "source_to_concept_map":
            df = pd.read_csv(file_path, sep=",", dtype=str)
        else:
            df = pd.read_csv(file_path, sep="\t", dtype=str)

        df.to_sql(table_name, con, index=False, if_exists="replace")

    con.commit()
    con.close()


def build_mapping_datasets(standard_vocabularies: list[str]) -> None:
    """Generate terminology-mapping datasets from SQL templates and a local database.

    This function loads SQL command templates from a packaged YAML file, formats
    them using the provided list of standard vocabularies, executes each query
    against the local OMOP SQLite database, and saves the resulting datasets as
    CSV files in the local datasets directory.

    Args:
        standard_vocabularies: List of standard vocabulary names used to
            parameterize the SQL queries that generate the mapping datasets.

    Returns:
        None.
    """
    import yaml
    from importlib.resources import files

    sql_commands_path = files("aatm").joinpath("sql_commands.yaml")
    sql_commands = yaml.safe_load(sql_commands_path.read_text(encoding="utf-8"))

    if len(standard_vocabularies) == 0:
        raise ValueError("No standard vocabularies provided")

    datasets_base_path = Path(".aatm/datasets")

    if datasets_base_path.exists() and len(list(datasets_base_path.glob("*.csv"))) > 0:
        user_preference = questionary.select(
            "The datasets for terminology mapping already exist. What would you like to do?",
            choices=[
                "Skip",
                "Overwrite",
            ],
        ).ask()

        if user_preference == "Skip":
            return

    datasets_base_path.mkdir(exist_ok=True, parents=True)

    con = sqlite3.connect(".aatm/omop.db")

    try:
        for command_name in track(
            sql_commands.keys(),
            description="Building datasets for terminology mapping...",
        ):
            sql_prompt = sql_commands[command_name]["sql"]
            sql_prompt_formatted = sql_prompt.format(
                standard_vocabulary_list=tuple(standard_vocabularies)
            )

            sql_prompt_df = pd.read_sql(sql_prompt_formatted, con)
            sql_prompt_df.to_csv(
                datasets_base_path / f"{command_name}.csv",
                index=False,
            )
    finally:
        con.close()


def measure_embedding_vram(
    model: Any,
    batch_size: int = 1,
    seq_len: int = 1000,
    device: str = "cuda",
) -> dict[str, Any]:
    """Measure approximate peak VRAM used by an embedding model forward pass.

    Args:
        model: Local embedding model.
        batch_size: Batch size used in the synthetic forward pass.
        seq_len: Sequence length used in the synthetic forward pass.
        device: Device on which to run the measurement.

    Returns:
        Dictionary with peak allocated memory, peak reserved memory, and output
        embedding shape.
    """
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA is not available")

    gpu_memory_cleanup(reset_peak=True)

    model = model.to(device)
    model.eval()

    input_ids = torch.ones(
        (batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones(
        (batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        outputs = model(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    sentence_embedding = outputs["sentence_embedding"]
    embedding_shape = tuple(sentence_embedding.shape)

    peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3

    del sentence_embedding
    del outputs
    del input_ids
    del attention_mask

    gpu_memory_cleanup()

    return {
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "embedding_shape": embedding_shape,
    }


def estimate_batch_size(embedding_function: Any) -> int:
    """Estimate a suitable batch size for local GPU embedding generation.

    The estimate reuses the model that is already loaded for building the
    database, so the weights are not loaded a second time and the free-memory
    reading reflects what the build will actually have available.

    Args:
        embedding_function: Loaded local embedding function. It must expose
            ``is_local = True`` and a SentenceTransformer ``model`` attribute.

    Returns:
        Estimated maximum batch size.

    Raises:
        ValueError: If the embedding function is not a local model, or the batch
            size cannot be estimated.
    """
    if not getattr(embedding_function, "is_local", False):
        raise ValueError(
            f"{type(embedding_function).__name__} is not a local embedding model"
        )

    model = embedding_function.model

    measurements = []

    free_bytes, _ = torch.cuda.mem_get_info()
    run_time_overhead_gib = 1.5

    for batch_size in [1, 10]:
        for seq_len in [500]:
            result = measure_embedding_vram(
                model=model,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            measurements.append((batch_size, seq_len, result))

    peak_reserved_gib_diff = (
        measurements[-1][2]["peak_reserved_gib"]
        - measurements[-2][2]["peak_reserved_gib"]
    )
    batch_size_diff = measurements[-1][0] - measurements[-2][0]

    if batch_size_diff <= 0:
        raise ValueError("Invalid batch size measurements")

    if peak_reserved_gib_diff <= 0:
        raise ValueError(
            "Unable to estimate memory per batch item for "
            f"{type(embedding_function).__name__}. "
            f"Measured peak reserved memory did not increase."
        )

    memory_per_batch_item_gib = peak_reserved_gib_diff / batch_size_diff

    max_batch_size = int(
        (free_bytes / 1024**3 - run_time_overhead_gib) / memory_per_batch_item_gib
    )

    gpu_memory_cleanup()

    if max_batch_size > 0:
        return max_batch_size

    raise ValueError(
        f"Unable to estimate batch size for {type(embedding_function).__name__}"
    )


PREFETCH_CHUNKS = 2
"""How many prepared chunks may wait for the GPU while it encodes."""


def get_existing_collection(client: Any, name: str) -> Any | None:
    """Return the named ChromaDB collection, or ``None`` if it does not exist."""
    from chromadb.errors import NotFoundError

    try:
        return client.get_collection(name)
    except NotFoundError:
        return None


def release_chromadb_clients() -> None:
    """Drop this process's cached ChromaDB clients so another process can open the files."""
    from chromadb.api.shared_system_client import SharedSystemClient

    SharedSystemClient.clear_system_cache()
    gc.collect()


@dataclass
class PreparedChunk:
    """One chunk of dataset rows, ready to be embedded and written.

    Attributes:
        expression_origin: Dataset the rows come from.
        dataset_rows: Total rows in that dataset, for progress reporting.
        row_start: Index of the chunk's first row in the dataset.
        n_rows: Rows in the chunk before deduplication.
        n_unique: Rows left after removing duplicate IDs within the chunk.
        ids: IDs not yet in the database. Empty if every row already exists.
        documents: Expression text for each of ``ids``.
        metadatas: Metadata for each of ``ids``.
        prepared_at: `time.perf_counter` value when preparation started.
    """

    expression_origin: str
    dataset_rows: int
    row_start: int
    n_rows: int
    n_unique: int
    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, Any]]
    prepared_at: float


def prepare_chunks(
    dataset_paths: Iterable[Path],
    chunk_size: int,
    existing_ids: Callable[[list[str]], set[str]],
    profiler: TimingProfiler,
) -> Iterator[PreparedChunk]:
    """Read datasets and yield chunks of rows that still need embeddings.

    This runs in a background thread while the GPU encodes earlier chunks.

    Args:
        dataset_paths: CSV files built by `build_mapping_datasets`.
        chunk_size: Rows per chunk.
        existing_ids: Returns which of the given IDs are already stored.
        profiler: Profiler for per-stage timings.

    Yields:
        One `PreparedChunk` per ``chunk_size`` rows, including chunks whose rows
        are all present already, so progress can still be reported for them.

    Raises:
        ValueError: If a row has no ``expression_id``.
    """
    for dataset_path in dataset_paths:
        expression_origin = dataset_path.stem

        with profiler.time_block("read_csv"):
            df = pd.read_csv(dataset_path, low_memory=False)

        with profiler.time_block("drop_duplicates_dropna"):
            df = df.drop_duplicates().dropna()

        for row_start in range(0, len(df), chunk_size):
            prepared_at = time.perf_counter()

            with profiler.time_block("build_expression_metadata"):
                records = df.iloc[row_start : row_start + chunk_size].to_dict("records")
                # Keep the first row for each ID, as the database does.
                by_id: dict[str, ExpressionMetadata] = {}
                for record in records:
                    metadata = ExpressionMetadata(
                        **record,
                        expression_origin=expression_origin,  # type: ignore[arg-type]
                    )
                    if metadata.expression_id is None:
                        raise ValueError(
                            "expression_id cannot be None. "
                            f"Given: {metadata.model_dump()}"
                        )
                    by_id.setdefault(metadata.expression_id, metadata)

            with profiler.time_block("chromadb_get_existing_ids"):
                found = existing_ids(list(by_id))

            new = [
                (identifier, metadata)
                for identifier, metadata in by_id.items()
                if identifier not in found
            ]

            yield PreparedChunk(
                expression_origin=expression_origin,
                dataset_rows=len(df),
                row_start=row_start,
                n_rows=len(records),
                n_unique=len(by_id),
                ids=[identifier for identifier, _ in new],
                documents=[metadata.expression for _, metadata in new],  # type: ignore[misc]
                metadatas=[metadata.to_dict() for _, metadata in new],
                prepared_at=prepared_at,
            )

        del df


@dataclass
class _BuildTracker:
    """Track progress, counters and logs as chunks finish."""

    progress: Progress
    profiler: TimingProfiler
    summary_every_n_chunks: int
    pending: dict[int, PreparedChunk] = field(default_factory=dict)
    tasks: dict[str, TaskID] = field(default_factory=dict)
    processed_chunks: int = 0
    added_documents: int = 0

    def handle_acks(self, acks: list[WriteAck]) -> None:
        for ack in acks:
            chunk = self.pending.pop(ack.seq)
            self.profiler.add("chromadb_add (writer process)", ack.add_seconds)
            self.finish(chunk, n_new=ack.n_added)

    def finish(self, chunk: PreparedChunk, n_new: int) -> None:
        task = self.tasks.get(chunk.expression_origin)
        if task is None:
            task = self.progress.add_task(
                f"Adding embeddings for {chunk.expression_origin}",
                total=chunk.dataset_rows,
            )
            self.tasks[chunk.expression_origin] = task
        self.progress.advance(task, chunk.n_rows)

        self.profiler.log_batch(
            expression_origin=chunk.expression_origin,
            row_start=chunk.row_start,
            n_rows=chunk.n_rows,
            n_unique=chunk.n_unique,
            n_new=n_new,
            batch_elapsed=time.perf_counter() - chunk.prepared_at,
        )

        self.processed_chunks += 1
        self.added_documents += n_new

        if (
            self.summary_every_n_chunks > 0
            and self.processed_chunks % self.summary_every_n_chunks == 0
        ):
            self.profiler.log_summary(
                label=f"partial after {self.processed_chunks} chunks"
            )


LOCAL_MODEL_CHUNK_SIZE = 4096
"""Default rows per ChromaDB round-trip for local models.

Large chunks give each ``encode`` call enough texts to keep the GPU busy. The
GPU batch size is chosen separately, see `estimate_batch_size`.
"""

API_MODEL_CHUNK_SIZE = 100
"""Default rows per ChromaDB round-trip for API models.

For API models every chunk is one embeddings request, so this stays within
provider limits on inputs per request.
"""


def resolve_chunk_size(
    chunk_size: int | None, is_local: bool, max_chromadb_batch_size: int
) -> int:
    """Choose how many rows to read and write per ChromaDB round-trip.

    Args:
        chunk_size: Size requested by the caller, or ``None`` for the default.
        is_local: Whether the embedding model runs locally.
        max_chromadb_batch_size: Largest batch the ChromaDB client accepts in a
            single ``add`` call.

    Returns:
        The chunk size, capped at ``max_chromadb_batch_size``.

    Raises:
        ValueError: If ``chunk_size`` is not positive.
    """
    if chunk_size is None:
        chunk_size = LOCAL_MODEL_CHUNK_SIZE if is_local else API_MODEL_CHUNK_SIZE

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    return min(chunk_size, max_chromadb_batch_size)


def build_local_vector_database(
    embedding_model_name: str,
    vector_db_dir: Path | None = None,
    rate_limit: int | None = None,
    chunk_size: int | None = None,
    encode_batch_size: int | None = None,
    embedding_dtype: str | None = None,
    profile_timing: bool = False,
    profile_sync_cuda: bool = False,
    profile_summary_every_n_batches: int = 20,
) -> None:
    """Build or repair a local vector database for terminology mapping.

    Args:
        embedding_model_name: Registry key identifying the embedding model
            configuration to use.
        vector_db_dir: Optional path to the vector database directory. If not
            provided, the default path from the retriever model registry is used.
        rate_limit: Optional maximum number of documents to embed per minute. If
            not provided, the default value from the model registry is used.
        chunk_size: Number of dataset rows read, checked against the database
            and written to it per round-trip. Defaults to
            `LOCAL_MODEL_CHUNK_SIZE` for local models and
            `API_MODEL_CHUNK_SIZE` for API models.
        encode_batch_size: Number of texts a local model encodes per forward
            pass. If not provided, it is estimated from free GPU memory on CUDA,
            and otherwise left at the embedding function's default. Ignored for
            API models.
        embedding_dtype: Precision for local models: ``float32`` (default),
            ``bfloat16`` or ``float16``. If not provided,
            ``AATM_EMBEDDING_DTYPE`` is used. When repairing, the database's
            recorded precision is used unless a different one is requested,
            which raises `EmbeddingPrecisionError`. See `aatm.precision`.
        profile_timing: Whether to log per-stage timing information.
        profile_sync_cuda: Whether to synchronize CUDA around timed blocks.
            This makes GPU timings more accurate but adds overhead.
        profile_summary_every_n_batches: Log cumulative profile summary every N
            processed batches.

    Returns:
        None.

    Raises:
        EmbeddingPrecisionError: If the requested precision cannot run on this
            machine, or differs from the precision of the database being
            repaired.
    """
    from aatm.registries.retrievers import (
        CHROMADB_RETRIEVER_MODEL_REGISTRY as model_registry,
    )

    spec = model_registry[embedding_model_name]
    is_local = bool(getattr(spec.embedding_function_cls, "is_local", False))

    # Validate the requested precision before prompting or loading any weights.
    requested_dtype: EmbeddingDtype | None = None
    if is_local:
        requested_dtype = requested_embedding_dtype(embedding_dtype)
        if requested_dtype is not None:
            check_dtype_supported(requested_dtype, get_device())
    elif embedding_dtype is not None:
        raise ValueError(
            f"embedding_dtype only applies to local models; "
            f"'{embedding_model_name}' is an API model."
        )

    profiler = TimingProfiler(
        enabled=profile_timing,
        synchronize_cuda=profile_sync_cuda,
    )

    if vector_db_dir is not None and not vector_db_dir.exists():
        raise ValueError("vector_db_dir does not exist")

    if vector_db_dir is None:
        vector_db_dir = Path(spec.chromadb_path)

    # ChromaDB caches clients by path string; an absolute path keeps a cached
    # client from pointing at another directory after a change of cwd.
    vector_db_dir = vector_db_dir.resolve()

    if vector_db_dir.exists():
        user_preference = questionary.select(
            (
                f"A vector database using the embedding model "
                f"'{embedding_model_name}' already exists. What would you like to do?"
            ),
            choices=["Skip", "Repair", "Overwrite"],
            default="Skip",
        ).ask()

        if user_preference == "Skip":
            return

        if user_preference == "Overwrite":
            with profiler.time_block("delete_existing_vector_db"):
                shutil.rmtree(vector_db_dir)

        elif user_preference == "Repair":
            pass

    if rate_limit is None:
        rate_limit = model_registry[embedding_model_name].rate_limit

    if rate_limit is not None and rate_limit <= 0:
        raise ValueError("rate_limit must be > 0")

    next_allowed_time = time.monotonic() if rate_limit is not None else 0.0

    import chromadb

    console.print("Creating local vector database...")

    with profiler.time_block("chromadb_client_init"):
        client = chromadb.PersistentClient(path=str(vector_db_dir))

    collection_name = spec.collection_name
    existing_collection = get_existing_collection(client, collection_name)

    model_kwargs: dict[str, Any] = {}
    collection_metadata: dict[str, Any] | None = None

    if is_local:
        if existing_collection is None:
            build_dtype = requested_dtype or DEFAULT_EMBEDDING_DTYPE
        else:
            build_dtype = dtype_for_existing_collection(
                (existing_collection.metadata or {}).get(COLLECTION_DTYPE_KEY),
                requested_dtype,
            )
        model_kwargs["dtype"] = build_dtype
        collection_metadata = {COLLECTION_DTYPE_KEY: build_dtype}
        console.print(f"Embedding precision: {build_dtype}")

    with profiler.time_block("embedding_function_init"):
        embedding_function = spec.embedding_function_cls(
            model=spec.model_id,
            **model_kwargs,  # type: ignore[call-arg]
        )

    if is_local:
        if encode_batch_size is None and torch.cuda.is_available():
            with profiler.time_block("estimate_batch_size"):
                encode_batch_size = estimate_batch_size(embedding_function)

        if encode_batch_size is not None:
            if encode_batch_size <= 0:
                raise ValueError("encode_batch_size must be > 0")
            embedding_function.batch_size = encode_batch_size  # type: ignore[attr-defined]

        logger.info(
            f"Encoding with batch size {embedding_function.batch_size}"  # type: ignore[attr-defined]
        )

    chunk_size = resolve_chunk_size(
        chunk_size,
        is_local=is_local,
        max_chromadb_batch_size=client.get_max_batch_size(),
    )

    if existing_collection is None:
        with profiler.time_block("create_collection"):
            client.create_collection(
                collection_name,
                embedding_function=embedding_function,
                metadata=collection_metadata,
            )

    # From here on the writer process owns the database. Two clients in different
    # processes must not share one directory, so release this one first.
    del existing_collection
    del client
    release_chromadb_clients()

    dataset_paths = sorted(Path(".aatm/datasets").glob("*.csv"))

    with (
        ChromaWriterProcess(str(vector_db_dir), collection_name) as writer,
        BackgroundIterator(
            prepare_chunks(
                dataset_paths,
                chunk_size=chunk_size,
                existing_ids=writer.existing_ids,
                profiler=profiler,
            ),
            maxsize=PREFETCH_CHUNKS,
            name="aatm-chunk-preparer",
        ) as chunks,
        Progress(*Progress.get_default_columns(), console=console) as progress,
    ):
        tracker = _BuildTracker(
            progress=progress,
            profiler=profiler,
            summary_every_n_chunks=(
                profile_summary_every_n_batches if profile_timing else 0
            ),
        )
        seq = 0

        for chunk in chunks:
            tracker.handle_acks(writer.poll())

            if not chunk.ids:
                tracker.finish(chunk, n_new=0)
                continue

            if rate_limit is not None:
                with profiler.time_block("rate_limiter_total"):
                    next_allowed_time, slept_seconds = rate_limiter(
                        n_docs=len(chunk.ids),
                        rate_limit=rate_limit,
                        next_allowed_time=next_allowed_time,
                    )

                profiler.add("rate_limiter_sleep", slept_seconds)

            with profiler.time_block("embedding_function_call"):
                embeddings = embedding_function(chunk.documents)

            with profiler.time_block("ensure_cpu_embeddings"):
                embeddings_array = np.asarray(
                    ensure_cpu_embeddings(embeddings), dtype=np.float32
                )
                del embeddings

            seq += 1
            tracker.pending[seq] = chunk

            # Blocks only while the writer already has batches queued.
            with profiler.time_block("wait_for_writer"):
                writer.add(
                    seq,
                    chunk.ids,
                    embeddings_array,
                    chunk.documents,
                    chunk.metadatas,
                )

            if seq % 50 == 0:
                gpu_memory_cleanup()

        with profiler.time_block("wait_for_final_writes"):
            tracker.handle_acks(writer.close())

        profiler.add("wait_for_prepared_chunk", chunks.wait_seconds)

    console.print(
        f"Added {tracker.added_documents} embeddings in {tracker.processed_chunks} "
        "chunks."
    )

    with profiler.time_block("final_delete_objects"):
        del embedding_function

    with profiler.time_block("final_gpu_memory_cleanup"):
        gpu_memory_cleanup(synchronize=True)

    profiler.log_summary(label="final")
