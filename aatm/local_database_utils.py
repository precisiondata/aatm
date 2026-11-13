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
from functools import partial
import shutil
import sqlite3
import time
from pathlib import Path
from collections.abc import Callable, Iterable, Iterator, Sequence
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
from aatm.built_stores import record_vector_database
from aatm.device import get_device
from aatm.gpu_memory import (
    CANDIDATE_BATCH_SIZES,
    GpuMemoryLimit,
    calibrate_max_batch_tokens,
    encode_with_oom_backoff,
)
from aatm.logs import get_logger
from aatm.precision import (
    COLLECTION_DTYPE_KEY,
    EmbeddingDtype,
    check_dtype_supported,
    default_embedding_dtype,
    dtype_for_existing_collection,
    requested_embedding_dtype,
)
from aatm.data_models import ExpressionMetadata
from aatm.timing_profiler import TimingProfiler
from aatm.vector_db import (
    ChromaConnection,
    PineconeConnection,
    VectorDbConnection,
    call_with_retries,
    resolve_vector_db_mode,
)
from aatm.vector_db_writer import VectorStoreWriterProcess, WriteAck
from aatm.vector_stores import (
    PineconeVectorStore,
    VectorStore,
    create_pinecone_index,
    delete_pinecone_index,
    pinecone_index_exists,
)

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


def collect_calibration_texts(
    dataset_paths: Iterable[Path],
    n_longest: int = max(CANDIDATE_BATCH_SIZES),
    n_typical: int = 4096,
    token_lengths: Callable[[list[str]], list[int]] | None = None,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Collect texts for GPU calibration: the longest and a typical sample.

    Args:
        dataset_paths: CSV files built by `build_mapping_datasets`.
        n_longest: Number of longest texts to return.
        n_typical: Number of randomly sampled texts to return.
        token_lengths: Returns the token count of each text. When given, the
            longest texts are ranked by tokens, which is what GPU memory depends
            on, instead of by characters.
        seed: Random seed for the typical sample.

    Returns:
        The longest texts, longest first, and a random sample of texts.
    """
    longest: list[str] = []
    samples: list[pd.Series] = []

    for dataset_path in dataset_paths:
        expressions = (
            pd.read_csv(dataset_path, usecols=["expression"], low_memory=False)[
                "expression"
            ]
            .dropna()
            .astype(str)
            .drop_duplicates()
        )
        # Over-collect by characters, then rank by tokens below.
        top = expressions.str.len().nlargest(4 * n_longest).index
        longest.extend(expressions.loc[top].tolist())
        samples.append(
            expressions.sample(n=min(n_typical, len(expressions)), random_state=seed)
        )

    if token_lengths is not None and longest:
        ranked = sorted(zip(token_lengths(longest), longest), key=lambda pair: -pair[0])
        longest = [text for _, text in ranked]
    else:
        longest.sort(key=len, reverse=True)

    typical = pd.concat(samples) if samples else pd.Series(dtype=str)
    typical = typical.sample(n=min(n_typical, len(typical)), random_state=seed)
    return longest[:n_longest], typical.tolist()


def _encode_with_max_batch_tokens(
    embedding_function: Any, texts: Sequence[str], max_batch_tokens: int
) -> Any:
    embedding_function.max_batch_tokens = max_batch_tokens
    return embedding_function(list(texts))


def _token_lengths(embedding_function: Any) -> Callable[[list[str]], list[int]]:
    if hasattr(embedding_function, "token_lengths"):
        return lambda texts: embedding_function.token_lengths(texts)

    tokenizer = embedding_function.model.tokenizer

    def lengths(texts: list[str]) -> list[int]:
        return [len(ids) for ids in tokenizer(texts)["input_ids"]]

    return lengths


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
GPU batch size is chosen separately, see `aatm.gpu_memory`.
"""

API_MODEL_CHUNK_SIZE = 100
"""Default rows per ChromaDB round-trip for API models.

For API models every chunk is one embeddings request, so this stays within
provider limits on inputs per request.
"""


PINECONE_MAX_CHUNK_SIZE = 10_000
"""Largest chunk for a Pinecone build.

Pinecone has no single batch limit like ChromaDB's: the store splits each chunk
into upserts within Pinecone's record and request-size limits.
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
    encode_max_batch_tokens: int | None = None,
    embedding_dtype: str | None = None,
    profile_timing: bool = False,
    profile_sync_cuda: bool = False,
    profile_summary_every_n_batches: int = 20,
    vector_db: str | None = None,
) -> None:
    """Build or repair the vector database for terminology mapping.

    The database is local by default. With ``vector_db="cloud"`` (or
    ``AATM_VECTOR_DB=cloud``) it is built in Chroma Cloud instead, and with
    ``vector_db="pinecone"`` in a Pinecone index; see `aatm.vector_db`.
    Embeddings are still computed on this machine. A new Pinecone index gets
    its dimension from one probe embedding.

    Args:
        embedding_model_name: Registry key identifying the embedding model
            configuration to use.
        vector_db_dir: Optional path to the local vector database directory. If
            not provided, the default path from the retriever model registry is
            used. Not allowed in cloud mode.
        rate_limit: Optional maximum number of documents to embed per minute. If
            not provided, the default value from the model registry is used.
        chunk_size: Number of dataset rows read, checked against the database
            and written to it per round-trip. Defaults to
            `LOCAL_MODEL_CHUNK_SIZE` for local models and
            `API_MODEL_CHUNK_SIZE` for API models.
        encode_batch_size: Fixed number of texts a local model encodes per
            forward pass. Turns off batching by token budget. Ignored for API
            models.
        encode_max_batch_tokens: Padded tokens a local model encodes per
            forward pass; batches of short texts then hold more texts. If
            neither this nor ``encode_batch_size`` is provided, on CUDA the
            budget is calibrated on the datasets' longest and typical texts
            (see `aatm.gpu_memory`); otherwise the embedding function's default
            batch size is used. Ignored for API models.
        embedding_dtype: Precision for local models: ``float32``,
            ``bfloat16`` or ``float16``. If not provided,
            ``AATM_EMBEDDING_DTYPE`` is used, and otherwise the precision the
            model's checkpoint is stored in. When repairing, the database's
            recorded precision is used unless a different one is requested,
            which raises `EmbeddingPrecisionError`. See `aatm.precision`.
        profile_timing: Whether to log per-stage timing information.
        profile_sync_cuda: Whether to synchronize CUDA around timed blocks.
            This makes GPU timings more accurate but adds overhead.
        profile_summary_every_n_batches: Log cumulative profile summary every N
            processed batches.
        vector_db: ``local``, ``cloud`` or ``pinecone``. If not provided,
            ``AATM_VECTOR_DB`` decides, and otherwise ``local``.

    Returns:
        None.

    Raises:
        EmbeddingPrecisionError: If the requested precision cannot run on this
            machine, or differs from the precision of the database being
            repaired.
        VectorDatabaseConfigError: If the vector database configuration is
            invalid, e.g. cloud mode without ``CHROMA_API_KEY``.
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

    connection: VectorDbConnection
    client: Any | None = None
    # Set in Pinecone mode only; `client` is a ChromaDB client otherwise.
    pinecone_client: Any | None = None

    mode = resolve_vector_db_mode(vector_db)
    # Where the database is recorded in .aatm/config.yaml (see aatm.built_stores).
    record_location: str | Path
    if mode == "pinecone":
        if vector_db_dir is not None:
            raise ValueError("vector_db_dir only applies to a local vector database.")
        connection, collection_name = spec.pinecone_location()
        record_location = collection_name
        console.print(f"Using {connection.describe()}, index '{collection_name}'")
        # Opening the client does not authenticate; the first call does.
        with profiler.time_block("pinecone_client_init"):
            pinecone_client = connection.client()
        already_exists = pinecone_index_exists(pinecone_client, collection_name)
    elif (location := spec.chroma_location(vector_db))[0].is_cloud:
        connection, collection_name = location
        if vector_db_dir is not None:
            raise ValueError("vector_db_dir only applies to a local vector database.")
        record_location = collection_name
        console.print(f"Using {connection.describe()}, collection '{collection_name}'")
        # Opening the client authenticates, so bad credentials fail here.
        with profiler.time_block("chromadb_client_init"):
            client = connection.client()
        already_exists = get_existing_collection(client, collection_name) is not None
    else:
        collection_name = location[1]
        if vector_db_dir is not None and not vector_db_dir.exists():
            raise ValueError("vector_db_dir does not exist")

        if vector_db_dir is None:
            vector_db_dir = Path(spec.chromadb_path)
        record_location = vector_db_dir

        # ChromaDB caches clients by path string; an absolute path keeps a cached
        # client from pointing at another directory after a change of cwd.
        vector_db_dir = vector_db_dir.resolve()
        connection = ChromaConnection.local(str(vector_db_dir))
        already_exists = vector_db_dir.exists()

    if already_exists:
        user_preference = questionary.select(
            (
                f"A vector database using the embedding model "
                f"'{embedding_model_name}' already exists. What would you like to do?"
            ),
            choices=["Skip", "Repair", "Overwrite"],
            default="Skip",
        ).ask()

        if user_preference == "Skip":
            record_vector_database(
                embedding_model_name, mode, record_location, keep_date=True
            )
            return

        if user_preference == "Overwrite":
            with profiler.time_block("delete_existing_vector_db"):
                if pinecone_client is not None:
                    delete_pinecone_index(pinecone_client, collection_name)
                elif client is not None:
                    client.delete_collection(collection_name)
                else:
                    shutil.rmtree(vector_db_dir)  # type: ignore[arg-type]

        elif user_preference == "Repair":
            pass

    if rate_limit is None:
        rate_limit = model_registry[embedding_model_name].rate_limit

    if rate_limit is not None and rate_limit <= 0:
        raise ValueError("rate_limit must be > 0")

    next_allowed_time = time.monotonic() if rate_limit is not None else 0.0

    console.print(f"Creating vector database in {connection.describe()}...")

    # Metadata recorded on the existing collection or index, None if there is none.
    existing_metadata: dict[str, Any] | None
    existing_collection: Any | None = None
    if pinecone_client is not None:
        existing_metadata = (
            PineconeVectorStore.open(pinecone_client, collection_name).index_metadata()
            if pinecone_index_exists(pinecone_client, collection_name)
            else None
        )
    else:
        if client is None:
            with profiler.time_block("chromadb_client_init"):
                client = connection.client()
        existing_collection = get_existing_collection(client, collection_name)
        existing_metadata = (
            None
            if existing_collection is None
            else dict(existing_collection.metadata or {})
        )

    model_kwargs: dict[str, Any] = {}
    collection_metadata: dict[str, Any] | None = None

    if is_local:
        if existing_metadata is None:
            build_dtype = requested_dtype or default_embedding_dtype(
                spec.model_id, get_device()
            )
        else:
            build_dtype = dtype_for_existing_collection(
                existing_metadata.get(COLLECTION_DTYPE_KEY),
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

    if encode_batch_size is not None and encode_max_batch_tokens is not None:
        raise ValueError("Pass encode_batch_size or encode_max_batch_tokens, not both.")

    if is_local and encode_batch_size is not None:
        if encode_batch_size <= 0:
            raise ValueError("encode_batch_size must be > 0")
        embedding_function.batch_size = encode_batch_size  # type: ignore[attr-defined]
        embedding_function.max_batch_tokens = None  # type: ignore[attr-defined]

    if is_local and encode_max_batch_tokens is not None:
        if encode_max_batch_tokens <= 0:
            raise ValueError("encode_max_batch_tokens must be > 0")
        embedding_function.max_batch_tokens = encode_max_batch_tokens  # type: ignore[attr-defined]

    # Cap GPU memory and size batches by measurement, so the build never
    # overflows GPU memory. See `aatm.gpu_memory`.
    guard_gpu_memory = is_local and get_device() == "cuda" and torch.cuda.is_available()

    chunk_size = resolve_chunk_size(
        chunk_size,
        is_local=is_local,
        max_chromadb_batch_size=(
            PINECONE_MAX_CHUNK_SIZE if client is None else client.get_max_batch_size()
        ),
    )

    if pinecone_client is not None:
        if existing_metadata is None:
            # A Pinecone index has a fixed dimension, so embed once to learn it.
            with profiler.time_block("create_collection"):
                probe = np.asarray(
                    ensure_cpu_embeddings(embedding_function(["dimension probe"]))
                )
                create_pinecone_index(
                    pinecone_client,
                    collection_name,
                    dimension=int(probe.shape[-1]),
                    cloud=connection.cloud,  # type: ignore[union-attr]
                    region=connection.region,  # type: ignore[union-attr]
                    tags=collection_metadata,
                )
    elif existing_collection is None:
        with profiler.time_block("create_collection"):
            client.create_collection(  # type: ignore[union-attr]
                collection_name,
                embedding_function=embedding_function,
                metadata=collection_metadata,
            )

    # From here on the writer process owns the database. Two clients in different
    # processes must not share one directory, so release this one first.
    del existing_collection
    del client, pinecone_client
    release_chromadb_clients()

    dataset_paths = sorted(Path(".aatm/datasets").glob("*.csv"))

    with (
        GpuMemoryLimit(enabled=guard_gpu_memory) as memory_limit,
        VectorStoreWriterProcess(connection, collection_name) as writer,
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

        if (
            guard_gpu_memory
            and encode_batch_size is None
            and encode_max_batch_tokens is None
        ):
            with profiler.time_block("calibrate_max_batch_tokens"):
                token_lengths = _token_lengths(embedding_function)
                longest_texts, typical_texts = collect_calibration_texts(
                    dataset_paths,
                    token_lengths=token_lengths,
                )
                longest_tokens = max(token_lengths(longest_texts[:1]), default=1)
                embedding_function.max_batch_tokens = calibrate_max_batch_tokens(  # type: ignore[attr-defined]
                    partial(_encode_with_max_batch_tokens, embedding_function),
                    longest_texts,
                    longest_tokens,
                    typical_texts,
                    limit_bytes=memory_limit.limit_bytes,
                )

        if is_local:
            max_batch_tokens = getattr(embedding_function, "max_batch_tokens", None)
            if max_batch_tokens is not None:
                logger.info(f"Encoding with up to {max_batch_tokens} tokens per batch")
            else:
                logger.info(
                    f"Encoding with batch size {embedding_function.batch_size}"  # type: ignore[attr-defined]
                )

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
                if is_local:
                    embeddings = encode_with_oom_backoff(
                        embedding_function, chunk.documents
                    )
                else:
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

    record_vector_database(embedding_model_name, mode, record_location)


UPLOAD_READ_PAGE_SIZE = 5000
"""Rows read from the local database per page when uploading it to the cloud.

Reading uses offsets, so larger pages mean fewer, cheaper scans. Each page is
written to the cloud in batches no larger than the cloud's maximum batch size.
"""


@dataclass(frozen=True, slots=True)
class UploadSummary:
    """Result of `upload_vector_database_to_cloud`.

    Attributes:
        collection_name: Cloud collection that received the rows.
        total: Rows in the local collection.
        uploaded: Rows written to the cloud by this run.
        skipped: Rows already present in the cloud collection.
    """

    collection_name: str
    total: int
    uploaded: int
    skipped: int


UPLOAD_TARGETS = ("cloud", "pinecone")
"""Where `upload_vector_database_to_cloud` can copy a local index to."""

PINECONE_UPLOAD_BATCH_SIZE = 1000
"""Rows checked and written per round-trip when uploading to Pinecone.

The store splits each batch into upserts within Pinecone's limits.
"""


def upload_vector_database_to_cloud(
    embedding_model_name: str,
    vector_db_dir: Path | None = None,
    write_batch_size: int | None = None,
    read_page_size: int = UPLOAD_READ_PAGE_SIZE,
    cloud_connection: VectorDbConnection | None = None,
    to: str = "cloud",
    ids: Sequence[str] | None = None,
) -> UploadSummary:
    """Copy a local vector database to Chroma Cloud or Pinecone without re-embedding.

    Embeddings, documents and metadata are read from the local collection and
    added to the model's cloud collection (see
    `ChromaRetrieverModelSpec.cloud_collection_name`) or Pinecone index (see
    `ChromaRetrieverModelSpec.pinecone_index_name`). The collection or index is
    created with the local collection's metadata if needed; a new Pinecone
    index takes its dimension from the local embeddings. Rows already in the
    destination are skipped, so an interrupted upload can simply be run again.
    With ``ids``, only those rows are copied, e.g. a subset that fits a free
    plan.

    Args:
        embedding_model_name: Registry key of the embedding model whose index
            to upload.
        vector_db_dir: Local database directory. Defaults to the registry path.
        write_batch_size: Rows per write. For Chroma Cloud it defaults to, and
            is capped at, the cloud's maximum batch size; for Pinecone it
            defaults to `PINECONE_UPLOAD_BATCH_SIZE`, and each batch is split
            into upserts within Pinecone's limits.
        read_page_size: Rows read from the local database at a time.
        cloud_connection: Destination connection. Defaults to the one
            configured for ``to`` (see `aatm.vector_db`).
        to: ``cloud`` (Chroma Cloud, the default) or ``pinecone``. Ignored when
            ``cloud_connection`` is given.
        ids: Only copy the rows with these IDs. IDs missing from the local
            collection are reported and skipped. Defaults to every row.

    Returns:
        How many rows were uploaded and skipped.

    Raises:
        ValueError: If the local database or collection does not exist, the
            sizes are not positive, ``to`` is unknown, or an existing Pinecone
            index was built with another embedding precision.
        VectorDatabaseConfigError: If the destination configuration is missing.
    """
    from aatm.registries.retrievers import (
        CHROMADB_RETRIEVER_MODEL_REGISTRY as model_registry,
    )

    spec = model_registry[embedding_model_name]

    if read_page_size <= 0:
        raise ValueError("read_page_size must be > 0")
    if write_batch_size is not None and write_batch_size <= 0:
        raise ValueError("write_batch_size must be > 0")
    if cloud_connection is None and to not in UPLOAD_TARGETS:
        raise ValueError(
            f"Unknown upload destination '{to}'. Use one of: {', '.join(UPLOAD_TARGETS)}."
        )

    local_dir = (vector_db_dir or Path(spec.chromadb_path)).resolve()
    if not local_dir.exists():
        raise ValueError(f"No local vector database at '{local_dir}'.")

    if cloud_connection is None:
        cloud_connection = (
            PineconeConnection.from_env()
            if to == "pinecone"
            else ChromaConnection.from_env(local_path=str(local_dir), mode="cloud")
        )
    if not cloud_connection.is_cloud:
        raise ValueError("cloud_connection must point at Chroma Cloud or Pinecone.")

    local_client = ChromaConnection.local(str(local_dir)).client()
    local_collection = get_existing_collection(local_client, spec.collection_name)
    if local_collection is None:
        raise ValueError(
            f"The local database at '{local_dir}' has no collection "
            f"'{spec.collection_name}'."
        )

    # Keep the build's metadata, e.g. the embedding precision for query encoding.
    collection_metadata = local_collection.metadata or None
    selected = None if ids is None else list(dict.fromkeys(ids))
    total = local_collection.count() if selected is None else len(selected)

    def pages(collection: Any) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield how many rows each page covers, and the page."""
        include = ["embeddings", "documents", "metadatas"]
        if selected is None:
            for offset in range(0, total, read_page_size):
                page = collection.get(
                    limit=read_page_size, offset=offset, include=include
                )
                yield len(page["ids"]), page
            return
        for start in range(0, total, read_page_size):
            wanted = selected[start : start + read_page_size]
            page = collection.get(ids=wanted, include=include)
            if missing := len(wanted) - len(page["ids"]):
                logger.warning(f"{missing} of the given IDs are not in the local index")
            yield len(wanted), page

    store: VectorStore
    retry: Callable[[Callable[[int], Any]], Any]
    if isinstance(cloud_connection, PineconeConnection):
        collection_name = spec.pinecone_index_name(cloud_connection)
        store = _open_or_create_pinecone_index(
            cloud_connection, collection_name, local_collection, collection_metadata
        )
        batch_size = write_batch_size or PINECONE_UPLOAD_BATCH_SIZE

        # The Pinecone store retries each of its requests itself.
        def retry(func: Callable[[int], Any]) -> Any:
            return func(0)

    else:
        from aatm.vector_stores import ChromaVectorStore

        collection_name = spec.cloud_collection_name
        cloud_client = cloud_connection.client()
        store = ChromaVectorStore(
            call_with_retries(
                lambda _: cloud_client.get_or_create_collection(
                    collection_name, metadata=collection_metadata
                )
            )
        )
        max_batch = cloud_client.get_max_batch_size()
        batch_size = min(write_batch_size or max_batch, max_batch)
        retry = call_with_retries

    uploaded = 0
    skipped = 0

    console.print(
        f"Uploading {total} rows from '{local_dir}' to {cloud_connection.describe()}, "
        f"{'index' if isinstance(store, PineconeVectorStore) else 'collection'} "
        f"'{collection_name}'..."
    )

    with Progress(*Progress.get_default_columns(), console=console) as progress:
        task = progress.add_task("Uploading embeddings", total=total)
        for covered, page in pages(local_collection):
            # IDs missing from the local index count as done.
            progress.advance(task, covered - len(page["ids"]))
            for start in range(0, len(page["ids"]), batch_size):
                end = start + batch_size
                ids = page["ids"][start:end]
                present = retry(lambda _: store.existing_ids(ids))
                keep = [
                    i for i, identifier in enumerate(ids) if identifier not in present
                ]
                skipped += len(ids) - len(keep)

                if keep:
                    embeddings = np.asarray(page["embeddings"][start:end])[keep]
                    documents = page["documents"][start:end]
                    metadatas = page["metadatas"][start:end]
                    batch: dict[str, Any] = {
                        "ids": [ids[i] for i in keep],
                        "embeddings": embeddings,
                        "documents": [documents[i] for i in keep],
                        "metadatas": [metadatas[i] for i in keep],
                    }
                    # A retried add may already have reached the server; the
                    # store upserts on retries, which is idempotent.
                    retry(lambda attempt: store.write(attempt=attempt, **batch))
                    uploaded += len(keep)

                progress.advance(task, len(ids))

    del local_collection, local_client
    release_chromadb_clients()

    console.print(
        f"Uploaded {uploaded} rows; {skipped} were already in the destination."
    )
    record_vector_database(embedding_model_name, to, collection_name)  # type: ignore[arg-type]
    return UploadSummary(
        collection_name=collection_name,
        total=total,
        uploaded=uploaded,
        skipped=skipped,
    )


def _open_or_create_pinecone_index(
    connection: PineconeConnection,
    index_name: str,
    local_collection: Any,
    collection_metadata: dict[str, Any] | None,
) -> PineconeVectorStore:
    """Open the destination index of an upload, creating it like the local one."""
    client = connection.client()
    if not pinecone_index_exists(client, index_name):
        first = local_collection.get(limit=1, include=["embeddings"])
        if not first["ids"]:
            raise ValueError(
                "The local collection is empty; there is nothing to upload."
            )
        create_pinecone_index(
            client,
            index_name,
            dimension=len(first["embeddings"][0]),
            cloud=connection.cloud,
            region=connection.region,
            tags=collection_metadata,
        )

    store = PineconeVectorStore.open(client, index_name)
    local_dtype = (collection_metadata or {}).get(COLLECTION_DTYPE_KEY)
    remote_dtype = store.index_metadata().get(COLLECTION_DTYPE_KEY)
    if local_dtype != remote_dtype:
        raise ValueError(
            f"The Pinecone index '{index_name}' records embedding precision "
            f"{remote_dtype!r}, but the local database has {local_dtype!r}. "
            "Uploading would mix precisions in one index; delete the index first."
        )
    return store
