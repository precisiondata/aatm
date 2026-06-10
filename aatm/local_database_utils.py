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
import logging
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Hashable, List, Tuple

import pandas as pd
import questionary
import torch
from chromadb import Collection, Embeddings
from rich.console import Console
from rich.progress import track

from aatm.logs import get_logger
from aatm.registries.retrievers import CHROMADB_RETRIEVER_MODEL_REGISTRY
from aatm.data_models import ExpressionMetadata
from aatm.timing_profiler import TimingProfiler

logger = get_logger(__name__, level=logging.DEBUG)
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


def estimate_batch_size(embedding_model_name: str) -> int:
    """Estimate a suitable batch size for local GPU embedding generation.

    Args:
        embedding_model_name: Registry key for the embedding model.

    Returns:
        Estimated maximum batch size.

    Raises:
        ValueError: If the model is unknown, non-local, or batch size cannot be
        estimated.
    """
    retriever_spec = CHROMADB_RETRIEVER_MODEL_REGISTRY.get(embedding_model_name)

    if retriever_spec is None:
        raise ValueError(f"Unknown embedding model: {embedding_model_name}")

    embedding_function = retriever_spec.embedding_function_cls(
        model=retriever_spec.model_id
    )

    if not getattr(embedding_function, "is_local", False):
        raise ValueError(f"Embedding model {embedding_model_name} is not a local model")

    model = embedding_function.model  # type: ignore[attr-defined]

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
            f"Unable to estimate memory per batch item for {embedding_model_name}. "
            f"Measured peak reserved memory did not increase."
        )

    memory_per_batch_item_gib = peak_reserved_gib_diff / batch_size_diff

    max_batch_size = int(
        (free_bytes / 1024**3 - run_time_overhead_gib) / memory_per_batch_item_gib
    )

    del model
    del embedding_function

    gpu_memory_cleanup()

    if max_batch_size > 0:
        return max_batch_size

    raise ValueError(
        f"Unable to estimate batch size for embedding model {embedding_model_name}"
    )


def build_local_vector_database(
    embedding_model_name: str,
    vector_db_dir: Path | None = None,
    rate_limit: int | None = None,
    batch_size: int = 100,
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
        batch_size: Number of records to process and add to the vector database
            in each batch.
        profile_timing: Whether to log per-stage timing information.
        profile_sync_cuda: Whether to synchronize CUDA around timed blocks.
            This makes GPU timings more accurate but adds overhead.
        profile_summary_every_n_batches: Log cumulative profile summary every N
            processed batches.

    Returns:
        None.
    """
    from aatm.registries.retrievers import (
        CHROMADB_RETRIEVER_MODEL_REGISTRY as model_registry,
    )

    profiler = TimingProfiler(
        enabled=profile_timing,
        synchronize_cuda=profile_sync_cuda,
    )

    processed_batches = 0
    added_documents = 0

    if vector_db_dir is not None and not vector_db_dir.exists():
        raise ValueError("vector_db_dir does not exist")

    if vector_db_dir is None:
        vector_db_dir = Path(model_registry[embedding_model_name].chromadb_path)

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

    with profiler.time_block("embedding_function_init"):
        embedding_function = model_registry[
            embedding_model_name
        ].embedding_function_cls(model=model_registry[embedding_model_name].model_id)

    if getattr(embedding_function, "is_local", False) and torch.cuda.is_available():
        with profiler.time_block("estimate_batch_size"):
            batch_size = estimate_batch_size(embedding_model_name)

    embedding_function.batch_size = batch_size  # type: ignore[attr-defined]

    with profiler.time_block("get_or_create_collection"):
        collection: Collection = client.get_or_create_collection(
            model_registry[embedding_model_name].collection_name,
            embedding_function=embedding_function,
        )

    datasets_base_path = Path(".aatm/datasets")

    for dataset_path in datasets_base_path.glob("*.csv"):
        expression_origin = dataset_path.stem

        with profiler.time_block("read_csv"):
            df = pd.read_csv(dataset_path, low_memory=False)

        with profiler.time_block("drop_duplicates_dropna"):
            df = df.drop_duplicates().dropna()

        for row_start in track(
            range(0, len(df), batch_size),
            description=f"Adding embeddings for {expression_origin}",
        ):
            batch_start = time.perf_counter()

            with profiler.time_block("slice_dataframe_to_records"):
                records: list[dict[Hashable, Any]] = df.iloc[
                    row_start : row_start + batch_size
                ].to_dict("records")

            with profiler.time_block("build_expression_metadata"):
                expression_metadatas: list[ExpressionMetadata] = [
                    ExpressionMetadata(**record, expression_origin=expression_origin)  # type: ignore[arg-type]
                    for record in records
                ]

            with profiler.time_block("deduplicate_batch_ids"):
                unique_pairs: list[tuple[str, ExpressionMetadata]] = []
                seen: set[str] = set()

                for metadata in expression_metadatas:
                    if metadata.expression_id in seen:
                        continue

                    if metadata.expression_id is None:
                        raise ValueError(
                            f"expression_id cannot be None. Given: {metadata.model_dump()}"
                        )

                    seen.add(metadata.expression_id)
                    unique_pairs.append((metadata.expression_id, metadata))

                ids = [identifier for identifier, _ in unique_pairs]

                assert all(identifier is not None for identifier in ids), (
                    f"Missing identifier in {expression_origin}"
                )

            with profiler.time_block("chromadb_get_existing_ids"):
                results = collection.get(ids=ids)  # type: ignore[arg-type]
                found_ids = set(results["ids"])

            with profiler.time_block("filter_existing_ids"):
                filtered_unique_pairs: List[Tuple[str, ExpressionMetadata]] = [
                    (identifier, metadata)
                    for identifier, metadata in unique_pairs
                    if identifier not in found_ids and identifier is not None
                ]

            n_rows = len(records)
            n_unique = len(unique_pairs)
            n_new = len(filtered_unique_pairs)

            if not filtered_unique_pairs:
                batch_elapsed = time.perf_counter() - batch_start

                profiler.log_batch(
                    expression_origin=expression_origin,
                    row_start=row_start,
                    n_rows=n_rows,
                    n_unique=n_unique,
                    n_new=0,
                    batch_elapsed=batch_elapsed,
                )

                del records
                del expression_metadatas
                del unique_pairs
                del ids
                del results
                del found_ids
                del filtered_unique_pairs

                processed_batches += 1

                if (
                    profile_timing
                    and profile_summary_every_n_batches > 0
                    and processed_batches % profile_summary_every_n_batches == 0
                ):
                    profiler.log_summary(
                        label=f"partial after {processed_batches} batches"
                    )

                continue

            if rate_limit is not None:
                with profiler.time_block("rate_limiter_total"):
                    next_allowed_time, slept_seconds = rate_limiter(
                        n_docs=len(filtered_unique_pairs),
                        rate_limit=rate_limit,
                        next_allowed_time=next_allowed_time,
                    )

                profiler.add("rate_limiter_sleep", slept_seconds)

            with profiler.time_block("prepare_embedding_input_texts"):
                texts_to_embed = [
                    metadata.expression for _, metadata in filtered_unique_pairs
                ]

            with profiler.time_block("embedding_function_call"):
                embeddings = embedding_function(texts_to_embed)

            with profiler.time_block("ensure_cpu_embeddings"):
                embeddings = ensure_cpu_embeddings(embeddings)

            with profiler.time_block("prepare_chromadb_payload"):
                add_ids = [identifier for identifier, _ in filtered_unique_pairs]
                add_documents = [
                    metadata.expression for _, metadata in filtered_unique_pairs
                ]
                add_metadatas = [
                    metadata.to_dict() for _, metadata in filtered_unique_pairs
                ]

            with profiler.time_block("chromadb_add"):
                collection.add(
                    ids=add_ids,
                    documents=add_documents,  # type: ignore[misc,arg-type]
                    embeddings=embeddings,
                    metadatas=add_metadatas,  # type: ignore[arg-type]
                )

            with profiler.time_block("delete_batch_objects"):
                del embeddings
                del records
                del expression_metadatas
                del unique_pairs
                del ids
                del results
                del found_ids
                del filtered_unique_pairs
                del texts_to_embed
                del add_ids
                del add_documents
                del add_metadatas

            batch_elapsed = time.perf_counter() - batch_start

            processed_batches += 1
            added_documents += n_new

            profiler.log_batch(
                expression_origin=expression_origin,
                row_start=row_start,
                n_rows=n_rows,
                n_unique=n_unique,
                n_new=n_new,
                batch_elapsed=batch_elapsed,
            )

            if (
                profile_timing
                and profile_summary_every_n_batches > 0
                and processed_batches % profile_summary_every_n_batches == 0
            ):
                profiler.log_summary(label=f"partial after {processed_batches} batches")

            if processed_batches % 50 == 0:
                gpu_memory_cleanup()

        del df

        with profiler.time_block("dataset_cleanup"):
            gpu_memory_cleanup(synchronize=False)

    with profiler.time_block("final_delete_objects"):
        del collection
        del embedding_function
        del client

    with profiler.time_block("final_gpu_memory_cleanup"):
        gpu_memory_cleanup(synchronize=True)

    profiler.log_summary(label="final")
