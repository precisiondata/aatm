"""
Provide the main terminology-mapping orchestration utilities for the package.

This module defines the high-level mapping workflow that converts source
concepts into standardized concepts using a configurable pipeline composed of
translation, retrieval, reranking, and selection stages. It also includes a
rate-limiting helper and utilities for loading mapping tasks from structured
configuration objects.

The main entry point is ``TerminologyMapper``, which can read OMOP-like source
concept files, process them in batches, and write the mapped output to disk.
"""

from pathlib import Path
from selectors import BaseSelector
import time
from typing import Any, List, Literal, Optional, Tuple
from rich.console import Console
from rich.progress import track
from tqdm import tqdm
import chromadb
import pandas as pd

from aatm.api.data_models import TerminologyMappingRequest
from aatm.pipeline import PipelineBaseClass
from aatm.data_models import (
    MappedSourceConcept,
    SelectorResults,
    SourceConcept,
    TerminologyMappingTask,
    Translation,
)
from aatm.registries.rerankers import load_reranker
from aatm.registries.retrievers import load_retriever
from aatm.registries.selectors import load_selector
from aatm.registries.translators import load_translator
from aatm.rerankers import BaseReranker
from aatm.translators import BaseTranslator, EmptyTranslator
from aatm.retrievers import BaseRetriever, ChromaDBRetriever
from aatm.selectors import FirstResultSelector
from aatm.embedding_functions import GoogleEmbeddingFunction


console = Console()


def rate_limit(n_docs: int, next_allowed_time: float, rate_limit: int) -> None:
    """Apply a document-based rate limit and return the next allowed time.

    This helper delays execution when necessary to ensure that processing does
    not exceed the configured throughput in documents per minute. It uses a
    monotonic clock to avoid issues caused by system clock adjustments.

    Args:
        n_docs: Number of documents in the current batch.
        next_allowed_time: Monotonic timestamp representing the next permitted
            processing time.
        rate_limit: Maximum number of documents allowed per minute.

    Returns:
        The updated monotonic timestamp indicating when the next batch may be
            processed.

    Notes:
        The return annotation in the current implementation is ``None``, but
        the function actually returns the updated next allowed time.
    """
    now = time.monotonic()
    if now < next_allowed_time:
        time.sleep(next_allowed_time - now)
        now = time.monotonic()
    # Reserve time for this batch
    next_allowed_time = max(next_allowed_time, now) + n_docs * 60 / rate_limit
    return next_allowed_time


class TerminologyMapper:
    """Coordinate end-to-end terminology mapping through a configurable pipeline.

    This class orchestrates the full mapping workflow for source concepts,
    including optional translation, candidate retrieval, reranking, final
    selection, batching, rate limiting, and output generation. It is designed
    to work with OMOP-style source concept files and pluggable pipeline
    components.

    Instances can be created directly by passing configured components or built
    from a structured ``TerminologyMappingTask`` configuration object.
    """

    def __init__(
        self,
        input_file: Optional[str | Path] = None,
        output_dir: str | Path = Path("output"),
        translator: Optional[BaseTranslator | str] = None,
        retriever: Optional[BaseRetriever | str] = None,
        selector: Optional[BaseSelector | str] = None,
        reranker: Optional[BaseReranker | str] = None,
        batch_size: int = 100,
        rate_limit: Optional[int] = None,
        column_mapping: Optional[dict] = None,
        limit_to: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the terminology mapper and its pipeline components.

        This constructor sets up the terminology-mapping pipeline, defaulting to
        built-in translator, retriever, selector, and reranker behaviors when
        custom components are not provided. It also prepares output paths,
        stores batching and rate-limiting settings, and defines the expected
        input schema for source concept files.

        Args:
            input_file: Optional path to the source concept file to map.
            output_dir: Directory where mapping outputs will be written.
            translator: Optional translator component used before retrieval. Expects a BaseTranslator or the translator id in the registry.
            retriever: Optional retriever component used to fetch candidate
                concepts. Expects a BaseRetriever or the retriever id in the registry.
            selector: Optional selector component used to choose the final
                mapped concept. Expects a BaseSelector or the selector id in the registry.
            reranker: Optional reranker component used to reorder retrieved
                candidates before selection. Expects a BaseReranker or the reranker id in the registry.
            batch_size: Number of source concepts to process per batch.
            rate_limit: Optional maximum number of items to process per minute.
            column_mapping: Optional mapping from input column names to the
                expected OMOP-style column names.
            limit_to: Optional maximum number of input rows to process.
            *args: Additional positional arguments reserved for compatibility.
            **kwargs: Additional keyword arguments reserved for compatibility.

        Returns:
            None.
        """
        # Define translator
        if translator is None:
            self.translator = EmptyTranslator()

        elif isinstance(translator, str):
            self.translator = load_translator(translator)

        elif isinstance(BaseTranslator, translator):
            self.translator = translator

        else:
            raise TypeError(
                f"Translator must be a BaseTranslator or str representing the translator name.  Given type: {type(translator)}."
            )

        # Define retriever
        if retriever is None:
            client = chromadb.PersistentClient()
            self.retriever = ChromaDBRetriever(
                client=client,
                collection_name="expressions",
                embedding_function=GoogleEmbeddingFunction(
                    model="gemini-embedding-001"
                ),
            )

        elif isinstance(retriever, str):
            self.retriever = load_retriever(retriever)

        elif isinstance(BaseRetriever, retriever):
            self.retriever = retriever

        else:
            raise TypeError(
                f"Retriever must be a BaseRetriever or str representing the retriever name.  Given type: {type(retriever)}."
            )

        # Define selector
        if selector is None:
            self.selector = FirstResultSelector()

        elif isinstance(selector, str):
            self.selector = load_selector(selector)

        elif isinstance(BaseSelector, selector):
            self.selector = selector

        else:
            raise TypeError(
                f"Selector must be a BaseSelector or str representing the selector name.  Given type: {type(selector)}."
            )

        # Define reranker
        if reranker is None:
            self.reranker = PipelineBaseClass()

        elif isinstance(reranker, str):
            self.reranker = load_reranker(reranker)

        elif isinstance(BaseReranker, reranker):
            self.reranker = reranker

        else:
            raise TypeError(
                f"Reranker must be a BaseReranker or str representing the reranker name.  Given type: {type(reranker)}."
            )

        # Other attributes
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        self.input_file: Optional[Path] = Path(input_file) if input_file else None
        self.output_dir: Path = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size: int = batch_size
        self.rate_limit: Optional[int] = rate_limit
        self.expected_columns: set[str] = set(
            [
                "source_code",
                "source_concept_id",
                "source_vocabulary_id",
                "source_code_description",
                "valid_start_date",
                "valid_end_date",
                "invalid_reason",
            ]
        )
        self.column_mapping = column_mapping
        self.limit_to = limit_to

    @classmethod
    def from_task_config(
        cls, task_config: TerminologyMappingTask
    ) -> "TerminologyMapper":
        """Create a terminology mapper from a task configuration object.

        This factory method resolves the configured translator, retriever,
        selector, and reranker from their registries and initializes a
        ``TerminologyMapper`` with the remaining task parameters.

        Args:
            task_config: Structured task configuration describing the mapping
                pipeline and runtime settings.

        Returns:
            A configured ``TerminologyMapper`` instance.
        """
        translator = (
            load_translator(task_config.translator_id)
            if task_config.translator_id
            else None
        )

        retriever = (
            load_retriever(task_config.retriever_id)
            if task_config.retriever_id
            else None
        )

        selector = (
            load_selector(task_config.selector_id) if task_config.selector_id else None
        )

        reranker = (
            load_reranker(task_config.reranker_id) if task_config.reranker_id else None
        )

        return cls(
            translator=translator,
            retriever=retriever,
            selector=selector,
            reranker=reranker,
            batch_size=task_config.batch_size,
            rate_limit=task_config.rate_limit,
            input_file=task_config.input_file,
            output_dir=task_config.output_dir,
            column_mapping=task_config.column_mapping,
            limit_to=task_config.limit_to,
        )

    @classmethod
    def from_task_request(
        cls, task_request: TerminologyMappingRequest, *args, **kwargs
    ) -> "TerminologyMapper":
        return cls(
            translator=task_request.translator_id,
            retriever=task_request.retriever_id,
            selector=task_request.selector_id,
            reranker=task_request.reranker_id,
            *args,
            **kwargs,
        )

    def map(
        self,
        expressions: Optional[List[str | SourceConcept]] = None,
        file_path: str | Path = None,
        limit_to: int = None,
        output_dir: str | Path = None,
        return_as: Literal["df", "mapped_source_concepts"] = "df",
        save_to_disk: bool = True,
    ) -> pd.DataFrame:
        """Map source concepts from a file to standardized concepts.

        This method loads source concepts from a supported input file or from a list of strings or SourceConcept objects, processes them in batches through the pipeline, and returns the mapped results as a DataFrame. The resulting mappings are also written to a CSV file in the output directory.

        Args:
            expressions: Optional list of expressions to map. Expects a list of
                strings or SourceConcept objects.
            file_path: Optional path to the source concept file. If not
                provided, the mapper's configured input file is used.
            limit_to: Optional maximum number of rows to process from the
                source file.
            output_dir: Optional output directory override for this mapping
                operation.
            return_as: Return type. Options: "df" or
                "mapped_source_concepts".
            save_to_disk: Whether to save the results to disk.

        Returns:
            A pandas DataFrame containing the mapped source concepts.

        Raises:
            ValueError: If no file path is available or the file type is not
                supported.
        """
        file_path = self.input_file if file_path is None else file_path
        limit_to = self.limit_to if limit_to is None else limit_to

        if expressions is not None:
            assert all(
                isinstance(e, str) or isinstance(e, SourceConcept) for e in expressions
            ), "Expressions must be either strings or SourceConcept objects."

            source_concepts = [
                SourceConcept(source_code_description=e) if isinstance(e, str) else e
                for e in expressions
            ]

        elif file_path is not None:
            if isinstance(file_path, str):
                file_path = Path(file_path)
            if file_path.suffix == ".csv":
                source_concepts = self.map_csv_to_source_concepts(file_path, limit_to)
            else:
                raise ValueError(
                    f"Unsupported file type: {file_path.suffix}. File path provided: {file_path}"
                )
        else:
            raise ValueError("No file path provided")

        assert return_as in ["df", "mapped_source_concepts"], (
            "Parameter return_as must be either 'df' or 'mapped_source_concepts'."
        )

        # map source concepts
        mapped_source_concepts: List[MappedSourceConcept] = []
        translated_source_concepts: List[Translation] = []
        next_allowed_time = time.monotonic()
        for batch_idx in track(
            range(0, len(source_concepts), self.batch_size),
            description=f"Mapping source concepts (batch_size = {self.batch_size})",
        ):
            batch = source_concepts[batch_idx : batch_idx + self.batch_size]

            # rate limit if defined
            if self.rate_limit is not None:
                next_allowed_time = rate_limit(
                    len(batch), next_allowed_time, self.rate_limit
                )

            translated_batch: List[Translation] = batch | self.translator
            selected_source_concepts: SelectorResults = (
                translated_batch | self.retriever | self.reranker | self.selector
            )
            translated_source_concepts.extend(translated_batch)
            mapped_source_concepts.extend(
                MappedSourceConcept.from_selector_results(
                    batch, selected_source_concepts, translated_batch
                )
            )

        mapped_source_concepts_df = None
        if save_to_disk:
            # write mapped source concepts to csv
            mapped_source_concepts_df = pd.DataFrame(
                [
                    mapped_source_concept.to_dict()
                    for mapped_source_concept in mapped_source_concepts
                ]
            )

            output_dir = output_dir if output_dir is not None else self.output_dir

            mapped_source_concepts_df.to_csv(
                self.output_dir / "mapped_source_concepts.csv", index=False
            )

        if return_as == "df":
            if mapped_source_concepts_df is None:
                mapped_source_concepts_df = pd.DataFrame(
                    [
                        mapped_source_concept.to_dict()
                        for mapped_source_concept in mapped_source_concepts
                    ]
                )
            return mapped_source_concepts_df

        elif return_as == "mapped_source_concepts":
            return mapped_source_concepts

    async def amap(
        self,
        file_path: str | Path = None,
        limit_to: int = None,
        return_confidence_scores: bool = True,
        output_dir: str | Path = None,
    ) -> Tuple[pd.DataFrame, List[float]] | pd.DataFrame:
        """Asynchronously map source concepts from a file to standardized concepts.

        This method is the asynchronous counterpart to ``map()``. It processes
        source concepts in batches through the configured pipeline using async
        calls where supported, then writes the mapped results to a CSV file and
        returns them as a DataFrame.

        Args:
            file_path: Optional path to the source concept file.
            limit_to: Optional maximum number of rows to process from the
                source file.
            return_confidence_scores: Whether to include confidence scores in
                the returned output DataFrame.
            output_dir: Optional output directory override for this mapping
                operation.

        Returns:
            A pandas DataFrame containing the mapped source concepts. The
            current annotation allows for a tuple including confidence scores,
            but the present implementation returns only the DataFrame.

        Raises:
            ValueError: If no file path is provided or the file type is not
                supported.
        """
        if file_path is not None:
            if isinstance(file_path, str):
                file_path = Path(file_path)
            if file_path.suffix == ".csv":
                source_concepts = self.map_csv_to_source_concepts(file_path, limit_to)
            else:
                raise ValueError(
                    f"Unsupported file type: {file_path.suffix}. File path provided: {file_path}"
                )
        else:
            raise ValueError("No file path provided")

        # map source concepts
        mapped_source_concepts: List[MappedSourceConcept] = []
        confidence_scores = []
        next_allowed_time = time.monotonic()
        for batch_idx in tqdm(
            range(0, len(source_concepts), self.batch_size),
            desc=f"Mapping source concepts (batch_size = {self.batch_size})",
        ):
            batch = source_concepts[batch_idx : batch_idx + self.batch_size]

            # rate limit if defined
            if self.rate_limit is not None:
                next_allowed_time = rate_limit(
                    len(batch), next_allowed_time, self.rate_limit
                )

            translated_batch = await (batch | self.translator)

            selected_source_concepts: SelectorResults = await (
                translated_batch | self.retriever | self.reranker | self.selector
            )
            mapped_source_concepts.extend(
                MappedSourceConcept.from_selector_results(
                    batch, selected_source_concepts
                )
            )
            confidence_scores.extend(
                [
                    1 - (selected_source_concepts.results[i].distance or 0)
                    if selected_source_concepts.results[i] is not None
                    else None
                    for i in range(len(selected_source_concepts.results))
                ]
            )

        # write mapped source concepts to csv
        mapped_source_concepts_df = pd.DataFrame(
            [
                mapped_source_concept.to_dict()
                for mapped_source_concept in mapped_source_concepts
            ]
        )

        if return_confidence_scores:
            mapped_source_concepts_df["confidence_score"] = confidence_scores

        output_dir = output_dir if output_dir is not None else self.output_dir

        mapped_source_concepts_df.to_csv(
            self.output_dir / "mapped_source_concepts.csv", index=False
        )

        return mapped_source_concepts_df

    def map_csv_to_source_concepts(
        self, file_path: Path, limit_to: int = None
    ) -> List[SourceConcept]:
        """Load source concepts from a CSV file and convert them to model objects.

        This method reads a CSV file, optionally limits the number of rows,
        applies any configured column renaming, validates that the required
        OMOP-style columns are present, drops rows with missing source concept
        descriptions, and converts the remaining rows into ``SourceConcept``
        objects.

        Args:
            file_path: Path to the CSV file containing source concepts.
            limit_to: Optional maximum number of rows to load from the file.

        Returns:
            A list of ``SourceConcept`` objects created from the CSV rows.

        Raises:
            ValueError: If the input file does not contain the required columns
                after optional column remapping.
        """
        df = pd.read_csv(file_path, on_bad_lines="skip")

        if limit_to is not None:
            df = df.iloc[:limit_to]

        # Rename columns
        if self.column_mapping is not None:
            df = df.rename(columns=self.column_mapping)

        # Check if all expected columns are present
        if not self.expected_columns.issubset(set(df.columns)):
            raise ValueError(
                f"This function expects a SOURCE_TO_CONCEPT_MAP table as defined by the official OMOP Common Data Model. It must include the following columns: {self.expected_columns}. Please, either edit the column names or provide the column_mapping argument containing the mapping between the current column names and the expected column names. Got columns: {set(df.columns)}"
            )

        # Check for null values
        if df["source_code_description"].isnull().any():
            console.print(
                f"[yellow]Attention:[/yellow] There are {df['source_code_description'].isnull().sum()} null values in the source_code_description column. Those rows will be dropped."
            )
            console.print(
                f"Dropped rows: {df[df['source_code_description'].isnull()].index.to_list()}"
            )
            df = df.dropna(subset=["source_code_description"])

        df = df.astype(str).fillna("")
        # Convert to SourceConcept objects
        source_concepts: List[SourceConcept] = [
            SourceConcept(**row) for row in df.to_dict("records")
        ]

        return source_concepts

    def __call__(self, expression: str) -> str:
        """Invoke the mapper as a callable object.

        This method delegates to ``map()`` so that mapper instances can be used
        like callable pipeline components.

        Args:
            expression: Input expression or file reference to map.

        Returns:
            The result of calling ``map()`` with the provided input.

        Notes:
            The current type annotation suggests a string input and string
            output, but the underlying ``map()`` method expects file-based
            input and returns a DataFrame.
        """
        return self.map(expression)

    def __repr__(self) -> str:
        """Return the official string representation of the mapper.

        Returns:
            A string representation of the ``TerminologyMapper`` instance.
        """
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        """Return a human-readable string representation of the mapper.

        Returns:
            A string representation of the ``TerminologyMapper`` instance.
        """
        return f"{self.__class__.__name__}()"
