from pathlib import Path
from selectors import BaseSelector
import time
from typing import List, Optional, Tuple
from tqdm import tqdm
import chromadb
import pandas as pd

from aatm.pipeline import PipelineBaseClass
from aatm.data_models import MappedSourceConcept, SelectorResults, SourceConcept
from aatm.rerankers import BaseReranker
from aatm.translators import BaseTranslator, GeminiTranslator
from aatm.retrievers import BaseRetriever, ChromaDBRetriever
from aatm.selectors import FirstResultSelector
from aatm.embedding_functions import GoogleEmbeddingFunction


def rate_limit(n_docs: int, next_allowed_time: float, rate_limit: int) -> None:
    now = time.monotonic()
    if now < next_allowed_time:
        time.sleep(next_allowed_time - now)
        now = time.monotonic()
    # Reserve time for this batch
    next_allowed_time = max(next_allowed_time, now) + n_docs * 60 / rate_limit
    return next_allowed_time


class TerminologyMapper:
    def __init__(
        self,
        output_dir: str | Path = Path("output"),
        translator: Optional[BaseTranslator] = None,
        retriever: Optional[BaseRetriever] = None,
        selector: Optional[BaseSelector] = None,
        reranker: Optional[BaseReranker] = None,
        batch_size: int = 100,
        rate_limit: Optional[int] = None,
        *args,
        **kwargs,
    ):
        if translator is None:
            translator = GeminiTranslator(model="gemini-2.5-flash")

        if retriever is None:
            client = chromadb.PersistentClient()
            retriever = ChromaDBRetriever(
                client=client,
                collection_name="expressions",
                embedding_function=GoogleEmbeddingFunction(
                    model="gemini-embedding-001"
                ),
            )

        if selector is None:
            selector = FirstResultSelector()

        if reranker is None:
            reranker = PipelineBaseClass()  # empty reranker

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        self.output_dir: Path = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.translator: BaseTranslator = translator
        self.retriever: BaseRetriever = retriever
        self.selector: BaseSelector = selector
        self.reranker: BaseReranker | PipelineBaseClass = reranker
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

    def map(
        self,
        file_path: str | Path = None,
        limit_to: int = None,
        return_confidence_scores: bool = True,
        output_dir: str | Path = None,
    ) -> Tuple[pd.DataFrame, List[float]] | pd.DataFrame:
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

            selected_source_concepts: SelectorResults = (
                batch | self.translator | self.retriever | self.reranker | self.selector
            )
            mapped_source_concepts.extend(
                MappedSourceConcept.from_selector_results(
                    batch, selected_source_concepts
                )
            )
            confidence_scores.extend(
                [
                    1 - selected_source_concepts.results[i].distance
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

    async def amap(
        self,
        file_path: str | Path = None,
        limit_to: int = None,
        return_confidence_scores: bool = True,
        output_dir: str | Path = None,
    ) -> Tuple[pd.DataFrame, List[float]] | pd.DataFrame:
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

    def map_csv_to_source_concepts(self, file_path: Path, limit_to: int = None) -> str:
        df = pd.read_csv(file_path, on_bad_lines="skip")

        if df["source_code_description"].isnull().any():
            print(
                f"There are {df['source_code_description'].isnull().sum()} null values in the source_code_description column. Those rows will be dropped."
            )
            print(
                f"Dropped rows: {df[df['source_code_description'].isnull()].index.to_list()}"
            )
            df = df.dropna(subset=["source_code_description"])

        if limit_to is not None:
            df = df.iloc[:limit_to]

        # Check if all expected columns are present
        if not self.expected_columns.issubset(set(df.columns)):
            raise ValueError(
                f"Expected columns: {self.expected_columns}. Got columns: {set(df.columns)}"
            )

        df = df.astype(str).fillna("")
        # Convert to SourceConcept objects
        source_concepts: List[SourceConcept] = [
            SourceConcept(**row) for row in df.to_dict("records")
        ]

        return source_concepts

    def __call__(self, expression: str) -> str:
        return self.map(expression)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return f"{self.__class__.__name__}()"
