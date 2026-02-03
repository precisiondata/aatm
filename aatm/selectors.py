import dotenv
from abc import ABC, abstractmethod

# Custom modules
from aatm.data_models import (
    EmptySelectionMetadata,
    RetrieverResults,
    SelectedExpressionMetadata,
    SelectorResults,
)
from aatm.pipeline import PipelineBaseClass

# Load environment variables
dotenv.load_dotenv()


class BaseSelector(PipelineBaseClass, ABC):
    @abstractmethod
    def select(self, results: RetrieverResults) -> SelectorResults:
        pass

    def __call__(self, results: RetrieverResults) -> SelectorResults:
        return self.select(results)


class FirstResultSelector(BaseSelector):
    def __init__(self, selection_threshold: float = float("inf"), *args, **kwargs):
        self.selection_threshold = selection_threshold

    def select(self, results: RetrieverResults) -> SelectorResults:
        selected_results = []
        for idx, r in enumerate(results.results):
            try:
                if r[0].distance < self.selection_threshold:
                    selected_results.append(
                        SelectedExpressionMetadata(
                            **r[0].model_dump(), result_list_index=0
                        )
                    )
                else:
                    selected_results.append(EmptySelectionMetadata())
            except IndexError:
                selected_results.append(EmptySelectionMetadata())
                print(f"No results for query: {results.queries[idx]}. Results: {r}")
        return SelectorResults(results=selected_results, queries=results.queries)
