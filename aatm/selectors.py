import dotenv
from abc import ABC, abstractmethod

# Custom modules
from aatm.data_models import (
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
    def select(self, results: RetrieverResults) -> SelectorResults:
        selected_results = [
            SelectedExpressionMetadata(**r[0].model_dump(), result_list_index=0)
            for r in results.results
        ]
        return SelectorResults(results=selected_results, queries=results.queries)
