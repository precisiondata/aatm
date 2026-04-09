import copy
from dataclasses import dataclass
from typing import Any

from aatm.rerankers import (
    BM25Reranker,
    BaseReranker,
    Qwen3Reranker,
    Qwen3RerankerModels,
)


@dataclass(slots=True, frozen=True)
class RerankerRegistryEntry:
    "Specification for a reranker instance"

    name: str
    reranker_class: BaseReranker
    kwargs: dict[str, Any]


RERANKERS_SPECS = [
    RerankerRegistryEntry(
        name="bm25-reranker",
        reranker_class=BM25Reranker,
        kwargs={},
    ),
    RerankerRegistryEntry(
        name="qwen3-reranker-0.6b",
        reranker_class=Qwen3Reranker,
        kwargs={
            "model_id": Qwen3RerankerModels.QWEN3_06B.value,
        },
    ),
    RerankerRegistryEntry(
        name="qwen3-reranker-4b",
        reranker_class=Qwen3Reranker,
        kwargs={
            "model_id": Qwen3RerankerModels.QWEN3_4B.value,
        },
    ),
    RerankerRegistryEntry(
        name="qwen3-reranker-8b",
        reranker_class=Qwen3Reranker,
        kwargs={
            "model_id": Qwen3RerankerModels.QWEN3_8B.value,
        },
    ),
]

RERANKERS_REGISTRY: dict[str, RerankerRegistryEntry] = {
    entry.name: entry for entry in RERANKERS_SPECS
}


def load_reranker(name: str, **kwargs) -> BaseReranker:
    try:
        spec = RERANKERS_REGISTRY[name]
    except KeyError as e:
        available = ", ".join(sorted(RERANKERS_REGISTRY))
        raise ValueError(
            f"Reranker '{name}' not found. Available rerankers: {available}"
        ) from e

    reranker_kwargs = copy.deepcopy(spec.kwargs)
    reranker_kwargs.update(kwargs)

    try:
        return spec.reranker_class(**reranker_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Invalid arguments while instantiating reranker '{name}': {e}"
        ) from e
