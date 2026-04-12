"""
Register and instantiate reranker implementations used by the package.

This module defines a lightweight registry for available reranker components,
including their human-readable names, implementation classes, and default
initialization arguments. It also provides a helper function to construct
reranker instances by name with optional runtime overrides.

The registry centralizes reranker configuration so that rerankers can be
referenced declaratively from task configurations, CLI options, or other
factory-based workflows.
"""

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
    """Store the specification required to instantiate a reranker.

    Instances of this dataclass define a registry entry for a reranker,
    including its public name, the reranker class to instantiate, and the
    default keyword arguments used during construction.
    """

    name: str
    """Unique registry name used to identify the reranker."""
    reranker_class: BaseReranker
    """Reranker class or constructor used to create the reranker instance."""
    kwargs: dict[str, Any]
    """Default keyword arguments passed when instantiating the reranker."""


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


def load_reranker(name: str, **kwargs: Any) -> BaseReranker:
    """Load and instantiate a reranker from the registry.

    This function looks up a reranker specification by name, copies its default
    keyword arguments, applies any user-provided overrides, and returns a new
    reranker instance.

    Args:
        name: Registry name of the reranker to instantiate.
        **kwargs: Keyword arguments that override or extend the default
            constructor arguments stored in the registry entry.

    Returns:
        A newly instantiated reranker corresponding to the requested registry
            entry.

    Raises:
        ValueError: If no reranker with the given name exists in the registry.
        TypeError: If the provided arguments are invalid for the reranker
            constructor.
    """
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
