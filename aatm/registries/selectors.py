"""
Register and instantiate selector implementations used by the package.

This module defines a registry of available selector components, including
their public names, implementation classes, and default initialization
arguments. It also provides a helper function for constructing selector
instances by name with optional runtime argument overrides.

The registry allows selectors to be referenced declaratively from task
configurations, CLI options, or other factory-based workflows.
"""

from dataclasses import dataclass
from typing import Any
import copy

from aatm.selectors import (
    BaseSelector,
    FirstResultSelector,
    GeminiLLMSelector,
    OpenAILLMSelector,
)


@dataclass(slots=True, frozen=True)
class SelectorRegistryEntry:
    """Store the specification required to instantiate a selector.

    Instances of this dataclass define a registry entry for a selector,
    including its public name, the selector class to instantiate, and the
    default keyword arguments used during construction.
    """

    name: str
    """Unique registry name used to identify the selector."""

    selector_class: type[BaseSelector]
    """Selector class or constructor used to create the selector instance."""

    kwargs: dict[str, Any]
    """Default keyword arguments passed when instantiating the selector."""


SELECTORS_SPECS = [
    SelectorRegistryEntry(
        name="first-result-selector",
        selector_class=FirstResultSelector,
        kwargs={},
    ),
    SelectorRegistryEntry(
        name="gpt-5.2",
        selector_class=OpenAILLMSelector,
        kwargs={
            "model_id": "gpt-5.2",
        },
    ),
    SelectorRegistryEntry(
        name="gpt-5",
        selector_class=OpenAILLMSelector,
        kwargs={
            "model_id": "gpt-5",
        },
    ),
    SelectorRegistryEntry(
        name="gpt-5-nano",
        selector_class=OpenAILLMSelector,
        kwargs={
            "model_id": "gpt-5-nano",
        },
    ),
    SelectorRegistryEntry(
        name="gpt-5-mini",
        selector_class=OpenAILLMSelector,
        kwargs={
            "model_id": "gpt-5-mini",
        },
    ),
    SelectorRegistryEntry(
        name="gemini-3-pro-preview",
        selector_class=GeminiLLMSelector,
        kwargs={
            "model_id": "gemini-3-pro-preview",
        },
    ),
    SelectorRegistryEntry(
        name="gemini-3-flash-preview",
        selector_class=GeminiLLMSelector,
        kwargs={
            "model_id": "gemini-3-flash-preview",
        },
    ),
    SelectorRegistryEntry(
        name="gemini-2.5-flash",
        selector_class=GeminiLLMSelector,
        kwargs={
            "model_id": "gemini-2.5-flash",
        },
    ),
    SelectorRegistryEntry(
        name="gemini-2.5-flash-lite",
        selector_class=GeminiLLMSelector,
        kwargs={
            "model_id": "gemini-2.5-flash-lite",
        },
    ),
    SelectorRegistryEntry(
        name="gemini-2.5-pro",
        selector_class=GeminiLLMSelector,
        kwargs={
            "model_id": "gemini-2.5-pro",
        },
    ),
]

SELECTORS_REGISTRY: dict[str, SelectorRegistryEntry] = {
    entry.name: entry for entry in SELECTORS_SPECS
}


def load_selector(name: str, **kwargs: Any) -> BaseSelector:
    """Load and instantiate a selector from the registry.

    This function looks up a selector specification by name, copies its default
    keyword arguments, applies any user-provided overrides, and returns a new
    selector instance.

    Args:
        name: Registry name of the selector to instantiate.
        **kwargs: Keyword arguments that override or extend the default
            constructor arguments stored in the registry entry.

    Returns:
        A newly instantiated selector corresponding to the requested registry
            entry.

    Raises:
        ValueError: If no selector with the given name exists in the registry.
        TypeError: If the provided arguments are invalid for the selector
            constructor.
    """
    try:
        spec = SELECTORS_REGISTRY[name]
    except KeyError as e:
        available = ", ".join(sorted(SELECTORS_REGISTRY))
        raise ValueError(
            f"Selector '{name}' not found. Available selectors: {available}"
        ) from e

    selector_kwargs = copy.deepcopy(spec.kwargs)
    selector_kwargs.update(kwargs)

    try:
        return spec.selector_class(**selector_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Invalid arguments while instantiating selector '{name}': {e}"
        ) from e
