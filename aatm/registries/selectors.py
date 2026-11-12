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
    DecisionSelector,
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


def _agentic_selector(**kwargs: Any) -> BaseSelector:
    # Imported on use: the agent pulls in the vocabulary tools.
    from aatm.concept_search.selector import AgenticSelector

    return AgenticSelector(**kwargs)


# The concept search agent (aatm.concept_search): it searches beyond the retrieved
# candidates with the mapper's retriever and the vocabulary database.
SELECTORS_SPECS += [
    SelectorRegistryEntry(
        name=f"agent-{model}",
        selector_class=_agentic_selector,  # type: ignore[arg-type]
        kwargs={"model_id": f"{provider}/{model}"},
    )
    for provider, model in (
        ("openai", "gpt-5-mini"),
        ("openai", "gpt-5"),
        ("openai", "gpt-5.2"),
        ("gemini", "gemini-3-flash-preview"),
        ("gemini", "gemini-2.5-flash"),
    )
]

DECISION_SELECTOR_PREFIX = "decision-"
"""``decision-<decision model id>`` names a `DecisionSelector`."""

# Decision models (aatm.llm.create_decision_model) choosing among the retrieved
# candidates. Any other pinned id works too: "decision-jev-compatible/<model>", or
# "decision-llm/<llm id>" (no probabilities). See `load_selector`.
SELECTORS_SPECS += [
    SelectorRegistryEntry(
        name=f"{DECISION_SELECTOR_PREFIX}{model_id}",
        selector_class=DecisionSelector,
        kwargs={"model": model_id},
    )
    for model_id in (
        "openai/gpt-6-luna",
        "gemini/gemini-decision-1",
        "openrouter/typesafe/jev-1.13",
    )
]

SELECTORS_REGISTRY: dict[str, SelectorRegistryEntry] = {
    entry.name: entry for entry in SELECTORS_SPECS
}


def load_selector(name: str, **kwargs: Any) -> BaseSelector:
    """Load and instantiate a selector from the registry.

    This function looks up a selector specification by name, copies its default
    keyword arguments, applies any user-provided overrides, and returns a new
    selector instance.

    Names starting with ``decision-`` that are not registered are read as
    ``decision-<decision model id>`` (a `DecisionSelector`), e.g.
    ``"decision-jev-compatible/my-model"`` with ``model_kwargs={"base_url": ...}``.

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
    if name not in SELECTORS_REGISTRY and name.startswith(DECISION_SELECTOR_PREFIX):
        # Any decision model id, e.g. "decision-llm/anthropic/claude-haiku-5-5".
        spec = SelectorRegistryEntry(
            name=name,
            selector_class=DecisionSelector,
            kwargs={"model": name[len(DECISION_SELECTOR_PREFIX) :]},
        )
        return _instantiate(spec, kwargs)
    try:
        spec = SELECTORS_REGISTRY[name]
    except KeyError as e:
        available = ", ".join(sorted(SELECTORS_REGISTRY))
        raise ValueError(
            f"Selector '{name}' not found. Available selectors: {available}"
        ) from e
    return _instantiate(spec, kwargs)


def _instantiate(spec: SelectorRegistryEntry, kwargs: dict[str, Any]) -> BaseSelector:
    name = spec.name
    selector_kwargs = copy.deepcopy(spec.kwargs)
    selector_kwargs.update(kwargs)

    try:
        return spec.selector_class(**selector_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Invalid arguments while instantiating selector '{name}': {e}"
        ) from e
