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
    "Specification for a selector instance"

    name: str
    selector_class: BaseSelector
    kwargs: dict[str, Any]


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


def load_selector(name: str, **kwargs) -> BaseSelector:
    try:
        selector_kwargs: dict = copy.deepcopy(SELECTORS_REGISTRY[name].kwargs)
        selector_kwargs.update(kwargs)  # override with user provided kwargs
        return SELECTORS_REGISTRY[name].selector_class(**selector_kwargs)
    except KeyError:
        raise KeyError(
            f"Selector '{name}' not found. Available selectors: {list(SELECTORS_REGISTRY.keys())}"
        )
    except Exception as e:
        raise e(f"Unexpected error while loading selector '{name}': {e}")
