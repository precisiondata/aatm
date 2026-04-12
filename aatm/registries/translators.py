"""
Register and instantiate translator implementations used by the package.

This module defines a lightweight registry for available translator
components, including their public names, implementation classes, and default
initialization arguments. It also provides a helper function to construct
translator instances by name with optional runtime overrides.

The registry centralizes translator configuration so that translators can be
referenced declaratively from task configurations, CLI options, or other
factory-based workflows.
"""

from dataclasses import dataclass
from typing import Any
import copy

from aatm.translators import EmptyTranslator, GeminiTranslator
from aatm.translators import BaseTranslator


@dataclass(slots=True, frozen=True)
class TranslatorRegistryEntry:
    """Store the specification required to instantiate a translator.

    Instances of this dataclass define a registry entry for a translator,
    including its public name, the translator class to instantiate, and the
    default keyword arguments used during construction.
    """

    name: str
    """Unique registry name used to identify the translator."""

    translator_class: BaseTranslator
    """Translator class or constructor used to create the translator instance."""

    kwargs: dict[str, Any]
    """Default keyword arguments passed when instantiating the translator."""


TRANSLATORS_SPECS = [
    TranslatorRegistryEntry(
        name="empty-translator", translator_class=EmptyTranslator, kwargs={}
    ),
    TranslatorRegistryEntry(
        name="gemini-2.5-flash",
        translator_class=GeminiTranslator,
        kwargs={"model": "gemini-2.5-flash"},
    ),
]

TRANSLATORS_REGISTRY = {entry.name: entry for entry in TRANSLATORS_SPECS}


def load_translator(name: str, **kwargs: Any) -> BaseTranslator:
    """Load and instantiate a translator from the registry.

    This function looks up a translator specification by name, copies its
    default keyword arguments, applies any user-provided overrides, and
    returns a new translator instance.

    Args:
        name: Registry name of the translator to instantiate.
        **kwargs: Keyword arguments that override or extend the default
            constructor arguments stored in the registry entry.

    Returns:
        A newly instantiated translator corresponding to the requested
            registry entry.

    Raises:
        ValueError: If no translator with the given name exists in the
            registry.
        TypeError: If the provided arguments are invalid for the translator
            constructor.
    """
    try:
        spec = TRANSLATORS_REGISTRY[name]
    except KeyError as e:
        available = ", ".join(sorted(TRANSLATORS_REGISTRY))
        raise ValueError(
            f"Translator '{name}' not found. Available translators: {available}"
        ) from e

    translator_kwargs = copy.deepcopy(spec.kwargs)
    translator_kwargs.update(kwargs)

    try:
        return spec.translator_class(**translator_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Invalid arguments while instantiating translator '{name}': {e}"
        ) from e
