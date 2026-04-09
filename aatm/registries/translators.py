from dataclasses import dataclass
from typing import Any
import copy

from .translators import EmptyTranslator, GeminiTranslator
from .translators import BaseTranslator


@dataclass(slots=True, frozen=True)
class TranslatorRegistryEntry:
    "Specification for a translator instance"

    name: str
    translator_class: BaseTranslator
    kwargs: dict[str, Any]


TRANSLATORS_SPECS = [
    TranslatorRegistryEntry(name="empty-translator", translator_class=EmptyTranslator),
    TranslatorRegistryEntry(
        name="gemini-2.5-flash",
        translator_class=GeminiTranslator,
        kwargs={"model": "gemini-2.5-flash"},
    ),
]

TRANSLATORS_REGISTRY = {entry.name: entry for entry in TRANSLATORS_SPECS}


def load_translator(name: str, **kwargs) -> BaseTranslator:
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
