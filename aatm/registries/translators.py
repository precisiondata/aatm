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
        translator_kwargs = copy.deepcopy(TRANSLATORS_REGISTRY[name].kwargs)
        translator_kwargs.update(kwargs)  # override with user provided kwargs
        return TRANSLATORS_REGISTRY[name].translator_class(**translator_kwargs)
    except KeyError:
        raise KeyError(
            f"Translator '{name}' not found. Available translators: {list(TRANSLATORS_REGISTRY.keys())}"
        )
    except Exception as e:
        raise e(f"Unexpected error while loading translator '{name}': {e}")
