from collections.abc import Callable
from typing import Optional
from pydantic import BaseModel

OMOP_EXTRACTION_MODEL_REGISTRY: dict[str, Callable] = {}


def register_omop_extraction_model(
    obj: type[BaseModel] = None,
    *,
    register_id: Optional[str] = None,
):
    def decorator(o) -> type[BaseModel]:
        key = register_id or o.__name__
        if key in OMOP_EXTRACTION_MODEL_REGISTRY:
            raise ValueError(f"Model '{key}' is already registered.")
        OMOP_EXTRACTION_MODEL_REGISTRY[key] = o
        return o

    if obj is None:
        return decorator

    return decorator(obj)
