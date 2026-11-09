from aatm.omop.registry import register_omop_extraction_model  # noqa: F401
import aatm.omop.drug_exposure  # noqa: F401
import aatm.omop.condition_occurrence  # noqa: F401
import aatm.omop.device_exposure  # noqa: F401
from aatm.omop.mapping import (
    ClinicalNote,
    ConceptRef,
    FieldSource,
    MapSpec,
    OmopMapping,
)
from aatm.omop.tables import (
    OMOP_TABLES,
    ConditionOccurrence,
    DrugExposure,
    Measurement,
    Observation,
    OmopTableRow,
)

# The builder depends on aatm.data_models.extractors, which itself imports
# aatm.omop.mapping, so it is imported lazily to avoid a circular import.
_LAZY = {
    "OmopTableBuilder": "aatm.omop.builder",
    "OmopExtractionPipeline": "aatm.omop.builder",
    "OmopTables": "aatm.omop.builder",
    "OmopIssue": "aatm.omop.builder",
    "SqliteConceptLookup": "aatm.omop.vocabulary",
    "Concept": "aatm.omop.vocabulary",
    "ConceptMapping": "aatm.omop.builder",
    "CodedText": "aatm.omop.coding",
    "TerminologyMapperCoder": "aatm.omop.coding",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'aatm.omop' has no attribute {name!r}")


__all__ = [
    "ClinicalNote",
    "ConceptRef",
    "FieldSource",
    "MapSpec",
    "OmopMapping",
    "OMOP_TABLES",
    "ConditionOccurrence",
    "DrugExposure",
    "Measurement",
    "Observation",
    "OmopTableRow",
    *_LAZY,
]
