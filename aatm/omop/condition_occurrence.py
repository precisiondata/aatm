"""Pydantic models for OMOP CDM condition occurrence extraction and representation.

This module defines data models related to the OMOP Common Data Model
`condition_occurrence` table. It includes one model intended for extracting
condition-occurrence information from unstructured clinical documents and another
model that extends it with additional OMOP fields required for full table
representation.

The module also includes field-level validation for temporal consistency and
registers the extraction model in the OMOP extraction model registry.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from aatm.omop.registry import register_omop_extraction_model


@register_omop_extraction_model(register_id="condition_occurrence")
class ConditionOccurrenceExtractionModel(BaseModel):
    """Structured extraction model for OMOP CDM `condition_occurrence` information.

    This model is intended for information extraction from clinical documents such
    as clinical notes, encounter summaries, reports, and diagnosis lists. It
    contains only the subset of `condition_occurrence` fields that may reasonably
    be inferred from unstructured text.

    Attributes:
        condition_source_value: Verbatim condition code or text from the source.
        condition_source_concept_id: Source concept identifier for the original
            condition code or value.
        condition_start_date: Start date of the condition.
        condition_start_datetime: Start date and time of the condition.
        condition_end_date: End date of the condition.
        condition_end_datetime: End date and time of the condition.
        condition_status_source_value: Verbatim source value describing the
            condition status, such as admitting, principal, or secondary diagnosis.
        stop_reason: Reason the condition is no longer valid in the source data.
    """

    model_config = ConfigDict(extra="forbid")

    condition_source_value: Annotated[
        Optional[str], StringConstraints(max_length=50)
    ] = Field(
        None,
        description=(
            "Verbatim value from the source data representing the condition, "
            "such as a diagnosis code, sign, symptom, or source text."
        ),
    )
    condition_source_concept_id: Optional[int] = Field(
        None,
        description="Source concept identifier representing the original condition code/value.",
    )
    condition_start_date: Optional[date] = Field(
        None,
        description="Start date of the condition. Must follow the format YYYY-MM-DD.",
    )
    condition_start_datetime: Optional[datetime] = Field(
        None,
        description="Start datetime of the condition.",
    )
    condition_end_date: Optional[date] = Field(
        None,
        description="End date of the condition. Must follow the format YYYY-MM-DD.",
    )
    condition_end_datetime: Optional[datetime] = Field(
        None,
        description="End datetime of the condition.",
    )
    condition_status_source_value: Annotated[
        Optional[str], StringConstraints(max_length=50)
    ] = Field(
        None,
        description=(
            "Verbatim source value representing condition status, such as "
            "admitting diagnosis, principal diagnosis, secondary diagnosis, "
            "preliminary diagnosis, or exclusionary diagnosis."
        ),
    )

    stop_reason: Annotated[Optional[str], StringConstraints(max_length=20)] = Field(
        None,
        description=(
            "Reason the condition is no longer valid with respect to the source data. "
            "This does not necessarily mean the condition is no longer occurring."
        ),
    )

    @field_validator("condition_end_date")
    @classmethod
    def validate_end_date(cls, v: Optional[date], info) -> Optional[date]:
        """Validate that the end date is not earlier than the start date.

        Args:
            v: End date value provided for the condition occurrence.
            info: Pydantic validation context containing previously validated fields.

        Returns:
            The validated end date.

        Raises:
            ValueError: If `condition_end_date` is earlier than
                `condition_start_date`.
        """
        start_date = info.data.get("condition_start_date")
        if start_date is not None and v is not None and v < start_date:
            raise ValueError(
                "condition_end_date cannot be earlier than condition_start_date"
            )
        return v

    @field_validator("condition_end_datetime")
    @classmethod
    def validate_end_datetime(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Validate that the end datetime is not earlier than the start datetime.

        Args:
            v: End datetime value provided for the condition occurrence.
            info: Pydantic validation context containing previously validated fields.

        Returns:
            The validated end datetime.

        Raises:
            ValueError: If `condition_end_datetime` is earlier than
                `condition_start_datetime`.
        """
        start_dt = info.data.get("condition_start_datetime")
        if v is not None and start_dt is not None and v < start_dt:
            raise ValueError(
                "condition_end_datetime cannot be earlier than condition_start_datetime"
            )
        return v


@register_omop_extraction_model(register_id="condition_occurrence_list")
class ListOfConditionOccurrenceExtractionModel(BaseModel):
    """List wrapper for multiple condition occurrence extraction records."""

    model_config = ConfigDict(extra="forbid")
    records: list[ConditionOccurrenceExtractionModel]


class ConditionOccurrence(ConditionOccurrenceExtractionModel):
    """Complete OMOP CDM model for the `condition_occurrence` table.

    This model extends `ConditionOccurrenceExtractionModel` by adding OMOP-specific
    fields that are typically required in the structured table but are not usually
    expected to be extracted directly from clinical free text.

    Attributes:
        condition_occurrence_id: Unique identifier for the condition record.
        condition_status_concept_id: Standard concept identifier representing the
            status of the condition during the visit.
        person_id: Identifier of the person for whom the condition is recorded.
        condition_concept_id: Standard concept identifier for the condition.
        condition_type_concept_id: Concept identifier representing the provenance
            or type of the condition record.
        provider_id: Identifier of the provider associated with the condition.
        visit_occurrence_id: Identifier of the visit during which the condition
            occurred or was recorded.
        visit_detail_id: Identifier of the visit detail during which the condition
            occurred or was recorded.
    """

    model_config = ConfigDict(extra="forbid")

    condition_occurrence_id: int = Field(
        ...,
        description="Unique identifier for the condition occurrence record.",
    )
    condition_status_concept_id: Optional[int] = Field(
        None,
        description="Standard concept identifier representing the condition status.",
    )
    person_id: int = Field(
        ...,
        description="Identifier of the person for whom the condition is recorded.",
    )
    condition_concept_id: int = Field(
        ...,
        description=(
            "Standard concept identifier mapped from the source value and "
            "representing a condition."
        ),
    )
    condition_start_date: date = Field(
        ...,
        description="Start date of the condition. Required in the OMOP CDM table.",
    )
    condition_type_concept_id: int = Field(
        ...,
        description=(
            "Concept identifier representing the provenance or type of the "
            "condition record, such as EHR, claim, registry, or another source."
        ),
    )
    provider_id: Optional[int] = Field(
        None,
        description=(
            "Identifier of the provider associated with the condition record, "
            "such as the provider who made the diagnosis or recorded the symptom."
        ),
    )
    visit_occurrence_id: Optional[int] = Field(
        None,
        description="Identifier of the visit during which the condition occurred or was recorded.",
    )
    visit_detail_id: Optional[int] = Field(
        None,
        description=(
            "Identifier of the visit detail during which the condition occurred "
            "or was recorded."
        ),
    )
