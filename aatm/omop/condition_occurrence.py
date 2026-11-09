"""Pydantic extraction models for OMOP CDM condition occurrence information.

These models are the target data models of the older `ExtractionTask` interface, kept
in the OMOP extraction model registry as ``condition_occurrence`` and
``condition_occurrence_list``. They hold the subset of `condition_occurrence` fields
that may be read from unstructured text, with checks for temporal consistency.

Rows of the CDM table itself are `aatm.omop.tables.ConditionOccurrence`, built from
entity definitions by `aatm.omop.OmopTableBuilder`.
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
