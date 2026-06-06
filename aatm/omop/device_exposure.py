"""Pydantic models for OMOP CDM device exposure extraction and representation."""

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


@register_omop_extraction_model(register_id="device_exposure")
class DeviceExposureExtractionModel(BaseModel):
    """Structured extraction model for OMOP CDM `device_exposure` information."""

    model_config = ConfigDict(extra="forbid")

    device_source_value: Annotated[Optional[str], StringConstraints(max_length=50)] = (
        Field(
            None,
            description="Verbatim value from the source data representing the device exposure.",
        )
    )
    device_source_concept_id: Optional[int] = Field(
        None,
        description="Source concept identifier representing the original device source value.",
    )
    device_exposure_start_date: Optional[date] = Field(
        None,
        description="Start date of the device exposure. Must follow the format YYYY-MM-DD.",
    )
    device_exposure_start_datetime: Optional[datetime] = Field(
        None,
        description="Start datetime of the device exposure.",
    )
    device_exposure_end_date: Optional[date] = Field(
        None,
        description="End date or discontinuation date of the device exposure, if available.",
    )
    device_exposure_end_datetime: Optional[datetime] = Field(
        None,
        description="End datetime of the device exposure, if available.",
    )
    unique_device_id: Annotated[Optional[str], StringConstraints(max_length=255)] = (
        Field(
            None,
            description="Unique Device Identification number for regulated devices, if available.",
        )
    )
    production_id: Annotated[Optional[str], StringConstraints(max_length=255)] = Field(
        None,
        description="Production Identifier portion of the Unique Device Identification.",
    )
    quantity: Optional[int] = Field(
        None,
        description="Quantity of device exposure. If exposure exists but quantity is not specified, this may be set to 1.",
    )
    unit_source_value: Annotated[Optional[str], StringConstraints(max_length=50)] = (
        Field(
            None,
            description="Verbatim source value representing the unit of the device quantity.",
        )
    )
    unit_source_concept_id: Optional[int] = Field(
        None,
        description="Source concept identifier representing the original unit source value.",
    )

    @field_validator("device_exposure_end_date")
    @classmethod
    def validate_end_date(cls, v: Optional[date], info) -> Optional[date]:
        start_date = info.data.get("device_exposure_start_date")
        if start_date is not None and v is not None and v < start_date:
            raise ValueError(
                "device_exposure_end_date cannot be earlier than device_exposure_start_date"
            )
        return v

    @field_validator("device_exposure_end_datetime")
    @classmethod
    def validate_end_datetime(cls, v: Optional[datetime], info) -> Optional[datetime]:
        start_dt = info.data.get("device_exposure_start_datetime")
        if start_dt is not None and v is not None and v < start_dt:
            raise ValueError(
                "device_exposure_end_datetime cannot be earlier than device_exposure_start_datetime"
            )
        return v

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("quantity cannot be negative")
        return v


@register_omop_extraction_model(register_id="device_exposure_list")
class ListOfDeviceExposureExtractionModel(BaseModel):
    """List wrapper for extracted OMOP CDM `device_exposure` records."""

    model_config = ConfigDict(extra="forbid")

    records: list[DeviceExposureExtractionModel]


class DeviceExposure(DeviceExposureExtractionModel):
    """Complete OMOP CDM model for the `device_exposure` table."""

    model_config = ConfigDict(extra="forbid")

    device_exposure_id: int = Field(
        ...,
        description="Unique identifier for the device exposure record.",
    )
    person_id: int = Field(
        ...,
        description="Identifier of the person associated with the device exposure.",
    )
    device_concept_id: int = Field(
        ...,
        description="Standard Device concept identifier mapped from the source value.",
    )
    device_type_concept_id: int = Field(
        ...,
        description="Concept identifier representing the provenance or type of the device record.",
    )
    provider_id: Optional[int] = Field(
        None,
        description="Identifier of the provider associated with the device record.",
    )
    visit_occurrence_id: Optional[int] = Field(
        None,
        description="Identifier of the visit during which the device was prescribed or given.",
    )
    visit_detail_id: Optional[int] = Field(
        None,
        description="Identifier of the visit detail during which the device was prescribed or given.",
    )
    unit_concept_id: Optional[int] = Field(
        None,
        description="Standard Unit concept identifier representing the unit associated with the device quantity.",
    )
