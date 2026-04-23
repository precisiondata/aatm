from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    conint,
    constr,
)

from aatm.omop.registry import register_omop_extraction_model


@register_omop_extraction_model(register_id="drug_exposure")
class DrugExposureExtractionModel(BaseModel):
    """
    Pydantic model to be used for information extraction from clinical documents based on the OMOP CDM `drug_exposure` table.
    """

    model_config = ConfigDict(extra="forbid")

    drug_exposure_start_date: Optional[date] = Field(
        None,
        description="Start date of the drug exposure. Must follow the format YYYY-MM-DD.",
    )
    drug_exposure_start_datetime: Optional[datetime] = Field(
        None,
        description="Start datetime of the drug exposure.",
    )
    drug_exposure_end_date: Optional[date] = Field(
        None,
        description="End date of the drug exposure. Must follow the format YYYY-MM-DD.",
    )
    drug_exposure_end_datetime: Optional[datetime] = Field(
        None,
        description="End datetime of the drug exposure.",
    )
    verbatim_end_date: Optional[date] = Field(
        None,
        description="End date of the drug exposure as represented in the source data if available.",
    )
    stop_reason: Optional[constr(max_length=20)] = Field(
        None,
        description="Reason the medication was stopped, as represented in the source.",
    )
    refills: Optional[int] = Field(
        None,
        description="Number of intended refills for a prescription.",
    )
    quantity: Optional[float] = Field(
        None,
        description="Quantity of drug dispensed, prescribed, or administered.",
    )
    days_supply: Optional[conint(ge=0)] = Field(
        None,
        description="Number of days of supply recorded in the source.",
    )
    sig: Optional[str] = Field(
        None,
        description="Verbatim instruction for the drug as written by the provider.",
    )
    route_concept_id: Optional[int] = Field(
        None,
        description="Standard concept identifier for the route of administration.",
    )
    lot_number: Optional[constr(max_length=50)] = Field(
        None,
        description="Lot number of the drug product.",
    )

    route_source_value: Optional[constr(max_length=50)] = Field(
        None,
        description="Verbatim route value from the source data.",
    )
    dose_unit_source_value: Optional[constr(max_length=50)] = Field(
        None,
        description="Verbatim dose unit value from the source data.",
    )

    @field_validator("drug_exposure_end_date")
    @classmethod
    def validate_end_date(cls, v: date, info) -> date:
        start_date = info.data.get("drug_exposure_start_date")
        if start_date is not None and v < start_date:
            raise ValueError(
                "drug_exposure_end_date cannot be earlier than drug_exposure_start_date"
            )
        return v

    @field_validator("drug_exposure_end_datetime")
    @classmethod
    def validate_end_datetime(cls, v: Optional[datetime], info) -> Optional[datetime]:
        start_dt = info.data.get("drug_exposure_start_datetime")
        if v is not None and start_dt is not None and v < start_dt:
            raise ValueError(
                "drug_exposure_end_datetime cannot be earlier than drug_exposure_start_datetime"
            )
        return v

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("quantity cannot be negative")
        return v


class DrugExposure(DrugExposureExtractionModel):
    """
    Pydantic model for the OMOP CDM `drug_exposure` table.
    """

    model_config = ConfigDict(extra="forbid")

    drug_exposure_id: int = Field(
        ...,
        description="Unique identifier for the drug exposure record.",
    )
    person_id: int = Field(
        ...,
        description="Identifier of the person for whom the drug exposure is recorded.",
    )
    drug_concept_id: int = Field(
        ...,
        description="Standard drug concept identifier representing the drug product or ingredient.",
    )
    drug_type_concept_id: int = Field(
        ...,
        description="Concept identifier representing the provenance/type of the drug record.",
    )
    provider_id: Optional[int] = Field(
        None,
        description="Identifier of the provider associated with the drug record.",
    )
    visit_occurrence_id: Optional[int] = Field(
        None,
        description="Identifier of the visit during which the drug exposure occurred.",
    )
    visit_detail_id: Optional[int] = Field(
        None,
        description="Identifier of the visit detail during which the drug exposure occurred.",
    )
    drug_source_value: Optional[constr(max_length=50)] = Field(
        None,
        description="Verbatim drug code or value from the source data.",
    )
    drug_source_concept_id: Optional[int] = Field(
        None,
        description="Source concept identifier representing the original drug code/value.",
    )
