"""Pydantic models for OMOP CDM drug exposure extraction and representation.

This module defines data models related to the OMOP Common Data Model
`drug_exposure` table. It includes one model intended for extracting
drug-exposure information from unstructured clinical documents and another
model that extends it with additional OMOP fields required for full table
representation.

The module also includes field-level validation for temporal consistency and
numeric constraints, and registers the extraction model in the OMOP extraction
model registry.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Optional

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    StringConstraints,
    field_validator,
)

from aatm.omop.registry import register_omop_extraction_model


@register_omop_extraction_model(register_id="drug_exposure")
class DrugExposureExtractionModel(BaseModel):
    """Structured extraction model for OMOP CDM `drug_exposure` information.

    This model is intended for information extraction from clinical documents such
    as patient notes and prescriptions. It contains only the subset of
    `drug_exposure` fields that may reasonably be inferred from unstructured text.

    Attributes:
        drug_exposure_start_date: Start date of the drug exposure.
        drug_exposure_start_datetime: Start date and time of the drug exposure.
        drug_exposure_end_date: End date of the drug exposure.
        drug_exposure_end_datetime: End date and time of the drug exposure.
        verbatim_end_date: End date as represented in the original source.
        stop_reason: Reason the medication was stopped.
        refills: Intended number of prescription refills.
        quantity: Quantity prescribed, dispensed, or administered.
        days_supply: Number of days of supply recorded in the source.
        sig: Verbatim medication instructions.
        lot_number: Lot number of the drug product.
        route_source_value: Verbatim administration route from the source.
        dose_unit_source_value: Verbatim dose unit from the source.
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
    stop_reason: Annotated[Optional[str], StringConstraints(max_length=20)] = Field(
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
    days_supply: Annotated[Optional[int], Field(strict=True, gt=0)] = Field(
        None,
        description="Number of days of supply recorded in the source.",
    )
    sig: Optional[str] = Field(
        None,
        description="Verbatim instruction for the drug as written by the provider.",
    )
    lot_number: Annotated[Optional[str], StringConstraints(max_length=50)] = Field(
        None,
        description="Lot number of the drug product.",
    )
    route_source_value: Annotated[Optional[str], StringConstraints(max_length=50)] = (
        Field(
            None,
            description="Verbatim route value from the source data.",
        )
    )
    dose_unit_source_value: Annotated[
        Optional[str], StringConstraints(max_length=50)
    ] = Field(
        None,
        description="Verbatim dose unit value from the source data.",
    )

    @field_validator("drug_exposure_end_date")
    @classmethod
    def validate_end_date(cls, v: date, info) -> date:
        """Validate that the end date is not earlier than the start date.

        Args:
            v: End date value provided for the drug exposure.
            info: Pydantic validation context containing previously validated fields.

        Returns:
            The validated end date.

        Raises:
            ValueError: If `drug_exposure_end_date` is earlier than
                `drug_exposure_start_date`.
        """
        start_date = info.data.get("drug_exposure_start_date")
        if start_date is not None and v < start_date:
            raise ValueError(
                "drug_exposure_end_date cannot be earlier than drug_exposure_start_date"
            )
        return v

    @field_validator("drug_exposure_end_datetime")
    @classmethod
    def validate_end_datetime(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Validate that the end datetime is not earlier than the start datetime.

        Args:
            v: End datetime value provided for the drug exposure.
            info: Pydantic validation context containing previously validated fields.

        Returns:
            The validated end datetime.

        Raises:
            ValueError: If `drug_exposure_end_datetime` is earlier than
                `drug_exposure_start_datetime`.
        """
        start_dt = info.data.get("drug_exposure_start_datetime")
        if v is not None and start_dt is not None and v < start_dt:
            raise ValueError(
                "drug_exposure_end_datetime cannot be earlier than drug_exposure_start_datetime"
            )
        return v

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: Optional[float]) -> Optional[float]:
        """Validate that the quantity is not negative.

        Args:
            v: Quantity value provided for the drug exposure.

        Returns:
            The validated quantity.

        Raises:
            ValueError: If `quantity` is negative.
        """
        if v is not None and v < 0:
            raise ValueError("quantity cannot be negative")
        return v


class DrugExposure(DrugExposureExtractionModel):
    """Complete OMOP CDM model for the `drug_exposure` table.

    This model extends `DrugExposureExtractionModel` by adding OMOP-specific
    fields that are typically required in the structured table but are not usually
    expected to be extracted directly from clinical free text.

    Attributes:
        drug_exposure_id: Unique identifier for the drug exposure record.
        person_id: Identifier of the person associated with the record.
        drug_concept_id: Standard concept identifier for the drug.
        drug_type_concept_id: Concept identifier representing the provenance or
            type of the drug record.
        provider_id: Identifier of the associated provider.
        visit_occurrence_id: Identifier of the associated visit occurrence.
        visit_detail_id: Identifier of the associated visit detail.
        drug_source_value: Verbatim drug code or value from the source.
        drug_source_concept_id: Source concept identifier for the original drug
            code or value.
        route_concept_id: Standard concept identifier for the route of
            administration.
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
    drug_source_value: Annotated[Optional[str], StringConstraints(max_length=50)] = (
        Field(
            None,
            description="Verbatim drug code or value from the source data.",
        )
    )
    drug_source_concept_id: Optional[int] = Field(
        None,
        description="Source concept identifier representing the original drug code/value.",
    )
    route_concept_id: Optional[int] = Field(
        None,
        description="Standard concept identifier for the route of administration.",
    )
