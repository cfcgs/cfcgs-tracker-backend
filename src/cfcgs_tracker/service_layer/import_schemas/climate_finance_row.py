from typing import Any

from pydantic import field_validator

from src.cfcgs_tracker.service_layer.import_schemas.common import (
    ImportRowSchema,
    normalize_float,
    normalize_int,
    normalize_text,
)


class ClimateFinanceRowSchema(ImportRowSchema):
    year: int
    project_title: str | None = None
    beneficiary_country: str | None = None
    provider_type: str | None = None
    funding_provider: str | None = None
    approved_amount_usd_millions: float | None = None
    climate_finance_amount_usd_millions: float | None = None
    adaptation_climate_finance_amount_usd_millions: float | None = None
    mitigation_climate_finance_amount_usd_millions: float | None = None
    both_climate_objectives_finance_amount_usd_millions: float | None = None
    sector: str | None = None
    sub_sector: str | None = None
    financial_instrument: str | None = None
    source: str
    source_url: str
    hash_updated: str | None = None

    @field_validator(
        "project_title",
        "beneficiary_country",
        "provider_type",
        "funding_provider",
        "sector",
        "sub_sector",
        "financial_instrument",
        "source",
        "source_url",
        "hash_updated",
        mode="before",
    )
    @classmethod
    def normalize_strings(cls, value: Any) -> str | None:
        return normalize_text(value)

    @field_validator("year", mode="before")
    @classmethod
    def normalize_year(cls, value: Any) -> int:
        normalized = normalize_int(value)
        if normalized is None:
            raise ValueError("year is required")
        return normalized

    @field_validator(
        "approved_amount_usd_millions",
        "climate_finance_amount_usd_millions",
        "adaptation_climate_finance_amount_usd_millions",
        "mitigation_climate_finance_amount_usd_millions",
        "both_climate_objectives_finance_amount_usd_millions",
        mode="before",
    )
    @classmethod
    def normalize_amounts(cls, value: Any) -> float | None:
        return normalize_float(value)
