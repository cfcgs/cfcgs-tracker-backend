from typing import Any

from pydantic import field_validator

from src.cfcgs_tracker.service_layer.import_schemas.common import (
    ImportRowSchema,
    normalize_float,
    normalize_int,
    normalize_text,
)


class FundRowSchema(ImportRowSchema):
    fund: str
    fund_type: str | None = None
    fund_focus: str | None = None
    pledge_usd_mn: float | None = None
    deposit_usd_mn: float | None = None
    approval_usd_mn: float | None = None
    disbursement_usd_mn: float | None = None
    number_of_projects_approved: int | None = None

    @field_validator(
        "fund",
        "fund_type",
        "fund_focus",
        mode="before",
    )
    @classmethod
    def normalize_strings(cls, value: Any) -> str | None:
        return normalize_text(value)

    @field_validator("number_of_projects_approved", mode="before")
    @classmethod
    def normalize_count(cls, value: Any) -> int | None:
        return normalize_int(value)

    @field_validator(
        "pledge_usd_mn",
        "deposit_usd_mn",
        "approval_usd_mn",
        "disbursement_usd_mn",
        mode="before",
    )
    @classmethod
    def normalize_amounts(cls, value: Any) -> float | None:
        return normalize_float(value)
