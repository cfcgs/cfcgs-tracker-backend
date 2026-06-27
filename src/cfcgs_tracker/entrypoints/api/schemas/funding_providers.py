from pydantic import BaseModel

from src.cfcgs_tracker.entrypoints.api.schemas.records import (
    SourceReferencePublic,
)


class FundingProviderReadFilter(BaseModel):
    fund_types: list[int] | None = None
    fund_focuses: list[int] | None = None


class FundingProviderSummaryFilter(BaseModel):
    funding_providers: list[int] | None = None
    fund_types: list[int] | None = None
    fund_focuses: list[int] | None = None


class FundingProviderPublic(BaseModel):
    id: int
    funding_provider_name: str
    fund_type: str | None = None
    fund_focus: str | None = None
    pledge: float | None = None
    deposit: float | None = None
    approval: float | None = None
    disbursement: float | None = None
    projects_approved: int | None = None


class FundingProviderProfileUpdate(BaseModel):
    fund_type: str | None = None
    fund_focus: str | None = None
    pledge: float | None = None
    deposit: float | None = None
    approval: float | None = None
    disbursement: float | None = None
    projects_approved: int | None = None


class FundingProviderList(BaseModel):
    funding_providers: list[FundingProviderPublic]
    sources: list[SourceReferencePublic]


class FundingProviderSummaryPublic(BaseModel):
    total_pledge: float
    total_deposit: float
    total_approval: float
    sources: list[SourceReferencePublic]
