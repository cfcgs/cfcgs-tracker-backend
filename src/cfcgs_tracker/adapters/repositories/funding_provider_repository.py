from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.domain.models.fund_focus import FundFocus
from src.cfcgs_tracker.domain.models.fund_type import FundType
from src.cfcgs_tracker.domain.models.funding_provider import FundingProvider
from src.cfcgs_tracker.domain.models.provider_fund_profile import (
    ProviderFundProfile,
)


class FundingProviderRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, provider: FundingProvider) -> None:
        self.session.add(provider)

    async def get_by_id(self, provider_id: int) -> FundingProvider | None:
        statement = select(FundingProvider).where(
            FundingProvider.id == provider_id
        )
        return await self.session.scalar(statement)

    async def get_by_name(self, name: str) -> FundingProvider | None:
        statement = select(FundingProvider).where(FundingProvider.name == name)
        return await self.session.scalar(statement)

    async def list_funding_providers(
        self,
        *,
        fund_type_ids: list[int] | None = None,
        fund_focus_ids: list[int] | None = None,
        limit: int = 40,
        offset: int = 0,
    ):
        statement = (
            select(
                FundingProvider.id.label("id"),
                FundingProvider.name.label("funding_provider_name"),
                FundType.name.label("fund_type"),
                FundFocus.name.label("fund_focus"),
                ProviderFundProfile.pledge.label("pledge"),
                ProviderFundProfile.deposit.label("deposit"),
                ProviderFundProfile.approval.label("approval"),
                ProviderFundProfile.disbursement.label("disbursement"),
                ProviderFundProfile.projects_approved.label(
                    "projects_approved"
                ),
            )
            .join(
                ProviderFundProfile,
                ProviderFundProfile.funding_provider_id == FundingProvider.id,
            )
            .outerjoin(
                FundType,
                FundType.id == ProviderFundProfile.fund_type_id,
            )
            .outerjoin(
                FundFocus,
                FundFocus.id == ProviderFundProfile.fund_focus_id,
            )
            .order_by(FundingProvider.name.asc())
            .limit(limit)
            .offset(offset)
        )

        if fund_type_ids:
            statement = statement.where(
                ProviderFundProfile.fund_type_id.in_(fund_type_ids)
            )
        if fund_focus_ids:
            statement = statement.where(
                ProviderFundProfile.fund_focus_id.in_(fund_focus_ids)
            )

        result = await self.session.execute(statement)
        return list(result.mappings().all())

    async def summarize_funding_providers(
        self,
        *,
        funding_provider_ids: list[int] | None = None,
        fund_type_ids: list[int] | None = None,
        fund_focus_ids: list[int] | None = None,
    ):
        statement = select(
            func.coalesce(func.sum(ProviderFundProfile.pledge), 0).label(
                "total_pledge"
            ),
            func.coalesce(func.sum(ProviderFundProfile.deposit), 0).label(
                "total_deposit"
            ),
            func.coalesce(func.sum(ProviderFundProfile.approval), 0).label(
                "total_approval"
            ),
        ).select_from(ProviderFundProfile)

        if funding_provider_ids:
            statement = statement.where(
                ProviderFundProfile.funding_provider_id.in_(
                    funding_provider_ids
                )
            )
        if fund_type_ids:
            statement = statement.where(
                ProviderFundProfile.fund_type_id.in_(fund_type_ids)
            )
        if fund_focus_ids:
            statement = statement.where(
                ProviderFundProfile.fund_focus_id.in_(fund_focus_ids)
            )

        result = await self.session.execute(statement)
        return result.mappings().one()

    async def get_funding_provider_row_by_id(
        self,
        funding_provider_id: int,
    ):
        statement = (
            select(
                FundingProvider.id.label("id"),
                FundingProvider.name.label("funding_provider_name"),
                FundType.name.label("fund_type"),
                FundFocus.name.label("fund_focus"),
                ProviderFundProfile.pledge.label("pledge"),
                ProviderFundProfile.deposit.label("deposit"),
                ProviderFundProfile.approval.label("approval"),
                ProviderFundProfile.disbursement.label("disbursement"),
                ProviderFundProfile.projects_approved.label(
                    "projects_approved"
                ),
            )
            .join(
                ProviderFundProfile,
                ProviderFundProfile.funding_provider_id == FundingProvider.id,
            )
            .outerjoin(
                FundType,
                FundType.id == ProviderFundProfile.fund_type_id,
            )
            .outerjoin(
                FundFocus,
                FundFocus.id == ProviderFundProfile.fund_focus_id,
            )
            .where(FundingProvider.id == funding_provider_id)
        )
        row = (await self.session.execute(statement)).mappings().first()
        return dict(row) if row else None
