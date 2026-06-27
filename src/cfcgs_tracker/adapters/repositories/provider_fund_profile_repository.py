from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.domain.models.provider_fund_profile import (
    ProviderFundProfile,
)


class ProviderFundProfileRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, profile: ProviderFundProfile) -> None:
        self.session.add(profile)

    async def get_by_funding_provider_id(
        self,
        funding_provider_id: int,
    ) -> ProviderFundProfile | None:
        statement = select(ProviderFundProfile).where(
            ProviderFundProfile.funding_provider_id == funding_provider_id
        )
        return await self.session.scalar(statement)
