from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.domain.models.beneficiary_country import (
    BeneficiaryCountry,
)


class BeneficiaryCountryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, country: BeneficiaryCountry) -> None:
        self.session.add(country)

    async def get_by_name(self, name: str) -> BeneficiaryCountry | None:
        statement = select(BeneficiaryCountry).where(
            BeneficiaryCountry.name == name
        )
        return await self.session.scalar(statement)

    async def list_all(self) -> list[BeneficiaryCountry]:
        statement = select(BeneficiaryCountry).order_by(
            BeneficiaryCountry.name.asc()
        )
        result = await self.session.scalars(statement)
        return list(result.all())
