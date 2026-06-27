from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.domain.models.fund_type import FundType


class FundTypeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, fund_type: FundType) -> None:
        self.session.add(fund_type)

    async def get_by_name(self, name: str) -> FundType | None:
        statement = select(FundType).where(FundType.name == name)
        return await self.session.scalar(statement)

    async def list_all(self) -> list[FundType]:
        statement = select(FundType).order_by(FundType.name.asc())
        result = await self.session.scalars(statement)
        return list(result.all())
