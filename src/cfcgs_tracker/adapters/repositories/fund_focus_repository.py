from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.domain.models.fund_focus import FundFocus


class FundFocusRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, fund_focus: FundFocus) -> None:
        self.session.add(fund_focus)

    async def get_by_name(self, name: str) -> FundFocus | None:
        statement = select(FundFocus).where(FundFocus.name == name)
        return await self.session.scalar(statement)

    async def list_all(self) -> list[FundFocus]:
        statement = select(FundFocus).order_by(FundFocus.name.asc())
        result = await self.session.scalars(statement)
        return list(result.all())
