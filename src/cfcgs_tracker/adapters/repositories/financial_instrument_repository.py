from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.domain.models.financial_instrument import (
    FinancialInstrument,
)


class FinancialInstrumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self,
        financial_instrument: FinancialInstrument,
    ) -> None:
        self.session.add(financial_instrument)

    async def get_by_name(self, name: str) -> FinancialInstrument | None:
        statement = select(FinancialInstrument).where(
            FinancialInstrument.name == name
        )
        return await self.session.scalar(statement)
