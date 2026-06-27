from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.domain.models.sector import Sector


class SectorRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, sector: Sector) -> None:
        self.session.add(sector)

    async def get_by_name(self, name: str) -> Sector | None:
        statement = select(Sector).where(Sector.name == name)
        return await self.session.scalar(statement)
