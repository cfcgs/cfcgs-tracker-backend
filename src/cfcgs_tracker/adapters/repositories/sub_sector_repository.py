from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.domain.models.sub_sector import SubSector


class SubSectorRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, sub_sector: SubSector) -> None:
        self.session.add(sub_sector)

    async def get_by_name_and_sector_id(
        self,
        *,
        name: str,
        sector_id: int,
    ) -> SubSector | None:
        statement = select(SubSector).where(
            SubSector.name == name,
            SubSector.sector_id == sector_id,
        )
        return await self.session.scalar(statement)
