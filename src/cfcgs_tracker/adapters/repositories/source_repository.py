from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.domain.models.source import Source


class SourceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, source: Source) -> None:
        self.session.add(source)

    async def get_by_name(self, name: str) -> Source | None:
        statement = select(Source).where(Source.name == name)
        return await self.session.scalar(statement)

    async def get_by_url(self, url: str) -> Source | None:
        statement = select(Source).where(Source.url == url)
        return await self.session.scalar(statement)
