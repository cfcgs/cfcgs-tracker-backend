from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.domain.models.import_job import ImportJob


class ImportJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, import_job: ImportJob) -> None:
        self.session.add(import_job)

    async def get_by_id(self, import_job_id: int) -> ImportJob | None:
        statement = select(ImportJob).where(ImportJob.id == import_job_id)
        return await self.session.scalar(statement)

    async def get_by_id_and_user_id(
        self,
        import_job_id: int,
        user_id: int,
    ) -> ImportJob | None:
        statement = select(ImportJob).where(
            ImportJob.id == import_job_id,
            ImportJob.user_id == user_id,
        )
        return await self.session.scalar(statement)

    async def list_paginated(
        self,
        limit: int,
        offset: int,
    ) -> list[ImportJob]:
        statement = (
            select(ImportJob)
            .order_by(ImportJob.id.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.scalars(statement)
        return list(result.all())

    async def list_paginated_by_user_id(
        self,
        user_id: int,
        limit: int,
        offset: int,
    ) -> list[ImportJob]:
        statement = (
            select(ImportJob)
            .where(ImportJob.user_id == user_id)
            .order_by(ImportJob.id.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.scalars(statement)
        return list(result.all())
