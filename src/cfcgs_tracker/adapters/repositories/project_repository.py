from sqlalchemy import distinct, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.domain.models.climate_finance_record import (
    ClimateFinanceRecord,
)
from src.cfcgs_tracker.domain.models.project import Project


class ProjectRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, project: Project) -> None:
        self.session.add(project)

    async def get_by_title(self, title: str) -> Project | None:
        statement = select(Project).where(Project.title == title)
        return await self.session.scalar(statement)

    async def list_paginated_for_records(
        self,
        *,
        search: str | None = None,
        years: list[int] | None = None,
        country_ids: list[int] | None = None,
        objective: str = "all",
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        base_query = (
            select(Project.id, Project.title.label("name"))
            .join(
                ClimateFinanceRecord,
                Project.id == ClimateFinanceRecord.project_id,
            )
            .distinct()
        )

        if years:
            base_query = base_query.where(ClimateFinanceRecord.year.in_(years))
        if country_ids:
            base_query = base_query.where(
                ClimateFinanceRecord.beneficiary_country_id.in_(country_ids)
            )
        if objective == "adaptation":
            base_query = base_query.where(
                ClimateFinanceRecord.adaptation_amount_usd_millions > 0,
                (
                    ClimateFinanceRecord.mitigation_amount_usd_millions.is_(
                        None
                    )
                    | (
                        ClimateFinanceRecord.mitigation_amount_usd_millions
                        == 0
                    )
                ),
                (
                    ClimateFinanceRecord.both_objectives_amount_usd_millions.is_(
                        None
                    )
                    | (
                        ClimateFinanceRecord.both_objectives_amount_usd_millions
                        == 0
                    )
                ),
            )
        elif objective == "mitigation":
            base_query = base_query.where(
                ClimateFinanceRecord.mitigation_amount_usd_millions > 0,
                (
                    ClimateFinanceRecord.adaptation_amount_usd_millions.is_(
                        None
                    )
                    | (
                        ClimateFinanceRecord.adaptation_amount_usd_millions
                        == 0
                    )
                ),
                (
                    ClimateFinanceRecord.both_objectives_amount_usd_millions.is_(
                        None
                    )
                    | (
                        ClimateFinanceRecord.both_objectives_amount_usd_millions
                        == 0
                    )
                ),
            )
        elif objective == "both":
            base_query = base_query.where(
                ClimateFinanceRecord.both_objectives_amount_usd_millions > 0
            )

        if search:
            base_query = base_query.where(Project.title.ilike(f"%{search}%"))

        count_query = select(distinct(Project.id)).join(
            ClimateFinanceRecord,
            Project.id == ClimateFinanceRecord.project_id,
        )
        if years:
            count_query = count_query.where(
                ClimateFinanceRecord.year.in_(years)
            )
        if country_ids:
            count_query = count_query.where(
                ClimateFinanceRecord.beneficiary_country_id.in_(country_ids)
            )
        if objective == "adaptation":
            count_query = count_query.where(
                ClimateFinanceRecord.adaptation_amount_usd_millions > 0,
                (
                    ClimateFinanceRecord.mitigation_amount_usd_millions.is_(
                        None
                    )
                    | (
                        ClimateFinanceRecord.mitigation_amount_usd_millions
                        == 0
                    )
                ),
                (
                    ClimateFinanceRecord.both_objectives_amount_usd_millions.is_(
                        None
                    )
                    | (
                        ClimateFinanceRecord.both_objectives_amount_usd_millions
                        == 0
                    )
                ),
            )
        elif objective == "mitigation":
            count_query = count_query.where(
                ClimateFinanceRecord.mitigation_amount_usd_millions > 0,
                (
                    ClimateFinanceRecord.adaptation_amount_usd_millions.is_(
                        None
                    )
                    | (
                        ClimateFinanceRecord.adaptation_amount_usd_millions
                        == 0
                    )
                ),
                (
                    ClimateFinanceRecord.both_objectives_amount_usd_millions.is_(
                        None
                    )
                    | (
                        ClimateFinanceRecord.both_objectives_amount_usd_millions
                        == 0
                    )
                ),
            )
        elif objective == "both":
            count_query = count_query.where(
                ClimateFinanceRecord.both_objectives_amount_usd_millions > 0
            )
        if search:
            count_query = count_query.where(Project.title.ilike(f"%{search}%"))
        total = len((await self.session.scalars(count_query)).all())

        paginated_query = (
            base_query.order_by(Project.title).limit(limit).offset(offset)
        )
        projects = [
            dict(row._mapping)
            for row in (await self.session.execute(paginated_query)).all()
        ]
        return {
            "projects": projects,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(projects) < total,
        }
