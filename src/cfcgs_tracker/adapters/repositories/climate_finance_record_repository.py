from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import String, case, cast, distinct, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.domain.models.beneficiary_country import (
    BeneficiaryCountry,
)
from src.cfcgs_tracker.domain.models.climate_finance_record import (
    ClimateFinanceRecord,
)
from src.cfcgs_tracker.domain.models.funding_provider import FundingProvider
from src.cfcgs_tracker.domain.models.financial_instrument import (
    FinancialInstrument,
)
from src.cfcgs_tracker.domain.models.project import Project
from src.cfcgs_tracker.domain.models.sector import Sector
from src.cfcgs_tracker.domain.models.source import Source
from src.cfcgs_tracker.domain.models.sub_sector import SubSector


class ClimateFinanceRecordRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, record: ClimateFinanceRecord) -> None:
        self.session.add(record)

    async def get_by_id(self, record_id: int) -> ClimateFinanceRecord | None:
        statement = select(ClimateFinanceRecord).where(
            ClimateFinanceRecord.id == record_id
        )
        return await self.session.scalar(statement)

    async def has_any(self) -> bool:
        statement = select(ClimateFinanceRecord.id).limit(1)
        return await self.session.scalar(statement) is not None

    async def get_for_import(
        self,
        *,
        year: int,
        project_id: int | None,
        funding_provider_id: int | None,
        source_id: int,
        beneficiary_country_id: int | None,
        sector_id: int | None,
        sub_sector_id: int | None,
        financial_instrument_id: int | None,
    ) -> ClimateFinanceRecord | None:
        statement = select(ClimateFinanceRecord).where(
            ClimateFinanceRecord.year == year,
            ClimateFinanceRecord.project_id == project_id,
            ClimateFinanceRecord.funding_provider_id == funding_provider_id,
            ClimateFinanceRecord.source_id == source_id,
            ClimateFinanceRecord.beneficiary_country_id
            == beneficiary_country_id,
            ClimateFinanceRecord.sector_id == sector_id,
            ClimateFinanceRecord.sub_sector_id == sub_sector_id,
            ClimateFinanceRecord.financial_instrument_id
            == financial_instrument_id,
        )
        return await self.session.scalar(statement)

    async def list_years(self) -> list[int]:
        statement = (
            select(ClimateFinanceRecord.year)
            .distinct()
            .order_by(ClimateFinanceRecord.year.desc())
        )
        result = await self.session.scalars(statement)
        return list(result.all())

    async def list_admin_records(
        self,
        *,
        limit: int,
        offset: int,
        search: str | None = None,
        sort_by: str = "year",
        sort_order: str = "desc",
        filters: dict[str, list[str]] | None = None,
    ) -> dict:
        sort_column_map = {
            "id": ClimateFinanceRecord.id,
            "year": ClimateFinanceRecord.year,
            "project_title": Project.title,
            "beneficiary_country": BeneficiaryCountry.name,
            "funding_provider": FundingProvider.name,
            "source": Source.name,
            "sector": Sector.name,
            "sub_sector": SubSector.name,
            "financial_instrument": FinancialInstrument.name,
            "approved_amount_usd_millions": (
                ClimateFinanceRecord.approved_amount_usd_millions
            ),
            "climate_finance_amount_usd_millions": (
                ClimateFinanceRecord.climate_finance_amount_usd_millions
            ),
        }
        sort_column = sort_column_map.get(sort_by, ClimateFinanceRecord.year)
        sort_expression = (
            sort_column.asc()
            if sort_order.lower() == "asc"
            else sort_column.desc()
        )

        base_statement = (
            select(
                ClimateFinanceRecord.id,
                ClimateFinanceRecord.year,
                ClimateFinanceRecord.source_row_hash,
                Project.title.label("project_title"),
                BeneficiaryCountry.name.label("beneficiary_country"),
                FundingProvider.name.label("funding_provider"),
                Source.name.label("source"),
                Source.url.label("source_url"),
                FinancialInstrument.name.label("financial_instrument"),
                Sector.name.label("sector"),
                SubSector.name.label("sub_sector"),
                ClimateFinanceRecord.approved_amount_usd_millions,
                ClimateFinanceRecord.climate_finance_amount_usd_millions,
                ClimateFinanceRecord.adaptation_amount_usd_millions,
                ClimateFinanceRecord.mitigation_amount_usd_millions,
                ClimateFinanceRecord.both_objectives_amount_usd_millions,
            )
            .select_from(ClimateFinanceRecord)
            .outerjoin(Project, Project.id == ClimateFinanceRecord.project_id)
            .outerjoin(
                BeneficiaryCountry,
                BeneficiaryCountry.id
                == ClimateFinanceRecord.beneficiary_country_id,
            )
            .outerjoin(
                FundingProvider,
                FundingProvider.id == ClimateFinanceRecord.funding_provider_id,
            )
            .join(Source, Source.id == ClimateFinanceRecord.source_id)
            .outerjoin(
                FinancialInstrument,
                FinancialInstrument.id
                == ClimateFinanceRecord.financial_instrument_id,
            )
            .outerjoin(Sector, Sector.id == ClimateFinanceRecord.sector_id)
            .outerjoin(
                SubSector,
                SubSector.id == ClimateFinanceRecord.sub_sector_id,
            )
        )

        if search:
            search_term = f"%{search.strip()}%"
            base_statement = base_statement.where(
                or_(
                    cast(ClimateFinanceRecord.year, String).ilike(search_term),
                    Project.title.ilike(search_term),
                    BeneficiaryCountry.name.ilike(search_term),
                    FundingProvider.name.ilike(search_term),
                    Source.name.ilike(search_term),
                    Source.url.ilike(search_term),
                    Sector.name.ilike(search_term),
                    SubSector.name.ilike(search_term),
                    FinancialInstrument.name.ilike(search_term),
                )
            )

        base_statement = self._apply_admin_filters(
            base_statement,
            filters=filters,
        )

        count_statement = select(func.count()).select_from(
            base_statement.order_by(None).subquery()
        )
        total = int((await self.session.scalar(count_statement)) or 0)

        statement = (
            base_statement.order_by(sort_expression, ClimateFinanceRecord.id)
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(statement)
        records = [dict(row._mapping) for row in result.all()]

        return {
            "records": records,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(records) < total,
        }

    async def list_admin_filter_options(
        self,
        *,
        search: str | None = None,
    ) -> dict[str, list[str | int]]:
        years_statement = select(
            distinct(ClimateFinanceRecord.year)
        ).select_from(ClimateFinanceRecord)
        options = {
            "year": (
                await self._fetch_distinct_values(
                    years_statement,
                    search=search,
                    search_columns=(cast(ClimateFinanceRecord.year, String),),
                    value_column=ClimateFinanceRecord.year,
                    order_by=ClimateFinanceRecord.year.desc(),
                )
            )[0],
            "project_title": (
                await self._fetch_distinct_values(
                    select(distinct(Project.title))
                    .select_from(ClimateFinanceRecord)
                    .outerjoin(
                        Project, Project.id == ClimateFinanceRecord.project_id
                    ),
                    search=search,
                    search_columns=(Project.title,),
                    value_column=Project.title,
                    order_by=Project.title.asc(),
                )
            )[0],
            "beneficiary_country": (
                await self._fetch_distinct_values(
                    select(distinct(BeneficiaryCountry.name))
                    .select_from(ClimateFinanceRecord)
                    .outerjoin(
                        BeneficiaryCountry,
                        BeneficiaryCountry.id
                        == ClimateFinanceRecord.beneficiary_country_id,
                    ),
                    search=search,
                    search_columns=(BeneficiaryCountry.name,),
                    value_column=BeneficiaryCountry.name,
                    order_by=BeneficiaryCountry.name.asc(),
                )
            )[0],
            "funding_provider": (
                await self._fetch_distinct_values(
                    select(distinct(FundingProvider.name))
                    .select_from(ClimateFinanceRecord)
                    .outerjoin(
                        FundingProvider,
                        FundingProvider.id
                        == ClimateFinanceRecord.funding_provider_id,
                    ),
                    search=search,
                    search_columns=(FundingProvider.name,),
                    value_column=FundingProvider.name,
                    order_by=FundingProvider.name.asc(),
                )
            )[0],
            "source": (
                await self._fetch_distinct_values(
                    select(distinct(Source.name))
                    .select_from(ClimateFinanceRecord)
                    .join(Source, Source.id == ClimateFinanceRecord.source_id),
                    search=search,
                    search_columns=(Source.name, Source.url),
                    value_column=Source.name,
                    order_by=Source.name.asc(),
                )
            )[0],
            "source_url": (
                await self._fetch_distinct_values(
                    select(distinct(Source.url))
                    .select_from(ClimateFinanceRecord)
                    .join(Source, Source.id == ClimateFinanceRecord.source_id),
                    search=search,
                    search_columns=(Source.name, Source.url),
                    value_column=Source.url,
                    order_by=Source.url.asc(),
                )
            )[0],
            "financial_instrument": (
                await self._fetch_distinct_values(
                    select(distinct(FinancialInstrument.name))
                    .select_from(ClimateFinanceRecord)
                    .outerjoin(
                        FinancialInstrument,
                        FinancialInstrument.id
                        == ClimateFinanceRecord.financial_instrument_id,
                    ),
                    search=search,
                    search_columns=(FinancialInstrument.name,),
                    value_column=FinancialInstrument.name,
                    order_by=FinancialInstrument.name.asc(),
                )
            )[0],
            "sector": (
                await self._fetch_distinct_values(
                    select(distinct(Sector.name))
                    .select_from(ClimateFinanceRecord)
                    .outerjoin(
                        Sector, Sector.id == ClimateFinanceRecord.sector_id
                    ),
                    search=search,
                    search_columns=(Sector.name,),
                    value_column=Sector.name,
                    order_by=Sector.name.asc(),
                )
            )[0],
            "sub_sector": (
                await self._fetch_distinct_values(
                    select(distinct(SubSector.name))
                    .select_from(ClimateFinanceRecord)
                    .outerjoin(
                        SubSector,
                        SubSector.id == ClimateFinanceRecord.sub_sector_id,
                    ),
                    search=search,
                    search_columns=(SubSector.name,),
                    value_column=SubSector.name,
                    order_by=SubSector.name.asc(),
                )
            )[0],
        }
        return options

    async def list_admin_filter_suggestions(
        self,
        *,
        column: str,
        search: str | None = None,
        limit: int = 20,
        offset: int = 0,
        filters: dict[str, list[str]] | None = None,
    ) -> dict[str, list[str | int] | int | bool]:
        if column == "year":
            statement = self._admin_distinct_statement(
                distinct(ClimateFinanceRecord.year)
            )
            values, has_more = await self._fetch_distinct_values(
                statement,
                search=search,
                search_columns=(cast(ClimateFinanceRecord.year, String),),
                value_column=ClimateFinanceRecord.year,
                order_by=ClimateFinanceRecord.year.desc(),
                limit=limit,
                offset=offset,
                filters=self._filters_without_column(filters, column),
            )
            return {
                "values": values,
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
            }

        statement_map = {
            "project_title": (
                self._admin_distinct_statement(distinct(Project.title)),
                (Project.title,),
                Project.title,
                Project.title.asc(),
            ),
            "beneficiary_country": (
                self._admin_distinct_statement(
                    distinct(BeneficiaryCountry.name)
                ),
                (BeneficiaryCountry.name,),
                BeneficiaryCountry.name,
                BeneficiaryCountry.name.asc(),
            ),
            "funding_provider": (
                self._admin_distinct_statement(distinct(FundingProvider.name)),
                (FundingProvider.name,),
                FundingProvider.name,
                FundingProvider.name.asc(),
            ),
            "source": (
                self._admin_distinct_statement(distinct(Source.name)),
                (Source.name, Source.url),
                Source.name,
                Source.name.asc(),
            ),
            "source_url": (
                self._admin_distinct_statement(distinct(Source.url)),
                (Source.name, Source.url),
                Source.url,
                Source.url.asc(),
            ),
            "financial_instrument": (
                self._admin_distinct_statement(
                    distinct(FinancialInstrument.name)
                ),
                (FinancialInstrument.name,),
                FinancialInstrument.name,
                FinancialInstrument.name.asc(),
            ),
            "sector": (
                self._admin_distinct_statement(distinct(Sector.name)),
                (Sector.name,),
                Sector.name,
                Sector.name.asc(),
            ),
            "sub_sector": (
                self._admin_distinct_statement(distinct(SubSector.name)),
                (SubSector.name,),
                SubSector.name,
                SubSector.name.asc(),
            ),
        }

        statement_data = statement_map.get(column)
        if not statement_data:
            raise ValueError("Invalid admin filter column")

        statement, search_columns, value_column, order_by = statement_data
        values, has_more = await self._fetch_distinct_values(
            statement,
            search=search,
            search_columns=search_columns,
            value_column=value_column,
            order_by=order_by,
            limit=limit,
            offset=offset,
            filters=self._filters_without_column(filters, column),
        )
        return {
            "values": values,
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
        }

    async def get_admin_record_view_by_id(
        self,
        record_id: int,
    ) -> dict | None:
        statement = (
            select(
                ClimateFinanceRecord.id,
                ClimateFinanceRecord.year,
                ClimateFinanceRecord.source_row_hash,
                Project.title.label("project_title"),
                BeneficiaryCountry.name.label("beneficiary_country"),
                FundingProvider.name.label("funding_provider"),
                Source.name.label("source"),
                Source.url.label("source_url"),
                FinancialInstrument.name.label("financial_instrument"),
                Sector.name.label("sector"),
                SubSector.name.label("sub_sector"),
                ClimateFinanceRecord.approved_amount_usd_millions,
                ClimateFinanceRecord.climate_finance_amount_usd_millions,
                ClimateFinanceRecord.adaptation_amount_usd_millions,
                ClimateFinanceRecord.mitigation_amount_usd_millions,
                ClimateFinanceRecord.both_objectives_amount_usd_millions,
            )
            .select_from(ClimateFinanceRecord)
            .outerjoin(Project, Project.id == ClimateFinanceRecord.project_id)
            .outerjoin(
                BeneficiaryCountry,
                BeneficiaryCountry.id
                == ClimateFinanceRecord.beneficiary_country_id,
            )
            .outerjoin(
                FundingProvider,
                FundingProvider.id == ClimateFinanceRecord.funding_provider_id,
            )
            .join(Source, Source.id == ClimateFinanceRecord.source_id)
            .outerjoin(
                FinancialInstrument,
                FinancialInstrument.id
                == ClimateFinanceRecord.financial_instrument_id,
            )
            .outerjoin(Sector, Sector.id == ClimateFinanceRecord.sector_id)
            .outerjoin(
                SubSector,
                SubSector.id == ClimateFinanceRecord.sub_sector_id,
            )
            .where(ClimateFinanceRecord.id == record_id)
        )
        row = (await self.session.execute(statement)).mappings().first()
        return dict(row) if row else None

    def _apply_admin_filters(
        self,
        statement,
        *,
        filters: dict[str, list[str]] | None,
    ):
        if not filters:
            return statement

        if years := filters.get("year"):
            statement = statement.where(
                ClimateFinanceRecord.year.in_([int(year) for year in years])
            )
        if project_titles := filters.get("project_title"):
            statement = statement.where(Project.title.in_(project_titles))
        if countries := filters.get("beneficiary_country"):
            statement = statement.where(BeneficiaryCountry.name.in_(countries))
        if providers := filters.get("funding_provider"):
            statement = statement.where(FundingProvider.name.in_(providers))
        if sources := filters.get("source"):
            statement = statement.where(Source.name.in_(sources))
        if source_urls := filters.get("source_url"):
            statement = statement.where(Source.url.in_(source_urls))
        if instruments := filters.get("financial_instrument"):
            statement = statement.where(
                FinancialInstrument.name.in_(instruments)
            )
        if sectors := filters.get("sector"):
            statement = statement.where(Sector.name.in_(sectors))
        if sub_sectors := filters.get("sub_sector"):
            statement = statement.where(SubSector.name.in_(sub_sectors))

        return statement

    def _filters_without_column(
        self,
        filters: dict[str, list[str]] | None,
        column: str,
    ) -> dict[str, list[str]] | None:
        if not filters:
            return None
        return {
            key: value
            for key, value in filters.items()
            if key != column and value
        }

    def _admin_distinct_statement(self, value_expression):
        return (
            select(value_expression)
            .select_from(ClimateFinanceRecord)
            .outerjoin(Project, Project.id == ClimateFinanceRecord.project_id)
            .outerjoin(
                BeneficiaryCountry,
                BeneficiaryCountry.id
                == ClimateFinanceRecord.beneficiary_country_id,
            )
            .outerjoin(
                FundingProvider,
                FundingProvider.id == ClimateFinanceRecord.funding_provider_id,
            )
            .join(Source, Source.id == ClimateFinanceRecord.source_id)
            .outerjoin(
                FinancialInstrument,
                FinancialInstrument.id
                == ClimateFinanceRecord.financial_instrument_id,
            )
            .outerjoin(Sector, Sector.id == ClimateFinanceRecord.sector_id)
            .outerjoin(
                SubSector,
                SubSector.id == ClimateFinanceRecord.sub_sector_id,
            )
        )

    async def _fetch_distinct_values(
        self,
        statement,
        *,
        search: str | None,
        search_columns: Sequence,
        value_column,
        order_by,
        limit: int | None = None,
        offset: int = 0,
        filters: dict[str, list[str]] | None = None,
    ) -> tuple[list[str | int], bool]:
        statement = self._apply_admin_filters(statement, filters=filters)
        if search:
            search_term = f"%{search.strip()}%"
            statement = statement.where(
                or_(*[column.ilike(search_term) for column in search_columns])
            )

        statement = statement.where(value_column.is_not(None))
        statement = statement.order_by(order_by)
        if limit is not None:
            statement = statement.limit(limit + 1)
        if offset:
            statement = statement.offset(offset)
        result = await self.session.scalars(statement)
        values = [value for value in result.all() if value not in (None, "")]
        has_more = limit is not None and len(values) > limit
        if has_more:
            values = values[:limit]
        return values, has_more

    async def search(
        self,
        *,
        years: list[int] | None = None,
        country_ids: list[int] | None = None,
        limit: int = 40,
        offset: int = 0,
    ) -> list[dict]:
        amount = func.coalesce(
            ClimateFinanceRecord.climate_finance_amount_usd_millions,
            ClimateFinanceRecord.adaptation_amount_usd_millions,
            ClimateFinanceRecord.mitigation_amount_usd_millions,
            ClimateFinanceRecord.both_objectives_amount_usd_millions,
            0.0,
        )
        statement = (
            select(
                ClimateFinanceRecord.id,
                ClimateFinanceRecord.year,
                amount.label("amount_usd_millions"),
                FundingProvider.name.label("funding_provider"),
                Project.title.label("project"),
                BeneficiaryCountry.name.label("recipient_country"),
                Source.name.label("source"),
            )
            .select_from(ClimateFinanceRecord)
            .outerjoin(
                FundingProvider,
                FundingProvider.id == ClimateFinanceRecord.funding_provider_id,
            )
            .outerjoin(Project, Project.id == ClimateFinanceRecord.project_id)
            .outerjoin(
                BeneficiaryCountry,
                BeneficiaryCountry.id
                == ClimateFinanceRecord.beneficiary_country_id,
            )
            .join(Source, Source.id == ClimateFinanceRecord.source_id)
            .order_by(
                ClimateFinanceRecord.year.desc(), ClimateFinanceRecord.id
            )
            .limit(limit)
            .offset(offset)
        )
        statement = self._apply_common_filters(
            statement,
            years=years,
            country_ids=country_ids,
        )
        result = await self.session.execute(statement)
        return [dict(row._mapping) for row in result.all()]

    async def list_sources(
        self,
        *,
        years: list[int] | None = None,
        country_ids: list[int] | None = None,
        project_ids: list[int] | None = None,
        objective: str = "all",
    ) -> list[dict]:
        statement = (
            select(
                Source.name.label("label"),
                Source.url.label("url"),
            )
            .select_from(ClimateFinanceRecord)
            .join(Source, Source.id == ClimateFinanceRecord.source_id)
            .distinct()
            .order_by(Source.name)
        )
        statement = self._apply_common_filters(
            statement,
            years=years,
            country_ids=country_ids,
            project_ids=project_ids,
            objective=objective,
        )
        result = await self.session.execute(statement)
        return [dict(row._mapping) for row in result.all()]

    async def list_for_export_by_year(self, year: int) -> list[dict]:
        amount = func.coalesce(
            ClimateFinanceRecord.climate_finance_amount_usd_millions,
            ClimateFinanceRecord.adaptation_amount_usd_millions,
            ClimateFinanceRecord.mitigation_amount_usd_millions,
            ClimateFinanceRecord.both_objectives_amount_usd_millions,
            0.0,
        )
        statement = (
            select(
                ClimateFinanceRecord.id,
                ClimateFinanceRecord.year,
                FundingProvider.name.label("funding_provider"),
                Project.title.label("project"),
                BeneficiaryCountry.name.label("recipient_country"),
                Source.name.label("source"),
                amount.label("amount_usd_millions"),
            )
            .select_from(ClimateFinanceRecord)
            .outerjoin(
                FundingProvider,
                FundingProvider.id == ClimateFinanceRecord.funding_provider_id,
            )
            .outerjoin(Project, Project.id == ClimateFinanceRecord.project_id)
            .outerjoin(
                BeneficiaryCountry,
                BeneficiaryCountry.id
                == ClimateFinanceRecord.beneficiary_country_id,
            )
            .join(Source, Source.id == ClimateFinanceRecord.source_id)
            .where(ClimateFinanceRecord.year == year)
            .order_by(ClimateFinanceRecord.id)
        )
        result = await self.session.execute(statement)
        return [dict(row._mapping) for row in result.all()]

    async def summarize(
        self,
        *,
        years: list[int] | None = None,
        country_ids: list[int] | None = None,
        project_ids: list[int] | None = None,
        objective: str = "all",
    ) -> dict:
        project_key = self._project_count_key()
        statement = select(
            func.count(distinct(project_key)).label("total_projects"),
            func.count(
                distinct(ClimateFinanceRecord.beneficiary_country_id)
            ).label("total_funded_countries"),
            func.count(
                distinct(ClimateFinanceRecord.beneficiary_country_id)
            ).label("total_countries"),
            func.sum(self._total_amount_expr()).label("total_amount"),
            func.sum(self._adaptation_expr()).label("total_adaptation"),
            func.sum(self._mitigation_expr()).label("total_mitigation"),
            func.sum(self._overlap_expr()).label("total_overlap"),
        ).select_from(ClimateFinanceRecord)
        statement = self._apply_common_filters(
            statement,
            years=years,
            country_ids=country_ids,
            project_ids=project_ids,
            objective=objective,
        )
        row = (await self.session.execute(statement)).one()
        return {
            "total_projects": int(row.total_projects or 0),
            "total_funded_countries": int(row.total_funded_countries or 0),
            "total_countries": int(row.total_countries or 0),
            "total_amount": float(row.total_amount or 0),
            "total_adaptation": float(row.total_adaptation or 0),
            "total_mitigation": float(row.total_mitigation or 0),
            "total_overlap": float(row.total_overlap or 0),
        }

    async def totals_by_objective(
        self,
        *,
        years: list[int] | None = None,
        country_ids: list[int] | None = None,
    ) -> list[dict]:
        statement = (
            select(
                ClimateFinanceRecord.year.label("year"),
                func.sum(self._adaptation_expr()).label("total_adaptation"),
                func.sum(self._mitigation_expr()).label("total_mitigation"),
                func.sum(self._overlap_expr()).label("total_overlap"),
            )
            .select_from(ClimateFinanceRecord)
            .group_by(ClimateFinanceRecord.year)
            .order_by(ClimateFinanceRecord.year)
        )
        statement = self._apply_common_filters(
            statement,
            years=years,
            country_ids=country_ids,
        )
        result = await self.session.execute(statement)
        return [
            {
                "year": row.year,
                "total_adaptation": float(row.total_adaptation or 0),
                "total_mitigation": float(row.total_mitigation or 0),
                "total_overlap": float(row.total_overlap or 0),
            }
            for row in result.all()
        ]

    async def totals_by_year(
        self,
        *,
        years: list[int] | None = None,
        country_ids: list[int] | None = None,
    ) -> list[dict]:
        statement = (
            select(
                ClimateFinanceRecord.year.label("year"),
                BeneficiaryCountry.name.label("country_name"),
                func.sum(self._total_amount_expr()).label("total_amount"),
            )
            .select_from(ClimateFinanceRecord)
            .join(
                BeneficiaryCountry,
                BeneficiaryCountry.id
                == ClimateFinanceRecord.beneficiary_country_id,
            )
            .group_by(ClimateFinanceRecord.year, BeneficiaryCountry.name)
            .order_by(ClimateFinanceRecord.year, BeneficiaryCountry.name)
        )
        statement = self._apply_common_filters(
            statement,
            years=years,
            country_ids=country_ids,
        )
        result = await self.session.execute(statement)
        return [
            {
                "year": row.year,
                "country_name": row.country_name,
                "total_amount": float(row.total_amount or 0),
            }
            for row in result.all()
        ]

    async def filter_options(
        self,
        *,
        years: list[int] | None = None,
        country_ids: list[int] | None = None,
        project_ids: list[int] | None = None,
        objective: str = "all",
    ) -> dict:
        base_years = select(ClimateFinanceRecord.year).distinct()
        base_years = self._apply_common_filters(
            base_years,
            country_ids=country_ids,
            project_ids=project_ids,
            objective=objective,
        )
        years_result = await self.session.scalars(
            base_years.order_by(ClimateFinanceRecord.year.desc())
        )

        countries_statement = (
            select(
                BeneficiaryCountry.id,
                BeneficiaryCountry.name,
            )
            .select_from(ClimateFinanceRecord)
            .join(
                BeneficiaryCountry,
                BeneficiaryCountry.id
                == ClimateFinanceRecord.beneficiary_country_id,
            )
            .distinct()
            .order_by(BeneficiaryCountry.name)
        )
        countries_statement = self._apply_common_filters(
            countries_statement,
            years=years,
            project_ids=project_ids,
            objective=objective,
        )
        countries_result = await self.session.execute(countries_statement)

        projects_statement = (
            select(Project.id, Project.title.label("name"))
            .select_from(ClimateFinanceRecord)
            .join(Project, Project.id == ClimateFinanceRecord.project_id)
            .distinct()
            .order_by(Project.title)
            .limit(50)
        )
        projects_statement = self._apply_common_filters(
            projects_statement,
            years=years,
            country_ids=country_ids,
            objective=objective,
        )
        projects_result = await self.session.execute(projects_statement)

        objectives = ["all", "adaptation", "mitigation", "both"]
        return {
            "years": list(years_result.all()),
            "countries": [
                {"id": row.id, "name": row.name, "region": None}
                for row in countries_result.all()
            ],
            "projects": [dict(row._mapping) for row in projects_result.all()],
            "objectives": objectives,
        }

    async def country_year_grid(
        self,
        *,
        years: list[int] | None = None,
        country_ids: list[int] | None = None,
        project_ids: list[int] | None = None,
        objective: str = "all",
        view: str = "country_year",
        row_offset: int = 0,
        row_limit: int = 30,
        column_offset: int = 0,
        column_limit: int = 30,
    ) -> dict:
        country_name = BeneficiaryCountry.name.label("country_name")
        country_id = BeneficiaryCountry.id.label("country_id")
        project_count = func.count(distinct(self._project_count_key())).label(
            "project_count"
        )
        statement = (
            select(
                ClimateFinanceRecord.year.label("year"),
                country_id,
                country_name,
                func.sum(self._total_amount_expr()).label("total_amount"),
                func.sum(self._adaptation_expr()).label(
                    "adaptation_exclusive"
                ),
                func.sum(self._mitigation_expr()).label(
                    "mitigation_exclusive"
                ),
                func.sum(self._overlap_expr()).label("overlap"),
                project_count,
            )
            .select_from(ClimateFinanceRecord)
            .join(
                BeneficiaryCountry,
                BeneficiaryCountry.id
                == ClimateFinanceRecord.beneficiary_country_id,
            )
            .group_by(ClimateFinanceRecord.year, country_id, country_name)
            .order_by(country_name, ClimateFinanceRecord.year)
        )
        statement = self._apply_common_filters(
            statement,
            years=years,
            country_ids=country_ids,
            project_ids=project_ids,
            objective=objective,
        )
        rows = [
            {
                "year": row.year,
                "country_id": row.country_id,
                "country_name": row.country_name,
                "total_amount": float(row.total_amount or 0),
                "adaptation_exclusive": float(row.adaptation_exclusive or 0),
                "mitigation_exclusive": float(row.mitigation_exclusive or 0),
                "overlap": float(row.overlap or 0),
                "project_count": int(row.project_count or 0),
            }
            for row in (await self.session.execute(statement)).all()
        ]
        return self._build_grid_payload(
            rows,
            view=view,
            row_offset=row_offset,
            row_limit=row_limit,
            column_offset=column_offset,
            column_limit=column_limit,
        )

    async def projects_by_country_and_year(
        self,
        *,
        year: int,
        country_id: int,
        project_ids: list[int] | None = None,
        objective: str = "all",
        limit: int = 30,
        offset: int = 0,
    ) -> dict:
        statement = (
            select(
                Project.id.label("id"),
                Project.title.label("name"),
                func.sum(self._total_amount_expr()).label("total_amount"),
                func.sum(self._adaptation_expr()).label(
                    "adaptation_exclusive"
                ),
                func.sum(self._mitigation_expr()).label(
                    "mitigation_exclusive"
                ),
                func.sum(self._overlap_expr()).label("overlap"),
            )
            .select_from(ClimateFinanceRecord)
            .join(Project, Project.id == ClimateFinanceRecord.project_id)
            .where(
                ClimateFinanceRecord.year == year,
                ClimateFinanceRecord.beneficiary_country_id == country_id,
            )
            .group_by(Project.id, Project.title)
            .order_by(Project.title)
        )
        statement = self._apply_common_filters(
            statement,
            project_ids=project_ids,
            objective=objective,
        )
        rows = (await self.session.execute(statement)).all()
        total = len(rows)
        paginated_rows = rows[offset : offset + limit]
        projects = []
        for row in paginated_rows:
            adaptation = float(row.adaptation_exclusive or 0)
            mitigation = float(row.mitigation_exclusive or 0)
            overlap = float(row.overlap or 0)
            projects.append(
                {
                    "id": int(row.id),
                    "name": row.name,
                    "objective": self._classify_objective(
                        adaptation,
                        mitigation,
                        overlap,
                    ),
                    "total_amount": float(row.total_amount or 0),
                    "adaptation_exclusive": adaptation,
                    "mitigation_exclusive": mitigation,
                    "overlap": overlap,
                }
            )
        return {
            "total": total,
            "has_more": offset + len(projects) < total,
            "projects": projects,
        }

    def _apply_common_filters(
        self,
        statement,
        *,
        years: list[int] | None = None,
        country_ids: list[int] | None = None,
        project_ids: list[int] | None = None,
        objective: str = "all",
    ):
        if years:
            statement = statement.where(ClimateFinanceRecord.year.in_(years))
        if country_ids:
            statement = statement.where(
                ClimateFinanceRecord.beneficiary_country_id.in_(country_ids)
            )
        if project_ids:
            statement = statement.where(
                ClimateFinanceRecord.project_id.in_(project_ids)
            )
        if objective == "adaptation":
            statement = statement.where(
                self._adaptation_expr() > 0,
                self._mitigation_expr() == 0,
                self._overlap_expr() == 0,
            )
        elif objective == "mitigation":
            statement = statement.where(
                self._mitigation_expr() > 0,
                self._adaptation_expr() == 0,
                self._overlap_expr() == 0,
            )
        elif objective == "both":
            statement = statement.where(
                (self._overlap_expr() > 0)
                | (
                    (self._adaptation_expr() > 0)
                    & (self._mitigation_expr() > 0)
                )
            )
        return statement

    def _total_amount_expr(self):
        return func.coalesce(
            ClimateFinanceRecord.climate_finance_amount_usd_millions,
            self._adaptation_expr()
            + self._mitigation_expr()
            + self._overlap_expr(),
            0.0,
        )

    def _adaptation_expr(self):
        return func.coalesce(
            ClimateFinanceRecord.adaptation_amount_usd_millions,
            0.0,
        )

    def _mitigation_expr(self):
        return func.coalesce(
            ClimateFinanceRecord.mitigation_amount_usd_millions,
            0.0,
        )

    def _overlap_expr(self):
        return func.coalesce(
            ClimateFinanceRecord.both_objectives_amount_usd_millions,
            0.0,
        )

    def _project_count_key(self):
        return case(
            (
                ClimateFinanceRecord.project_id.is_not(None),
                cast(ClimateFinanceRecord.project_id, String),
            ),
            else_=ClimateFinanceRecord.source_row_hash,
        )

    def _classify_objective(
        self,
        adaptation_exclusive: float,
        mitigation_exclusive: float,
        overlap: float,
    ) -> str:
        if overlap > 0 or (
            adaptation_exclusive > 0 and mitigation_exclusive > 0
        ):
            return "both"
        if adaptation_exclusive > 0:
            return "adaptation"
        if mitigation_exclusive > 0:
            return "mitigation"
        return "unknown"

    def _build_grid_payload(
        self,
        rows: Sequence[dict],
        *,
        view: str,
        row_offset: int,
        row_limit: int,
        column_offset: int,
        column_limit: int,
    ) -> dict:
        grand_total = sum(row["total_amount"] for row in rows)
        grand_total_projects = sum(row["project_count"] for row in rows)

        if not rows:
            return {
                "view": view,
                "rows": [],
                "columns": [],
                "row_totals": [],
                "column_totals": [],
                "cells": [],
                "grand_total": 0.0,
                "grand_total_projects": 0,
                "row_count": 0,
                "column_count": 0,
                "row_offset": row_offset,
                "column_offset": column_offset,
                "row_limit": row_limit,
                "column_limit": column_limit,
                "sources": [],
            }

        countries = sorted(
            {(row["country_id"], row["country_name"]) for row in rows},
            key=lambda item: item[1],
        )
        years = sorted({row["year"] for row in rows})

        totals_by_country: list[dict] = []
        for country_value, country_name in countries:
            country_rows = [
                row for row in rows if row["country_id"] == country_value
            ]
            total_amount = sum(item["total_amount"] for item in country_rows)
            totals_by_country.append(
                {
                    "country_id": country_value,
                    "country_name": country_name,
                    "total_amount": total_amount,
                    "project_count": sum(
                        item["project_count"] for item in country_rows
                    ),
                }
            )

        totals_by_year: list[dict] = []
        for year in years:
            year_rows = [row for row in rows if row["year"] == year]
            totals_by_year.append(
                {
                    "year": year,
                    "total_amount": sum(
                        item["total_amount"] for item in year_rows
                    ),
                    "project_count": sum(
                        item["project_count"] for item in year_rows
                    ),
                }
            )

        row_count = len(countries) if view == "country_year" else len(years)
        column_count = len(years) if view == "country_year" else len(countries)

        row_offset = max(row_offset, 0)
        column_offset = max(column_offset, 0)
        if row_offset >= row_count:
            row_offset = max(row_count - row_limit, 0)
        if column_offset >= column_count:
            column_offset = max(column_count - column_limit, 0)

        if view == "country_year":
            row_slice = totals_by_country[row_offset : row_offset + row_limit]
            column_slice = totals_by_year[
                column_offset : column_offset + column_limit
            ]
            row_keys = {row["country_id"] for row in row_slice}
            column_keys = {row["year"] for row in column_slice}
            row_labels = [row["country_name"] for row in row_slice]
            column_labels = [str(row["year"]) for row in column_slice]
        else:
            row_slice = totals_by_year[row_offset : row_offset + row_limit]
            column_slice = totals_by_country[
                column_offset : column_offset + column_limit
            ]
            row_keys = {row["year"] for row in row_slice}
            column_keys = {row["country_id"] for row in column_slice}
            row_labels = [str(row["year"]) for row in row_slice]
            column_labels = [row["country_name"] for row in column_slice]

        visible_rows = []
        for row in rows:
            if view == "country_year":
                if (
                    row["country_id"] not in row_keys
                    or row["year"] not in column_keys
                ):
                    continue
                row_label = row["country_name"]
                column_label = str(row["year"])
                row_total = next(
                    item["total_amount"]
                    for item in row_slice
                    if item["country_id"] == row["country_id"]
                )
                column_total = next(
                    item["total_amount"]
                    for item in column_slice
                    if item["year"] == row["year"]
                )
            else:
                if (
                    row["year"] not in row_keys
                    or row["country_id"] not in column_keys
                ):
                    continue
                row_label = str(row["year"])
                column_label = row["country_name"]
                row_total = next(
                    item["total_amount"]
                    for item in row_slice
                    if item["year"] == row["year"]
                )
                column_total = next(
                    item["total_amount"]
                    for item in column_slice
                    if item["country_id"] == row["country_id"]
                )
            visible_rows.append(
                {
                    "year": row["year"],
                    "country_id": row["country_id"],
                    "country_name": row["country_name"],
                    "row_label": row_label,
                    "column_label": column_label,
                    "total_amount": row["total_amount"],
                    "adaptation_exclusive": row["adaptation_exclusive"],
                    "mitigation_exclusive": row["mitigation_exclusive"],
                    "overlap": row["overlap"],
                    "project_count": row["project_count"],
                    "percent_of_total": (
                        row["total_amount"] / grand_total * 100
                        if grand_total
                        else 0.0
                    ),
                    "percent_of_row": (
                        row["total_amount"] / row_total * 100
                        if row_total
                        else 0.0
                    ),
                    "percent_of_column": (
                        row["total_amount"] / column_total * 100
                        if column_total
                        else 0.0
                    ),
                }
            )

        row_totals_payload = []
        if view == "country_year":
            for row in row_slice:
                row_totals_payload.append(
                    {
                        "label": row["country_name"],
                        "total_amount": row["total_amount"],
                        "project_count": row["project_count"],
                        "percent_of_total": (
                            row["total_amount"] / grand_total * 100
                            if grand_total
                            else 0.0
                        ),
                        "country_id": row["country_id"],
                        "year": None,
                    }
                )
        else:
            for row in row_slice:
                row_totals_payload.append(
                    {
                        "label": str(row["year"]),
                        "total_amount": row["total_amount"],
                        "project_count": row["project_count"],
                        "percent_of_total": (
                            row["total_amount"] / grand_total * 100
                            if grand_total
                            else 0.0
                        ),
                        "country_id": None,
                        "year": row["year"],
                    }
                )

        column_totals_payload = []
        if view == "country_year":
            for row in column_slice:
                column_totals_payload.append(
                    {
                        "label": str(row["year"]),
                        "total_amount": row["total_amount"],
                        "project_count": row["project_count"],
                        "percent_of_total": (
                            row["total_amount"] / grand_total * 100
                            if grand_total
                            else 0.0
                        ),
                        "country_id": None,
                        "year": row["year"],
                    }
                )
        else:
            for row in column_slice:
                column_totals_payload.append(
                    {
                        "label": row["country_name"],
                        "total_amount": row["total_amount"],
                        "project_count": row["project_count"],
                        "percent_of_total": (
                            row["total_amount"] / grand_total * 100
                            if grand_total
                            else 0.0
                        ),
                        "country_id": row["country_id"],
                        "year": None,
                    }
                )

        return {
            "view": view,
            "rows": row_labels,
            "columns": column_labels,
            "row_totals": row_totals_payload,
            "column_totals": column_totals_payload,
            "cells": visible_rows,
            "grand_total": grand_total,
            "grand_total_projects": grand_total_projects,
            "row_count": row_count,
            "column_count": column_count,
            "row_offset": row_offset,
            "column_offset": column_offset,
            "row_limit": row_limit,
            "column_limit": column_limit,
            "sources": [],
        }
