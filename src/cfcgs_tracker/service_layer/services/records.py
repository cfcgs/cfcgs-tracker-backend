from src.cfcgs_tracker.domain.models import (
    BeneficiaryCountry,
    FinancialInstrument,
    FundingProvider,
    Project,
    Sector,
    Source,
    SubSector,
)
from src.cfcgs_tracker.service_layer.unit_of_work import AbstractUnitOfWork


class RecordService:
    def __init__(self, uow: AbstractUnitOfWork) -> None:
        self.uow = uow

    async def get_years(self) -> list[int]:
        return await self.uow.climate_finance_records.list_years()

    async def search_records(
        self,
        *,
        years: list[int] | None = None,
        country_ids: list[int] | None = None,
        limit: int = 40,
        offset: int = 0,
    ) -> list[dict]:
        return await self.uow.climate_finance_records.search(
            years=years,
            country_ids=country_ids,
            limit=limit,
            offset=offset,
        )

    async def export_records_by_year(self, year: int) -> list[dict]:
        return await self.uow.climate_finance_records.list_for_export_by_year(
            year
        )

    async def get_summary(
        self,
        *,
        years: list[int] | None = None,
        country_ids: list[int] | None = None,
        project_ids: list[int] | None = None,
        objective: str = "all",
    ) -> dict:
        summary = await self.uow.climate_finance_records.summarize(
            years=years,
            country_ids=country_ids,
            project_ids=project_ids,
            objective=objective,
        )
        summary[
            "sources"
        ] = await self.uow.climate_finance_records.list_sources(
            years=years,
            country_ids=country_ids,
            project_ids=project_ids,
            objective=objective,
        )
        return summary

    async def get_objective_aggregation(
        self,
        *,
        years: list[int] | None = None,
        country_ids: list[int] | None = None,
    ) -> list[dict]:
        totals = await self.uow.climate_finance_records.totals_by_objective(
            years=years,
            country_ids=country_ids,
        )
        sources = await self.uow.climate_finance_records.list_sources(
            years=years,
            country_ids=country_ids,
        )
        return {"totals": totals, "sources": sources}

    async def get_year_aggregation(
        self,
        *,
        years: list[int] | None = None,
        country_ids: list[int] | None = None,
    ) -> list[dict]:
        rows = await self.uow.climate_finance_records.totals_by_year(
            years=years,
            country_ids=country_ids,
        )
        series_map: dict[str, list[dict]] = {}
        totals_map: dict[int, float] = {}

        for row in rows:
            totals_map[row["year"]] = (
                totals_map.get(row["year"], 0) + row["total_amount"]
            )
            country_name = row["country_name"]
            series_map.setdefault(country_name, []).append(
                {
                    "year": row["year"],
                    "amount": row["total_amount"],
                }
            )

        if not country_ids:
            total_series = [
                {"year": year, "amount": amount}
                for year, amount in sorted(totals_map.items())
            ]
            return {
                "series": [
                    {
                        "name": "Financiamento Total Agregado",
                        "data": total_series,
                    }
                ],
                "sources": await self.uow.climate_finance_records.list_sources(
                    years=years,
                    country_ids=country_ids,
                ),
            }

        return {
            "series": [
                {
                    "name": country_name,
                    "data": sorted(data, key=lambda item: item["year"]),
                }
                for country_name, data in series_map.items()
            ],
            "sources": await self.uow.climate_finance_records.list_sources(
                years=years,
                country_ids=country_ids,
            ),
        }

    async def get_filter_options(
        self,
        *,
        years: list[int] | None = None,
        country_ids: list[int] | None = None,
        project_ids: list[int] | None = None,
        objective: str = "all",
    ) -> dict:
        return await self.uow.climate_finance_records.filter_options(
            years=years,
            country_ids=country_ids,
            project_ids=project_ids,
            objective=objective,
        )

    async def get_country_year_grid(
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
        payload = await self.uow.climate_finance_records.country_year_grid(
            years=years,
            country_ids=country_ids,
            project_ids=project_ids,
            objective=objective,
            view=view,
            row_offset=row_offset,
            row_limit=row_limit,
            column_offset=column_offset,
            column_limit=column_limit,
        )
        payload[
            "sources"
        ] = await self.uow.climate_finance_records.list_sources(
            years=years,
            country_ids=country_ids,
            project_ids=project_ids,
            objective=objective,
        )
        return payload

    async def get_projects_by_country_and_year(
        self,
        *,
        year: int,
        country_id: int,
        project_ids: list[int] | None = None,
        objective: str = "all",
        limit: int = 30,
        offset: int = 0,
    ) -> dict:
        return await self.uow.climate_finance_records.projects_by_country_and_year(
            year=year,
            country_id=country_id,
            project_ids=project_ids,
            objective=objective,
            limit=limit,
            offset=offset,
        )

    async def get_paginated_projects(
        self,
        *,
        search: str | None = None,
        years: list[int] | None = None,
        country_ids: list[int] | None = None,
        objective: str = "all",
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        return await self.uow.projects.list_paginated_for_records(
            search=search,
            years=years,
            country_ids=country_ids,
            objective=objective,
            limit=limit,
            offset=offset,
        )

    async def get_admin_records(
        self,
        *,
        limit: int,
        offset: int,
        search: str | None = None,
        sort_by: str = "year",
        sort_order: str = "desc",
        filters: dict[str, list[str]] | None = None,
    ) -> dict:
        return await self.uow.climate_finance_records.list_admin_records(
            limit=limit,
            offset=offset,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
            filters=filters,
        )

    async def get_admin_record_filter_options(
        self,
        *,
        search: str | None = None,
    ) -> dict[str, list[str] | list[int]]:
        return (
            await self.uow.climate_finance_records.list_admin_filter_options(
                search=search
            )
        )

    async def get_admin_record_filter_suggestions(
        self,
        *,
        column: str,
        search: str | None = None,
        limit: int = 20,
        offset: int = 0,
        filters: dict[str, list[str]] | None = None,
    ) -> dict[str, list[str] | list[int] | str | int | bool]:
        payload = await self.uow.climate_finance_records.list_admin_filter_suggestions(
            column=column,
            search=search,
            limit=limit,
            offset=offset,
            filters=filters,
        )
        return {"column": column, **payload}

    async def update_admin_record(
        self,
        *,
        record_id: int,
        payload: dict,
    ) -> dict:
        record = await self.uow.climate_finance_records.get_by_id(record_id)
        if not record:
            raise ValueError("Record not found")

        if "year" in payload and payload["year"] is not None:
            record.year = payload["year"]

        record.project_id = await self._resolve_project_id(
            payload.get("project_title")
        )
        record.beneficiary_country_id = await self._resolve_country_id(
            payload.get("beneficiary_country")
        )
        record.funding_provider_id = await self._resolve_provider_id(
            payload.get("funding_provider")
        )
        source = await self._resolve_source(
            payload.get("source"),
            payload.get("source_url"),
        )
        if source:
            record.source_id = source.id
        record.financial_instrument_id = (
            await self._resolve_financial_instrument_id(
                payload.get("financial_instrument")
            )
        )

        sector = await self._resolve_sector(payload.get("sector"))
        sub_sector = await self._resolve_sub_sector(
            sector=sector,
            sub_sector_name=payload.get("sub_sector"),
        )
        record.sector_id = sector.id if sector else None
        record.sub_sector_id = sub_sector.id if sub_sector else None

        amount_fields = (
            "approved_amount_usd_millions",
            "climate_finance_amount_usd_millions",
            "adaptation_amount_usd_millions",
            "mitigation_amount_usd_millions",
            "both_objectives_amount_usd_millions",
        )
        for field_name in amount_fields:
            if field_name in payload:
                setattr(record, field_name, payload[field_name])

        await self.uow.commit()
        updated_record = (
            await self.uow.climate_finance_records.get_admin_record_view_by_id(
                record.id
            )
        )
        if not updated_record:
            raise ValueError("Updated record could not be loaded")
        return updated_record

    async def _resolve_project_id(self, title: str | None) -> int | None:
        title = (title or "").strip()
        if not title:
            return None
        project = await self.uow.projects.get_by_title(title)
        if project:
            return project.id
        project = Project(title=title)
        await self.uow.projects.add(project)
        await self.uow.flush()
        return project.id

    async def _resolve_country_id(self, name: str | None) -> int | None:
        name = (name or "").strip()
        if not name:
            return None
        country = await self.uow.beneficiary_countries.get_by_name(name)
        if country:
            return country.id
        country = BeneficiaryCountry(name=name)
        await self.uow.beneficiary_countries.add(country)
        await self.uow.flush()
        return country.id

    async def _resolve_provider_id(self, name: str | None) -> int | None:
        name = (name or "").strip()
        if not name:
            return None
        provider = await self.uow.funding_providers.get_by_name(name)
        if provider:
            return provider.id
        provider = FundingProvider(name=name)
        await self.uow.funding_providers.add(provider)
        await self.uow.flush()
        return provider.id

    async def _resolve_source(
        self,
        name: str | None,
        url: str | None,
    ) -> Source | None:
        name = (name or "").strip()
        url = (url or "").strip()
        if not name and not url:
            return None
        if name:
            source = await self.uow.sources.get_by_name(name)
            if source:
                if url and source.url != url:
                    source.url = url
                return source
        if url:
            source = await self.uow.sources.get_by_url(url)
            if source:
                if name and source.name != name:
                    source.name = name
                return source
        if not name or not url:
            raise ValueError("Source and source_url are required together")
        source = Source(name=name, url=url)
        await self.uow.sources.add(source)
        await self.uow.flush()
        return source

    async def _resolve_financial_instrument_id(
        self,
        name: str | None,
    ) -> int | None:
        name = (name or "").strip()
        if not name:
            return None
        financial_instrument = (
            await self.uow.financial_instruments.get_by_name(name)
        )
        if financial_instrument:
            return financial_instrument.id
        financial_instrument = FinancialInstrument(name=name)
        await self.uow.financial_instruments.add(financial_instrument)
        await self.uow.flush()
        return financial_instrument.id

    async def _resolve_sector(self, name: str | None) -> Sector | None:
        name = (name or "").strip()
        if not name:
            return None
        sector = await self.uow.sectors.get_by_name(name)
        if sector:
            return sector
        sector = Sector(name=name)
        await self.uow.sectors.add(sector)
        await self.uow.flush()
        return sector

    async def _resolve_sub_sector(
        self,
        *,
        sector: Sector | None,
        sub_sector_name: str | None,
    ) -> SubSector | None:
        sub_sector_name = (sub_sector_name or "").strip()
        if not sub_sector_name or not sector:
            return None
        sub_sector = await self.uow.sub_sectors.get_by_name_and_sector_id(
            name=sub_sector_name,
            sector_id=sector.id,
        )
        if sub_sector:
            return sub_sector
        sub_sector = SubSector(name=sub_sector_name, sector_id=sector.id)
        await self.uow.sub_sectors.add(sub_sector)
        await self.uow.flush()
        return sub_sector
