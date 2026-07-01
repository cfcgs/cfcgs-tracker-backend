import pytest

from src.cfcgs_tracker.domain.models import (
    BeneficiaryCountry,
    FinancialInstrument,
    FundingProvider,
    Project,
    Sector,
    Source,
    SubSector,
)
from src.cfcgs_tracker.service_layer.services.records import RecordService
from src.cfcgs_tracker.service_layer.unit_of_work import AbstractUnitOfWork


class _RecordRepo:
    def __init__(self) -> None:
        self.summary = {"total_amount": 10.0}
        self.sources = [{"label": "OECD", "url": "https://www.oecd.org/"}]
        self.year_rows = []
        self.filter_payload = {
            "years": [2024],
            "countries": [{"id": 1, "name": "Brazil"}],
            "projects": [{"id": 2, "title": "Project A"}],
            "objectives": ["all", "adaptation"],
        }
        self.grid_payload = {"rows": [], "columns": []}
        self.projects_payload = {"projects": [], "total": 0}
        self.admin_records_payload = {"records": [], "total": 0}
        self.admin_filter_options_payload = {"year": [2024]}
        self.admin_filter_suggestions_payload = {
            "options": ["Brazil"],
            "total": 1,
            "has_more": False,
        }
        self.record_by_id = {}
        self.admin_view_by_id = {}

    async def list_years(self):
        return [2022, 2023, 2024]

    async def search(self, **kwargs):
        return [{"id": 1}]

    async def list_for_export_by_year(self, year):
        return [{"year": year}]

    async def summarize(self, **kwargs):
        return dict(self.summary)

    async def list_sources(self, **kwargs):
        return list(self.sources)

    async def totals_by_objective(self, **kwargs):
        return [{"year": 2024, "total_adaptation": 1.0}]

    async def totals_by_year(self, **kwargs):
        return list(self.year_rows)

    async def filter_options(self, **kwargs):
        return dict(self.filter_payload)

    async def country_year_grid(self, **kwargs):
        return dict(self.grid_payload)

    async def projects_by_country_and_year(self, **kwargs):
        return dict(self.projects_payload)

    async def list_admin_records(self, **kwargs):
        return dict(self.admin_records_payload)

    async def list_admin_filter_options(self, **kwargs):
        return dict(self.admin_filter_options_payload)

    async def list_admin_filter_suggestions(self, **kwargs):
        return dict(self.admin_filter_suggestions_payload)

    async def get_by_id(self, record_id: int):
        return self.record_by_id.get(record_id)

    async def get_admin_record_view_by_id(self, record_id: int):
        return self.admin_view_by_id.get(record_id)


class _ProjectRepo:
    def __init__(self) -> None:
        self.by_title = {}
        self.added = []
        self.paginated_payload = {"projects": [], "total": 0}

    async def list_paginated_for_records(self, **kwargs):
        return dict(self.paginated_payload)

    async def get_by_title(self, title: str):
        return self.by_title.get(title)

    async def add(self, instance):
        self.added.append(instance)
        self.by_title[instance.title] = instance


class _NamedRepo:
    def __init__(self) -> None:
        self.by_name = {}
        self.by_url = {}
        self.by_key = {}
        self.added = []

    async def get_by_name(self, name: str):
        return self.by_name.get(name)

    async def get_by_url(self, url: str):
        return self.by_url.get(url)

    async def get_by_name_and_sector_id(self, *, name: str, sector_id: int):
        return self.by_key.get((name, sector_id))

    async def add(self, instance):
        self.added.append(instance)
        if hasattr(instance, "name"):
            self.by_name[instance.name] = instance
        if hasattr(instance, "url"):
            self.by_url[instance.url] = instance
        if hasattr(instance, "sector_id"):
            self.by_key[(instance.name, instance.sector_id)] = instance


class _Record:
    def __init__(self) -> None:
        self.id = 1
        self.year = 2024
        self.project_id = None
        self.beneficiary_country_id = None
        self.funding_provider_id = None
        self.source_id = None
        self.financial_instrument_id = None
        self.sector_id = None
        self.sub_sector_id = None
        self.approved_amount_usd_millions = None
        self.climate_finance_amount_usd_millions = None
        self.adaptation_amount_usd_millions = None
        self.mitigation_amount_usd_millions = None
        self.both_objectives_amount_usd_millions = None


class FakeRecordUoW(AbstractUnitOfWork):
    def __init__(self) -> None:
        self.climate_finance_records = _RecordRepo()
        self.projects = _ProjectRepo()
        self.beneficiary_countries = _NamedRepo()
        self.funding_providers = _NamedRepo()
        self.sources = _NamedRepo()
        self.financial_instruments = _NamedRepo()
        self.sectors = _NamedRepo()
        self.sub_sectors = _NamedRepo()
        self.committed = False
        self._flush_counter = 100

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        return None

    async def refresh(self, instance) -> None:
        return None

    async def flush(self) -> None:
        self._flush_counter += 1
        for repository in (
            self.projects,
            self.beneficiary_countries,
            self.funding_providers,
            self.sources,
            self.financial_instruments,
            self.sectors,
            self.sub_sectors,
        ):
            for instance in repository.added:
                if getattr(instance, "id", None) is None:
                    instance.id = self._flush_counter

    def begin_nested(self):
        return None

    async def execute(self, statement, params=None):
        return None


@pytest.mark.asyncio
async def test_get_year_aggregation_returns_total_series_without_country_filter():
    uow = FakeRecordUoW()
    uow.climate_finance_records.year_rows = [
        {"year": 2023, "country_name": "Brazil", "total_amount": 10.0},
        {"year": 2023, "country_name": "Argentina", "total_amount": 5.0},
        {"year": 2024, "country_name": "Brazil", "total_amount": 7.5},
    ]
    service = RecordService(uow)

    payload = await service.get_year_aggregation()

    assert payload["series"] == [
        {
            "name": "Financiamento Total Agregado",
            "data": [
                {"year": 2023, "amount": 15.0},
                {"year": 2024, "amount": 7.5},
            ],
        }
    ]
    assert payload["sources"] == uow.climate_finance_records.sources


@pytest.mark.asyncio
async def test_get_year_aggregation_returns_country_series_when_filtered():
    uow = FakeRecordUoW()
    uow.climate_finance_records.year_rows = [
        {"year": 2022, "country_name": "Brazil", "total_amount": 4.0},
        {"year": 2023, "country_name": "Brazil", "total_amount": 6.0},
    ]
    service = RecordService(uow)

    payload = await service.get_year_aggregation(country_ids=[1])

    assert payload["series"] == [
        {
            "name": "Brazil",
            "data": [
                {"year": 2022, "amount": 4.0},
                {"year": 2023, "amount": 6.0},
            ],
        }
    ]


@pytest.mark.asyncio
async def test_record_service_overview_and_passthrough_methods():
    uow = FakeRecordUoW()
    uow.projects.paginated_payload = {
        "projects": [{"id": 2, "title": "Project A"}],
        "total": 1,
    }
    uow.climate_finance_records.projects_payload = {
        "projects": [{"project_title": "Project A"}],
        "total": 1,
    }
    service = RecordService(uow)

    overview = await service.get_overview()
    assert overview["summary"]["sources"] == uow.climate_finance_records.sources
    assert overview["years"] == [2024]
    assert overview["grid"]["sources"] == uow.climate_finance_records.sources

    assert await service.get_years() == [2022, 2023, 2024]
    assert await service.search_records() == [{"id": 1}]
    assert await service.export_records_by_year(2024) == [{"year": 2024}]
    assert await service.get_objective_aggregation() == {
        "totals": [{"year": 2024, "total_adaptation": 1.0}],
        "sources": uow.climate_finance_records.sources,
    }
    assert await service.get_filter_options() == uow.climate_finance_records.filter_payload
    assert await service.get_projects_by_country_and_year(
        year=2024,
        country_id=1,
    ) == {"projects": [{"project_title": "Project A"}], "total": 1}
    assert await service.get_paginated_projects() == {
        "projects": [{"id": 2, "title": "Project A"}],
        "total": 1,
    }
    assert await service.get_admin_records(limit=10, offset=0) == {
        "records": [],
        "total": 0,
    }
    assert await service.get_admin_record_filter_options() == {"year": [2024]}
    assert await service.get_admin_record_filter_suggestions(
        column="beneficiary_country",
    ) == {
        "column": "beneficiary_country",
        "options": ["Brazil"],
        "total": 1,
        "has_more": False,
    }


@pytest.mark.asyncio
async def test_update_admin_record_creates_missing_dimensions_and_amounts():
    uow = FakeRecordUoW()
    service = RecordService(uow)
    record = _Record()
    uow.climate_finance_records.record_by_id[1] = record
    uow.climate_finance_records.admin_view_by_id[1] = {
        "id": 1,
        "year": 2025,
        "project_title": "Project B",
    }

    payload = await service.update_admin_record(
        record_id=1,
        payload={
            "year": 2025,
            "project_title": "Project B",
            "beneficiary_country": "Chile",
            "funding_provider": "Provider B",
            "source": "OECD",
            "source_url": "https://www.oecd.org/",
            "financial_instrument": "Grant",
            "sector": "Energy",
            "sub_sector": "Transmission",
            "approved_amount_usd_millions": 10.0,
            "climate_finance_amount_usd_millions": 11.0,
            "adaptation_amount_usd_millions": 1.0,
            "mitigation_amount_usd_millions": 2.0,
            "both_objectives_amount_usd_millions": 3.0,
        },
    )

    assert uow.committed is True
    assert payload["year"] == 2025
    assert record.year == 2025
    assert isinstance(record.project_id, int)
    assert isinstance(record.beneficiary_country_id, int)
    assert isinstance(record.funding_provider_id, int)
    assert isinstance(record.source_id, int)
    assert isinstance(record.financial_instrument_id, int)
    assert isinstance(record.sector_id, int)
    assert isinstance(record.sub_sector_id, int)
    assert record.climate_finance_amount_usd_millions == 11.0


@pytest.mark.asyncio
async def test_update_admin_record_reuses_existing_dimensions_and_validates_source():
    uow = FakeRecordUoW()
    service = RecordService(uow)
    record = _Record()
    uow.climate_finance_records.record_by_id[1] = record
    uow.climate_finance_records.admin_view_by_id[1] = {"id": 1}

    project = Project(title="Existing Project")
    project.id = 1
    country = BeneficiaryCountry(name="Existing Country")
    country.id = 2
    provider = FundingProvider(name="Existing Provider")
    provider.id = 3
    source = Source(name="Existing Source", url="https://existing.example")
    source.id = 4
    instrument = FinancialInstrument(name="Loan")
    instrument.id = 5
    sector = Sector(name="Transport")
    sector.id = 6
    sub_sector = SubSector(name="Rail", sector_id=6)
    sub_sector.id = 7

    uow.projects.by_title[project.title] = project
    uow.beneficiary_countries.by_name[country.name] = country
    uow.funding_providers.by_name[provider.name] = provider
    uow.sources.by_name[source.name] = source
    uow.sources.by_url[source.url] = source
    uow.financial_instruments.by_name[instrument.name] = instrument
    uow.sectors.by_name[sector.name] = sector
    uow.sub_sectors.by_key[(sub_sector.name, sector.id)] = sub_sector

    await service.update_admin_record(
        record_id=1,
        payload={
            "project_title": "Existing Project",
            "beneficiary_country": "Existing Country",
            "funding_provider": "Existing Provider",
            "source": "Existing Source",
            "source_url": "https://updated.example",
            "financial_instrument": "Loan",
            "sector": "Transport",
            "sub_sector": "Rail",
        },
    )

    assert record.project_id == 1
    assert record.beneficiary_country_id == 2
    assert record.funding_provider_id == 3
    assert record.source_id == 4
    assert source.url == "https://updated.example"
    assert record.financial_instrument_id == 5
    assert record.sector_id == 6
    assert record.sub_sector_id == 7

    with pytest.raises(ValueError, match="Source and source_url are required together"):
        await service.update_admin_record(
            record_id=1,
            payload={"source": "Only Source"},
        )


@pytest.mark.asyncio
async def test_update_admin_record_raises_for_missing_record_and_missing_view():
    uow = FakeRecordUoW()
    service = RecordService(uow)

    with pytest.raises(ValueError, match="Record not found"):
        await service.update_admin_record(record_id=999, payload={})

    record = _Record()
    uow.climate_finance_records.record_by_id[1] = record

    with pytest.raises(
        ValueError,
        match="Updated record could not be loaded",
    ):
        await service.update_admin_record(record_id=1, payload={})
