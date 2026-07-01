import pytest

from src.cfcgs_tracker.domain.models import Sector
from src.cfcgs_tracker.service_layer.services.reference_data import (
    ReferenceDataService,
)
from src.cfcgs_tracker.service_layer.unit_of_work import AbstractUnitOfWork


class _NamedRepo:
    def __init__(self) -> None:
        self.by_name = {}
        self.added = []

    async def get_by_name(self, name: str):
        return self.by_name.get(name)

    async def add(self, instance):
        self.added.append(instance)
        key = getattr(instance, "name", None)
        if key is not None:
            self.by_name[key] = instance


class _ProjectRepo:
    def __init__(self) -> None:
        self.by_title = {}
        self.added = []

    async def get_by_title(self, title: str):
        return self.by_title.get(title)

    async def add(self, instance):
        self.added.append(instance)
        self.by_title[instance.title] = instance


class _SourceRepo:
    def __init__(self) -> None:
        self.by_name = {}
        self.by_url = {}
        self.added = []

    async def get_by_name(self, name: str):
        return self.by_name.get(name)

    async def get_by_url(self, url: str):
        return self.by_url.get(url)

    async def add(self, instance):
        self.added.append(instance)
        self.by_name[instance.name] = instance
        self.by_url[instance.url] = instance


class _SubSectorRepo:
    def __init__(self) -> None:
        self.by_key = {}
        self.added = []

    async def get_by_name_and_sector_id(self, *, name: str, sector_id: int):
        return self.by_key.get((name, sector_id))

    async def add(self, instance):
        self.added.append(instance)
        self.by_key[(instance.name, instance.sector_id)] = instance


class FakeReferenceUoW(AbstractUnitOfWork):
    def __init__(self) -> None:
        self.projects = _ProjectRepo()
        self.funding_providers = _NamedRepo()
        self.sources = _SourceRepo()
        self.fund_types = _NamedRepo()
        self.fund_focuses = _NamedRepo()
        self.financial_instruments = _NamedRepo()
        self.sectors = _NamedRepo()
        self.sub_sectors = _SubSectorRepo()
        self.beneficiary_countries = _NamedRepo()
        self._flush_counter = 0

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None

    async def refresh(self, instance) -> None:
        return None

    async def flush(self) -> None:
        self._flush_counter += 1
        repositories = (
            self.projects,
            self.funding_providers,
            self.sources,
            self.fund_types,
            self.fund_focuses,
            self.financial_instruments,
            self.sectors,
            self.sub_sectors,
            self.beneficiary_countries,
        )
        for repository in repositories:
            for instance in getattr(repository, "added", []):
                if getattr(instance, "id", None) is None:
                    instance.id = self._flush_counter

    def begin_nested(self):
        return None

    async def execute(self, statement, params=None):
        return None


@pytest.mark.asyncio
async def test_reference_data_service_creates_and_caches_entities():
    uow = FakeReferenceUoW()
    service = ReferenceDataService(uow)

    project = await service.get_or_create_project("Projeto A")
    provider = await service.get_or_create_funding_provider("Provider A")
    fund_type = await service.get_or_create_fund_type("Mitigation")
    fund_focus = await service.get_or_create_fund_focus("Adaptation")
    instrument = await service.get_or_create_financial_instrument("Grant")
    sector = await service.get_or_create_sector("Energy")
    country = await service.get_or_create_beneficiary_country("Brazil")

    assert project.title == "Projeto A"
    assert provider.name == "Provider A"
    assert fund_type.name == "Mitigation"
    assert fund_focus.name == "Adaptation"
    assert instrument.name == "Grant"
    assert sector.name == "Energy"
    assert country.name == "Brazil"

    assert await service.get_or_create_project("Projeto A") is project
    assert await service.get_or_create_funding_provider("Provider A") is provider
    assert await service.get_or_create_fund_type("Mitigation") is fund_type
    assert await service.get_or_create_fund_focus("Adaptation") is fund_focus
    assert (
        await service.get_or_create_financial_instrument("Grant")
        is instrument
    )
    assert await service.get_or_create_sector("Energy") is sector
    assert await service.get_or_create_beneficiary_country("Brazil") is country


@pytest.mark.asyncio
async def test_reference_data_service_handles_nullable_inputs():
    uow = FakeReferenceUoW()
    service = ReferenceDataService(uow)

    assert await service.get_or_create_project(None) is None
    assert await service.get_or_create_funding_provider(None) is None
    assert await service.get_or_create_fund_type(None) is None
    assert await service.get_or_create_fund_focus(None) is None
    assert await service.get_or_create_financial_instrument(None) is None
    assert await service.get_or_create_sector(None) is None
    assert await service.get_or_create_beneficiary_country(None) is None
    assert await service.get_or_create_sub_sector(None, None) is None


@pytest.mark.asyncio
async def test_reference_data_service_reuses_source_by_name_and_updates_url():
    uow = FakeReferenceUoW()
    service = ReferenceDataService(uow)

    source = await service.get_or_create_source("OECD", "https://old.example")
    updated_source = await service.get_or_create_source(
        "OECD",
        "https://new.example",
    )

    assert updated_source is source
    assert source.url == "https://new.example"


@pytest.mark.asyncio
async def test_reference_data_service_reuses_source_by_url_and_cache_reset():
    uow = FakeReferenceUoW()
    service = ReferenceDataService(uow)

    source = await service.get_or_create_source("OECD", "https://same.example")
    uow.sources.by_name.clear()
    service.reset_cache()
    uow.sources.by_url["https://same.example"] = source

    reused_source = await service.get_or_create_source(
        "Other name",
        "https://same.example",
    )

    assert reused_source is source


@pytest.mark.asyncio
async def test_reference_data_service_creates_sub_sector_per_sector():
    uow = FakeReferenceUoW()
    service = ReferenceDataService(uow)
    sector = Sector(name="Energy")
    sector.id = 10

    sub_sector = await service.get_or_create_sub_sector(
        "Transmission",
        sector,
    )
    cached_sub_sector = await service.get_or_create_sub_sector(
        "Transmission",
        sector,
    )

    assert sub_sector.name == "Transmission"
    assert sub_sector.sector_id == 10
    assert cached_sub_sector is sub_sector
