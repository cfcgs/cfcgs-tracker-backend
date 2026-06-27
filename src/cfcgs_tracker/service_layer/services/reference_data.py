from src.cfcgs_tracker.domain.models import (
    BeneficiaryCountry,
    FinancialInstrument,
    FundFocus,
    FundType,
    FundingProvider,
    Project,
    Sector,
    Source,
    SubSector,
)
from src.cfcgs_tracker.service_layer.unit_of_work import AbstractUnitOfWork


class ReferenceDataService:
    def __init__(self, uow: AbstractUnitOfWork) -> None:
        self.uow = uow
        self._projects: dict[str, Project] = {}
        self._funding_providers: dict[str, FundingProvider] = {}
        self._sources_by_name: dict[str, Source] = {}
        self._sources_by_url: dict[str, Source] = {}
        self._fund_types: dict[str, FundType] = {}
        self._fund_focuses: dict[str, FundFocus] = {}
        self._financial_instruments: dict[str, FinancialInstrument] = {}
        self._sectors: dict[str, Sector] = {}
        self._sub_sectors: dict[tuple[str, int], SubSector] = {}
        self._beneficiary_countries: dict[str, BeneficiaryCountry] = {}

    def reset_cache(self) -> None:
        self._projects.clear()
        self._funding_providers.clear()
        self._sources_by_name.clear()
        self._sources_by_url.clear()
        self._fund_types.clear()
        self._fund_focuses.clear()
        self._financial_instruments.clear()
        self._sectors.clear()
        self._sub_sectors.clear()
        self._beneficiary_countries.clear()

    async def get_or_create_project(self, title: str | None) -> Project | None:
        if not title:
            return None

        if title in self._projects:
            return self._projects[title]

        project = await self.uow.projects.get_by_title(title)
        if project:
            self._projects[title] = project
            return project

        project = Project(title=title)
        await self.uow.projects.add(project)
        await self.uow.flush()
        self._projects[title] = project
        return project

    async def get_or_create_funding_provider(
        self,
        name: str | None,
    ) -> FundingProvider | None:
        if not name:
            return None

        if name in self._funding_providers:
            return self._funding_providers[name]

        provider = await self.uow.funding_providers.get_by_name(name)
        if provider:
            self._funding_providers[name] = provider
            return provider

        provider = FundingProvider(name=name)
        await self.uow.funding_providers.add(provider)
        await self.uow.flush()
        self._funding_providers[name] = provider
        return provider

    async def get_or_create_source(self, name: str, url: str) -> Source:
        if name in self._sources_by_name:
            source = self._sources_by_name[name]
            if source.url != url:
                source.url = url
            self._sources_by_url[source.url] = source
            return source

        if url in self._sources_by_url:
            return self._sources_by_url[url]

        source = await self.uow.sources.get_by_name(name)
        if source:
            if source.url != url:
                source.url = url
            self._sources_by_name[name] = source
            self._sources_by_url[source.url] = source
            return source

        source = await self.uow.sources.get_by_url(url)
        if source:
            self._sources_by_name[source.name] = source
            self._sources_by_url[url] = source
            return source

        source = Source(name=name, url=url)
        await self.uow.sources.add(source)
        await self.uow.flush()
        self._sources_by_name[name] = source
        self._sources_by_url[url] = source
        return source

    async def get_or_create_fund_type(
        self,
        name: str | None,
    ) -> FundType | None:
        if not name:
            return None

        if name in self._fund_types:
            return self._fund_types[name]

        fund_type = await self.uow.fund_types.get_by_name(name)
        if fund_type:
            self._fund_types[name] = fund_type
            return fund_type

        fund_type = FundType(name=name)
        await self.uow.fund_types.add(fund_type)
        await self.uow.flush()
        self._fund_types[name] = fund_type
        return fund_type

    async def get_or_create_fund_focus(
        self,
        name: str | None,
    ) -> FundFocus | None:
        if not name:
            return None

        if name in self._fund_focuses:
            return self._fund_focuses[name]

        fund_focus = await self.uow.fund_focuses.get_by_name(name)
        if fund_focus:
            self._fund_focuses[name] = fund_focus
            return fund_focus

        fund_focus = FundFocus(name=name)
        await self.uow.fund_focuses.add(fund_focus)
        await self.uow.flush()
        self._fund_focuses[name] = fund_focus
        return fund_focus

    async def get_or_create_financial_instrument(
        self,
        name: str | None,
    ) -> FinancialInstrument | None:
        if not name:
            return None

        if name in self._financial_instruments:
            return self._financial_instruments[name]

        instrument = await self.uow.financial_instruments.get_by_name(name)
        if instrument:
            self._financial_instruments[name] = instrument
            return instrument

        instrument = FinancialInstrument(name=name)
        await self.uow.financial_instruments.add(instrument)
        await self.uow.flush()
        self._financial_instruments[name] = instrument
        return instrument

    async def get_or_create_sector(self, name: str | None) -> Sector | None:
        if not name:
            return None

        if name in self._sectors:
            return self._sectors[name]

        sector = await self.uow.sectors.get_by_name(name)
        if sector:
            self._sectors[name] = sector
            return sector

        sector = Sector(name=name)
        await self.uow.sectors.add(sector)
        await self.uow.flush()
        self._sectors[name] = sector
        return sector

    async def get_or_create_sub_sector(
        self,
        name: str | None,
        sector: Sector | None,
    ) -> SubSector | None:
        if not name or sector is None:
            return None

        cache_key = (name, sector.id)
        if cache_key in self._sub_sectors:
            return self._sub_sectors[cache_key]

        sub_sector = await self.uow.sub_sectors.get_by_name_and_sector_id(
            name=name,
            sector_id=sector.id,
        )
        if sub_sector:
            self._sub_sectors[cache_key] = sub_sector
            return sub_sector

        sub_sector = SubSector(name=name, sector_id=sector.id)
        await self.uow.sub_sectors.add(sub_sector)
        await self.uow.flush()
        self._sub_sectors[cache_key] = sub_sector
        return sub_sector

    async def get_or_create_beneficiary_country(
        self,
        name: str | None,
    ) -> BeneficiaryCountry | None:
        if not name:
            return None

        if name in self._beneficiary_countries:
            return self._beneficiary_countries[name]

        country = await self.uow.beneficiary_countries.get_by_name(name)
        if country:
            self._beneficiary_countries[name] = country
            return country

        country = BeneficiaryCountry(name=name)
        await self.uow.beneficiary_countries.add(country)
        await self.uow.flush()
        self._beneficiary_countries[name] = country
        return country
