from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.adapters.repositories.beneficiary_country_repository import (
    BeneficiaryCountryRepository,
)
from src.cfcgs_tracker.adapters.repositories.climate_finance_record_repository import (
    ClimateFinanceRecordRepository,
)
from src.cfcgs_tracker.adapters.repositories.financial_instrument_repository import (
    FinancialInstrumentRepository,
)
from src.cfcgs_tracker.adapters.repositories.fund_focus_repository import (
    FundFocusRepository,
)
from src.cfcgs_tracker.adapters.repositories.fund_type_repository import (
    FundTypeRepository,
)
from src.cfcgs_tracker.adapters.repositories.funding_provider_repository import (
    FundingProviderRepository,
)
from src.cfcgs_tracker.adapters.repositories.import_job_repository import (
    ImportJobRepository,
)
from src.cfcgs_tracker.adapters.repositories.project_repository import (
    ProjectRepository,
)
from src.cfcgs_tracker.adapters.repositories.provider_fund_profile_repository import (
    ProviderFundProfileRepository,
)
from src.cfcgs_tracker.adapters.repositories.sector_repository import (
    SectorRepository,
)
from src.cfcgs_tracker.adapters.repositories.source_repository import (
    SourceRepository,
)
from src.cfcgs_tracker.adapters.repositories.sub_sector_repository import (
    SubSectorRepository,
)
from src.cfcgs_tracker.adapters.repositories.user_repository import (
    UserRepository,
)


class AbstractUnitOfWork(ABC):
    beneficiary_countries: BeneficiaryCountryRepository
    climate_finance_records: ClimateFinanceRecordRepository
    financial_instruments: FinancialInstrumentRepository
    fund_focuses: FundFocusRepository
    fund_types: FundTypeRepository
    funding_providers: FundingProviderRepository
    import_jobs: ImportJobRepository
    projects: ProjectRepository
    provider_fund_profiles: ProviderFundProfileRepository
    sectors: SectorRepository
    sources: SourceRepository
    sub_sectors: SubSectorRepository
    users: UserRepository

    async def __aenter__(self) -> AbstractUnitOfWork:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if exc_type:
            await self.rollback()

    @abstractmethod
    async def commit(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def rollback(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def refresh(self, instance) -> None:
        raise NotImplementedError

    @abstractmethod
    async def flush(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def begin_nested(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, statement, params: Any = None) -> Any:
        raise NotImplementedError


class SqlAlchemyUnitOfWork(AbstractUnitOfWork):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.beneficiary_countries = BeneficiaryCountryRepository(self.session)
        self.climate_finance_records = ClimateFinanceRecordRepository(
            self.session
        )
        self.financial_instruments = FinancialInstrumentRepository(
            self.session
        )
        self.fund_focuses = FundFocusRepository(self.session)
        self.fund_types = FundTypeRepository(self.session)
        self.funding_providers = FundingProviderRepository(self.session)
        self.import_jobs = ImportJobRepository(self.session)
        self.projects = ProjectRepository(self.session)
        self.provider_fund_profiles = ProviderFundProfileRepository(
            self.session
        )
        self.sectors = SectorRepository(self.session)
        self.sources = SourceRepository(self.session)
        self.sub_sectors = SubSectorRepository(self.session)
        self.users = UserRepository(self.session)

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()

    async def refresh(self, instance):
        await self.session.refresh(instance)

    async def flush(self) -> None:
        await self.session.flush()

    def begin_nested(self):
        return self.session.begin_nested()

    async def execute(self, statement, params: Any = None) -> Any:
        return await self.session.execute(statement, params)
