from src.cfcgs_tracker.domain.models import (
    BeneficiaryCountry,
    FundFocus,
    FundType,
)
from src.cfcgs_tracker.service_layer.unit_of_work import AbstractUnitOfWork


class ReferenceReadService:
    def __init__(self, uow: AbstractUnitOfWork) -> None:
        self.uow = uow

    async def get_fund_types(self) -> list[FundType]:
        return await self.uow.fund_types.list_all()

    async def get_fund_focuses(self) -> list[FundFocus]:
        return await self.uow.fund_focuses.list_all()

    async def get_beneficiary_countries(self) -> list[BeneficiaryCountry]:
        return await self.uow.beneficiary_countries.list_all()
