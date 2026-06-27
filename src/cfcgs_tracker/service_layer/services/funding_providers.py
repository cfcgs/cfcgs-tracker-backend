from src.cfcgs_tracker.domain.exceptions import PermissionDeniedError
from src.cfcgs_tracker.domain.models import FundFocus, FundType, User, UserRole
from src.cfcgs_tracker.service_layer.unit_of_work import AbstractUnitOfWork

CFU_SOURCE = [
    {
        "label": "Climate Funds Update (CFU)",
        "url": "https://climatefundsupdate.org/data-dashboard/",
    }
]


class FundingProviderService:
    def __init__(self, uow: AbstractUnitOfWork) -> None:
        self.uow = uow

    async def get_funding_providers(
        self,
        *,
        fund_type_ids: list[int] | None = None,
        fund_focus_ids: list[int] | None = None,
        limit: int = 40,
        offset: int = 0,
    ) -> dict:
        rows = await self.uow.funding_providers.list_funding_providers(
            fund_type_ids=fund_type_ids,
            fund_focus_ids=fund_focus_ids,
            limit=limit,
            offset=offset,
        )
        return {
            "funding_providers": [dict(row) for row in rows],
            "sources": CFU_SOURCE,
        }

    async def get_funding_provider_summary(
        self,
        *,
        funding_provider_ids: list[int] | None = None,
        fund_type_ids: list[int] | None = None,
        fund_focus_ids: list[int] | None = None,
    ) -> dict:
        row = await self.uow.funding_providers.summarize_funding_providers(
            funding_provider_ids=funding_provider_ids,
            fund_type_ids=fund_type_ids,
            fund_focus_ids=fund_focus_ids,
        )
        payload = dict(row)
        payload["sources"] = CFU_SOURCE
        return payload

    async def update_funding_provider_profile(
        self,
        *,
        funding_provider_id: int,
        payload: dict,
        current_user: User,
    ) -> dict:
        if current_user.role not in {UserRole.admin, UserRole.importer}:
            raise PermissionDeniedError()

        provider = await self.uow.funding_providers.get_by_id(
            funding_provider_id
        )
        if not provider:
            raise ValueError("Funding provider not found")

        profile = (
            await self.uow.provider_fund_profiles.get_by_funding_provider_id(
                funding_provider_id
            )
        )
        if not profile:
            raise ValueError("Funding provider profile not found")

        if "fund_type" in payload:
            profile.fund_type_id = await self._resolve_fund_type_id(
                payload.get("fund_type")
            )
        if "fund_focus" in payload:
            profile.fund_focus_id = await self._resolve_fund_focus_id(
                payload.get("fund_focus")
            )

        for field_name in (
            "pledge",
            "deposit",
            "approval",
            "disbursement",
            "projects_approved",
        ):
            if field_name in payload:
                setattr(profile, field_name, payload[field_name])

        await self.uow.commit()
        row = await self.uow.funding_providers.get_funding_provider_row_by_id(
            funding_provider_id
        )
        if row:
            return row

        raise ValueError(
            "Updated funding provider profile could not be loaded"
        )

    async def _resolve_fund_type_id(self, name: str | None) -> int | None:
        name = (name or "").strip()
        if not name:
            return None
        fund_type = await self.uow.fund_types.get_by_name(name)
        if fund_type:
            return fund_type.id
        fund_type = FundType(name=name)
        await self.uow.fund_types.add(fund_type)
        await self.uow.flush()
        return fund_type.id

    async def _resolve_fund_focus_id(self, name: str | None) -> int | None:
        name = (name or "").strip()
        if not name:
            return None
        fund_focus = await self.uow.fund_focuses.get_by_name(name)
        if fund_focus:
            return fund_focus.id
        fund_focus = FundFocus(name=name)
        await self.uow.fund_focuses.add(fund_focus)
        await self.uow.flush()
        return fund_focus.id
