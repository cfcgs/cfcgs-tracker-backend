import pytest

from src.cfcgs_tracker.domain.exceptions import PermissionDeniedError
from src.cfcgs_tracker.domain.models import (
    FundFocus,
    FundType,
    ProviderFundProfile,
    User,
    UserRole,
)
from src.cfcgs_tracker.service_layer.services.funding_providers import (
    CFU_SOURCE,
    FundingProviderService,
)
from src.cfcgs_tracker.service_layer.unit_of_work import AbstractUnitOfWork


class _FundingProviderRepo:
    def __init__(self) -> None:
        self.rows = []
        self.summary_row = {}
        self.by_id = {}
        self.row_by_id = {}

    async def list_funding_providers(self, **kwargs):
        return self.rows

    async def summarize_funding_providers(self, **kwargs):
        return self.summary_row

    async def get_by_id(self, funding_provider_id: int):
        return self.by_id.get(funding_provider_id)

    async def get_funding_provider_row_by_id(self, funding_provider_id: int):
        return self.row_by_id.get(funding_provider_id)


class _ProfileRepo:
    def __init__(self) -> None:
        self.by_provider_id = {}

    async def get_by_funding_provider_id(self, funding_provider_id: int):
        return self.by_provider_id.get(funding_provider_id)


class _NamedRepo:
    def __init__(self) -> None:
        self.by_name = {}
        self.added = []

    async def get_by_name(self, name: str):
        return self.by_name.get(name)

    async def add(self, instance):
        self.added.append(instance)
        self.by_name[instance.name] = instance


class FakeFundingProviderUoW(AbstractUnitOfWork):
    def __init__(self) -> None:
        self.funding_providers = _FundingProviderRepo()
        self.provider_fund_profiles = _ProfileRepo()
        self.fund_types = _NamedRepo()
        self.fund_focuses = _NamedRepo()
        self._flush_counter = 0
        self.committed = False

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        return None

    async def refresh(self, instance) -> None:
        return None

    async def flush(self) -> None:
        self._flush_counter += 1
        for repository in (self.fund_types, self.fund_focuses):
            for instance in repository.added:
                if getattr(instance, "id", None) is None:
                    instance.id = self._flush_counter

    def begin_nested(self):
        return None

    async def execute(self, statement, params=None):
        return None


def _build_user(role: UserRole | str) -> User:
    user = User(
        username=f"{str(role)}-user",
        email=f"{str(role)}@example.com",
        password="secret",
        role=UserRole.importer,
    )
    user.id = 1
    user.role = role
    return user


@pytest.mark.asyncio
async def test_get_funding_providers_returns_rows_with_sources():
    uow = FakeFundingProviderUoW()
    uow.funding_providers.rows = [
        {
            "id": 1,
            "funding_provider_name": "Provider A",
            "fund_type": "Type A",
            "fund_focus": "Focus A",
            "pledge": 100.0,
            "deposit": 80.0,
            "approval": 70.0,
            "disbursement": 60.0,
            "projects_approved": 3,
        }
    ]
    service = FundingProviderService(uow)

    payload = await service.get_funding_providers(limit=10, offset=0)

    assert payload["funding_providers"][0]["funding_provider_name"] == "Provider A"
    assert payload["sources"] == CFU_SOURCE


@pytest.mark.asyncio
async def test_get_funding_provider_summary_returns_sources():
    uow = FakeFundingProviderUoW()
    uow.funding_providers.summary_row = {
        "total_pledge": 150.0,
        "total_deposit": 120.0,
        "total_approval": 90.0,
    }
    service = FundingProviderService(uow)

    payload = await service.get_funding_provider_summary()

    assert payload["total_pledge"] == 150.0
    assert payload["sources"] == CFU_SOURCE


@pytest.mark.asyncio
async def test_update_funding_provider_profile_validates_permissions_and_presence():
    uow = FakeFundingProviderUoW()
    service = FundingProviderService(uow)

    with pytest.raises(PermissionDeniedError):
        await service.update_funding_provider_profile(
            funding_provider_id=1,
            payload={},
            current_user=_build_user("unauthorized"),
        )

    with pytest.raises(ValueError, match="Funding provider not found"):
        await service.update_funding_provider_profile(
            funding_provider_id=1,
            payload={},
            current_user=_build_user(UserRole.admin),
        )

    uow.funding_providers.by_id[1] = object()
    with pytest.raises(
        ValueError,
        match="Funding provider profile not found",
    ):
        await service.update_funding_provider_profile(
            funding_provider_id=1,
            payload={},
            current_user=_build_user(UserRole.importer),
        )


@pytest.mark.asyncio
async def test_update_funding_provider_profile_updates_and_creates_dimensions():
    uow = FakeFundingProviderUoW()
    service = FundingProviderService(uow)
    profile = ProviderFundProfile(funding_provider_id=1)
    profile.id = 8
    uow.funding_providers.by_id[1] = object()
    uow.provider_fund_profiles.by_provider_id[1] = profile
    uow.funding_providers.row_by_id[1] = {
        "id": 1,
        "funding_provider_name": "Provider A",
        "fund_type": "New Type",
        "fund_focus": "New Focus",
        "pledge": 100.0,
        "deposit": 90.0,
        "approval": 80.0,
        "disbursement": 70.0,
        "projects_approved": 4,
    }

    payload = await service.update_funding_provider_profile(
        funding_provider_id=1,
        payload={
            "fund_type": "New Type",
            "fund_focus": "New Focus",
            "pledge": 100.0,
            "deposit": 90.0,
            "approval": 80.0,
            "disbursement": 70.0,
            "projects_approved": 4,
        },
        current_user=_build_user(UserRole.admin),
    )

    assert uow.committed is True
    assert payload["fund_type"] == "New Type"
    assert profile.pledge == 100.0
    assert profile.projects_approved == 4
    assert isinstance(profile.fund_type_id, int)
    assert isinstance(profile.fund_focus_id, int)


@pytest.mark.asyncio
async def test_update_funding_provider_profile_reuses_existing_dimensions():
    uow = FakeFundingProviderUoW()
    service = FundingProviderService(uow)
    fund_type = FundType(name="Existing Type")
    fund_type.id = 10
    fund_focus = FundFocus(name="Existing Focus")
    fund_focus.id = 20
    uow.fund_types.by_name["Existing Type"] = fund_type
    uow.fund_focuses.by_name["Existing Focus"] = fund_focus
    profile = ProviderFundProfile(funding_provider_id=1)
    profile.id = 9
    uow.funding_providers.by_id[1] = object()
    uow.provider_fund_profiles.by_provider_id[1] = profile
    uow.funding_providers.row_by_id[1] = {
        "id": 1,
        "funding_provider_name": "Provider B",
        "fund_type": "Existing Type",
        "fund_focus": "Existing Focus",
        "pledge": None,
        "deposit": None,
        "approval": None,
        "disbursement": None,
        "projects_approved": None,
    }

    await service.update_funding_provider_profile(
        funding_provider_id=1,
        payload={
            "fund_type": "Existing Type",
            "fund_focus": "Existing Focus",
        },
        current_user=_build_user(UserRole.importer),
    )

    assert profile.fund_type_id == 10
    assert profile.fund_focus_id == 20


@pytest.mark.asyncio
async def test_update_funding_provider_profile_raises_when_reload_fails():
    uow = FakeFundingProviderUoW()
    service = FundingProviderService(uow)
    profile = ProviderFundProfile(funding_provider_id=1)
    profile.id = 10
    uow.funding_providers.by_id[1] = object()
    uow.provider_fund_profiles.by_provider_id[1] = profile

    with pytest.raises(
        ValueError,
        match="Updated funding provider profile could not be loaded",
    ):
        await service.update_funding_provider_profile(
            funding_provider_id=1,
            payload={"pledge": 1.0},
            current_user=_build_user(UserRole.admin),
        )
