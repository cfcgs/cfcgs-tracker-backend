import pytest

from src.cfcgs_tracker.service_layer.import_schemas import FundRowSchema
from src.cfcgs_tracker.service_layer.services.provider_fund_profiles import (
    FundStatusImportService,
)
from src.cfcgs_tracker.service_layer.unit_of_work import AbstractUnitOfWork


class FakeResult:
    def __init__(self, rows) -> None:
        self._rows = rows

    def all(self):
        return self._rows


class FakeProviderFundProfileImportUoW(AbstractUnitOfWork):
    def __init__(self, existing_profiles=None) -> None:
        self.existing_profiles = existing_profiles or []
        self.executed = []
        self.rollback_calls = 0

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        self.rollback_calls += 1

    async def refresh(self, instance) -> None:
        return None

    async def flush(self) -> None:
        return None

    def begin_nested(self):
        return None

    async def execute(self, statement, params=None):
        if getattr(statement, "__visit_name__", "") == "select":
            return FakeResult(self.existing_profiles)

        self.executed.append((statement, params))
        return FakeResult([])


class _TestableFundStatusImportService(FundStatusImportService):
    def __init__(self, uow: AbstractUnitOfWork, mappings: dict[str, dict]) -> None:
        super().__init__(uow)
        self._mappings = mappings

    async def _ensure_named_dimension(
        self,
        model,
        field_name: str,
        values: list[str | None],
    ) -> dict[str, int]:
        return self._mappings[field_name]


@pytest.mark.asyncio
async def test_bulk_persist_counts_new_and_duplicated_fund_rows():
    uow = FakeProviderFundProfileImportUoW(
        existing_profiles=[
            (1, 10, 20, 100.0, 80.0, 60.0, 40.0, 5),
        ]
    )
    service = _TestableFundStatusImportService(
        uow,
        mappings={
            "name": {"Fund A": 1, "Fund B": 2},
        },
    )
    service._mappings = {
        "name": {"Fund A": 1, "Fund B": 2},
    }
    original_ensure = service._ensure_named_dimension

    async def _ensure(model, field_name, values):
        if model.__name__ == "FundingProvider":
            return {"Fund A": 1, "Fund B": 2}
        if model.__name__ == "FundType":
            return {"Type A": 10}
        return {"Focus A": 20}

    service._ensure_named_dimension = _ensure  # type: ignore[method-assign]

    rows = [
        FundRowSchema(
            fund="Fund A",
            fund_type="Type A",
            fund_focus="Focus A",
            pledge_usd_mn=100.0,
            deposit_usd_mn=80.0,
            approval_usd_mn=60.0,
            disbursement_usd_mn=40.0,
            number_of_projects_approved=5,
        ),
        FundRowSchema(
            fund="Fund B",
            fund_type="Type A",
            fund_focus="Focus A",
            pledge_usd_mn=90.0,
            deposit_usd_mn=70.0,
            approval_usd_mn=50.0,
            disbursement_usd_mn=30.0,
            number_of_projects_approved=3,
        ),
        FundRowSchema(
            fund="Fund B",
            fund_type="Type A",
            fund_focus="Focus A",
            pledge_usd_mn=90.0,
            deposit_usd_mn=70.0,
            approval_usd_mn=50.0,
            disbursement_usd_mn=30.0,
            number_of_projects_approved=3,
        ),
    ]

    inserted, updated, duplicated = await service._bulk_persist(rows)

    service._ensure_named_dimension = original_ensure  # type: ignore[method-assign]
    assert inserted == 1
    assert updated == 0
    assert duplicated == 2
    assert len(uow.executed) == 1


@pytest.mark.asyncio
async def test_bulk_persist_updates_existing_profile_when_values_change():
    uow = FakeProviderFundProfileImportUoW(
        existing_profiles=[
            (1, 10, 20, 100.0, 80.0, 60.0, 40.0, 5),
        ]
    )
    service = _TestableFundStatusImportService(
        uow,
        mappings={"name": {"Fund A": 1}},
    )

    async def _ensure(model, field_name, values):
        if model.__name__ == "FundingProvider":
            return {"Fund A": 1}
        if model.__name__ == "FundType":
            return {"Type A": 10}
        return {"Focus A": 20}

    service._ensure_named_dimension = _ensure  # type: ignore[method-assign]

    inserted, updated, duplicated = await service._bulk_persist(
        [
            FundRowSchema(
                fund="Fund A",
                fund_type="Type A",
                fund_focus="Focus A",
                pledge_usd_mn=111.0,
                deposit_usd_mn=80.0,
                approval_usd_mn=60.0,
                disbursement_usd_mn=40.0,
                number_of_projects_approved=5,
            )
        ]
    )

    assert inserted == 0
    assert updated == 1
    assert duplicated == 0
    assert len(uow.executed) == 1
    _, params = uow.executed[0]
    assert params[0]["b_funding_provider_id"] == 1
    assert params[0]["pledge"] == 111.0


@pytest.mark.asyncio
async def test_import_rows_falls_back_to_row_by_row_and_counts_failures():
    uow = FakeProviderFundProfileImportUoW()
    service = FundStatusImportService(uow)

    calls = {"count": 0}

    async def fake_bulk_persist(rows):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("batch failure")
        if rows[0].fund == "Broken Fund":
            raise ValueError("row failure")
        return (1, 0, 0)

    service._bulk_persist = fake_bulk_persist  # type: ignore[method-assign]

    inserted, updated, duplicated, failed = await service.import_rows(
        [
            FundRowSchema(fund="Good Fund"),
            FundRowSchema(fund="Broken Fund"),
        ]
    )

    assert inserted == 1
    assert updated == 0
    assert duplicated == 0
    assert failed == 1
    assert uow.rollback_calls == 2
