import pytest

from src.cfcgs_tracker.adapters.repositories.funding_provider_repository import (
    FundingProviderRepository,
)
from tests.conftest import (
    FundFocusFactory,
    FundTypeFactory,
    FundingProviderFactory,
    ProviderFundProfileFactory,
)


@pytest.mark.asyncio
async def test_funding_provider_repository_lists_and_summarizes_with_filters(
    session,
):
    fund_type_a = FundTypeFactory(name="Type A")
    fund_type_b = FundTypeFactory(name="Type B")
    fund_focus_a = FundFocusFactory(name="Focus A")
    fund_focus_b = FundFocusFactory(name="Focus B")
    session.add_all([fund_type_a, fund_type_b, fund_focus_a, fund_focus_b])
    await session.flush()

    provider_a = FundingProviderFactory(name="Provider A")
    provider_b = FundingProviderFactory(name="Provider B")
    session.add_all([provider_a, provider_b])
    await session.flush()

    session.add_all(
        [
            ProviderFundProfileFactory(
                funding_provider_id=provider_a.id,
                fund_type_id=fund_type_a.id,
                fund_focus_id=fund_focus_a.id,
                pledge=100.0,
                deposit=80.0,
                approval=60.0,
            ),
            ProviderFundProfileFactory(
                funding_provider_id=provider_b.id,
                fund_type_id=fund_type_b.id,
                fund_focus_id=fund_focus_b.id,
                pledge=50.0,
                deposit=20.0,
                approval=10.0,
            ),
        ]
    )
    await session.commit()

    repository = FundingProviderRepository(session)

    rows = await repository.list_funding_providers(
        fund_type_ids=[fund_type_a.id],
        fund_focus_ids=[fund_focus_a.id],
    )
    summary = await repository.summarize_funding_providers(
        fund_type_ids=[fund_type_a.id],
        fund_focus_ids=[fund_focus_a.id],
    )
    detail = await repository.get_funding_provider_row_by_id(provider_a.id)

    assert len(rows) == 1
    assert rows[0]["funding_provider_name"] == "Provider A"
    assert rows[0]["fund_type"] == "Type A"
    assert summary["total_pledge"] == 100.0
    assert summary["total_deposit"] == 80.0
    assert detail["fund_focus"] == "Focus A"
