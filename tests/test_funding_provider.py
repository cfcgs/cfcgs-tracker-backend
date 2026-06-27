from http import HTTPStatus

import pytest


@pytest.mark.asyncio
async def test_read_funding_providers(
    client,
    session,
    fund_type_factory,
    fund_focus_factory,
    funding_provider_factory,
    provider_fund_profile_factory,
):
    fund_type = await fund_type_factory(name="Multilateral")
    fund_focus = await fund_focus_factory(name="Adaptation")
    provider = await funding_provider_factory(name="Green Climate Fund")
    await provider_fund_profile_factory(
        funding_provider=provider,
        fund_type=fund_type,
        fund_focus=fund_focus,
        pledge=100.0,
        deposit=80.0,
        approval=60.0,
        disbursement=40.0,
        projects_approved=5,
    )
    await session.commit()

    response = client.post(
        "/funding_providers/",
        json={"fund_types": [], "fund_focuses": []},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "funding_providers": [
            {
                "id": 1,
                "funding_provider_name": "Green Climate Fund",
                "fund_type": "Multilateral",
                "fund_focus": "Adaptation",
                "pledge": 100.0,
                "deposit": 80.0,
                "approval": 60.0,
                "disbursement": 40.0,
                "projects_approved": 5,
            }
        ],
        "sources": [
            {
                "label": "Climate Funds Update (CFU)",
                "url": "https://climatefundsupdate.org/data-dashboard/",
            }
        ],
    }


@pytest.mark.asyncio
async def test_read_funding_provider_summary(
    client,
    session,
    fund_type_factory,
    fund_focus_factory,
    funding_provider_factory,
    provider_fund_profile_factory,
):
    fund_type = await fund_type_factory(name="Multilateral")
    fund_focus = await fund_focus_factory(name="Adaptation")
    provider_one = await funding_provider_factory(name="Provider One")
    provider_two = await funding_provider_factory(name="Provider Two")
    await provider_fund_profile_factory(
        funding_provider=provider_one,
        fund_type=fund_type,
        fund_focus=fund_focus,
        pledge=100.0,
        deposit=80.0,
        approval=60.0,
        disbursement=40.0,
        projects_approved=5,
    )
    await provider_fund_profile_factory(
        funding_provider=provider_two,
        fund_type=fund_type,
        fund_focus=fund_focus,
        pledge=50.0,
        deposit=20.0,
        approval=10.0,
        disbursement=5.0,
        projects_approved=2,
    )
    await session.commit()

    response = client.post(
        "/funding_providers/summary",
        json={
            "funding_providers": [],
            "fund_types": [],
            "fund_focuses": [],
        },
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "total_pledge": 150.0,
        "total_deposit": 100.0,
        "total_approval": 70.0,
        "sources": [
            {
                "label": "Climate Funds Update (CFU)",
                "url": "https://climatefundsupdate.org/data-dashboard/",
            }
        ],
    }
