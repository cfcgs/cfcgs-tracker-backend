from http import HTTPStatus

import pytest


@pytest.mark.asyncio
async def test_read_fund_types(client, session, fund_type_factory):
    await fund_type_factory(name="Adaptation")
    await fund_type_factory(name="Mitigation")
    await session.commit()

    response = client.get("/fund_types/")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "fund_types": [
            {"id": 1, "name": "Adaptation"},
            {"id": 2, "name": "Mitigation"},
        ]
    }


@pytest.mark.asyncio
async def test_read_fund_focuses(client, session, fund_focus_factory):
    await fund_focus_factory(name="Crosscutting")
    await fund_focus_factory(name="Resilience")
    await session.commit()

    response = client.get("/fund_focuses/")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "fund_focuses": [
            {"id": 1, "name": "Crosscutting"},
            {"id": 2, "name": "Resilience"},
        ]
    }


@pytest.mark.asyncio
async def test_read_beneficiary_countries(
    client,
    session,
    beneficiary_country_factory,
):
    await beneficiary_country_factory(name="Kenya")
    await beneficiary_country_factory(name="Peru")
    await session.commit()

    response = client.get("/beneficiary_countries/")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "countries": [
            {"id": 1, "name": "Kenya", "region": None},
            {"id": 2, "name": "Peru", "region": None},
        ]
    }
