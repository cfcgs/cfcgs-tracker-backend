from http import HTTPStatus

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def seeded_records(
    session,
    beneficiary_country_factory,
    funding_provider_factory,
    project_factory,
    source_factory,
    climate_finance_record_factory,
):
    brazil = await beneficiary_country_factory(name="Brazil")
    kenya = await beneficiary_country_factory(name="Kenya")
    provider = await funding_provider_factory(name="Provider Alpha")
    source = await source_factory(name="OECD", url="https://oecd.org")
    project_one = await project_factory(title="Solar")
    project_two = await project_factory(title="Forest")

    await climate_finance_record_factory(
        year=2022,
        source_id=source.id,
        source_row_hash="hash-a",
        project_id=project_one.id,
        funding_provider_id=provider.id,
        beneficiary_country_id=brazil.id,
        climate_finance_amount_usd_millions=10.0,
        adaptation_amount_usd_millions=10.0,
        mitigation_amount_usd_millions=0.0,
        both_objectives_amount_usd_millions=0.0,
    )
    await climate_finance_record_factory(
        year=2023,
        source_id=source.id,
        source_row_hash="hash-b",
        project_id=project_two.id,
        funding_provider_id=provider.id,
        beneficiary_country_id=kenya.id,
        climate_finance_amount_usd_millions=20.0,
        adaptation_amount_usd_millions=0.0,
        mitigation_amount_usd_millions=12.0,
        both_objectives_amount_usd_millions=8.0,
    )
    await session.commit()
    return {
        "brazil": brazil,
        "kenya": kenya,
        "project_one": project_one,
        "project_two": project_two,
    }


@pytest.mark.asyncio
async def test_read_record_years(client, seeded_records):
    response = client.get("/records/years")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == [2023, 2022]


@pytest.mark.asyncio
async def test_search_records(client, seeded_records):
    response = client.post(
        "/records/search?limit=10&offset=0",
        json={"years": [2022], "countries": [seeded_records["brazil"].id]},
    )

    assert response.status_code == HTTPStatus.OK
    payload = response.json()

    assert len(payload["records"]) == 1
    assert payload["records"][0]["project"] == "Solar"
    assert payload["records"][0]["recipient_country"] == "Brazil"
    assert payload["records"][0]["amount_usd_millions"] == 10.0


@pytest.mark.asyncio
async def test_read_record_summary(client, seeded_records):
    response = client.get("/records/summary")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "total_projects": 2,
        "total_funded_countries": 2,
        "total_countries": 2,
        "total_amount": 30.0,
        "total_adaptation": 10.0,
        "total_mitigation": 12.0,
        "total_overlap": 8.0,
        "sources": [{"label": "OECD", "url": "https://oecd.org"}],
    }


@pytest.mark.asyncio
async def test_read_record_aggregations(client, seeded_records):
    objective_response = client.get("/records/aggregations/by-objective")
    year_response = client.get(
        f"/records/aggregations/by-year?country_id={seeded_records['brazil'].id}"
    )

    assert objective_response.status_code == HTTPStatus.OK
    assert objective_response.json() == {
        "totals": [
            {
                "year": 2022,
                "total_adaptation": 10.0,
                "total_mitigation": 0.0,
                "total_overlap": 0.0,
            },
            {
                "year": 2023,
                "total_adaptation": 0.0,
                "total_mitigation": 12.0,
                "total_overlap": 8.0,
            },
        ],
        "sources": [{"label": "OECD", "url": "https://oecd.org"}],
    }

    assert year_response.status_code == HTTPStatus.OK
    assert year_response.json() == {
        "series": [
            {
                "name": "Brazil",
                "data": [{"year": 2022, "amount": 10.0}],
            }
        ],
        "sources": [{"label": "OECD", "url": "https://oecd.org"}],
    }


@pytest.mark.asyncio
async def test_read_record_filter_options_and_projects(client, seeded_records):
    filter_response = client.get("/records/filter-options")
    projects_response = client.get("/projects/records/paginated?search=So")

    assert filter_response.status_code == HTTPStatus.OK
    filter_payload = filter_response.json()
    assert filter_payload["years"] == [2023, 2022]
    assert filter_payload["objectives"] == [
        "all",
        "adaptation",
        "mitigation",
        "both",
    ]
    assert {country["name"] for country in filter_payload["countries"]} == {
        "Brazil",
        "Kenya",
    }

    assert projects_response.status_code == HTTPStatus.OK
    assert projects_response.json() == {
        "projects": [
            {"id": seeded_records["project_one"].id, "name": "Solar"}
        ],
        "total": 1,
        "limit": 20,
        "offset": 0,
        "has_more": False,
    }


@pytest.mark.asyncio
async def test_read_country_year_grid_and_projects_by_cell(
    client,
    seeded_records,
):
    grid_response = client.get(
        "/records/aggregations/by-country-and-year?view=country_year"
    )
    cell_projects_response = client.get(
        f"/records/projects/by-country-and-year?year=2022&country_id={seeded_records['brazil'].id}"
    )

    assert grid_response.status_code == HTTPStatus.OK
    grid_payload = grid_response.json()
    assert grid_payload["grand_total"] == 30.0
    assert grid_payload["grand_total_projects"] == 2
    assert len(grid_payload["cells"]) == 2
    assert grid_payload["sources"] == [
        {"label": "OECD", "url": "https://oecd.org"}
    ]

    assert cell_projects_response.status_code == HTTPStatus.OK
    assert cell_projects_response.json() == {
        "total": 1,
        "has_more": False,
        "projects": [
            {
                "id": seeded_records["project_one"].id,
                "name": "Solar",
                "objective": "adaptation",
                "total_amount": 10.0,
                "adaptation_exclusive": 10.0,
                "mitigation_exclusive": 0.0,
                "overlap": 0.0,
            }
        ],
    }


@pytest.mark.asyncio
async def test_read_admin_record_grid_and_filter_options(
    client,
    token,
    seeded_records,
):
    grid_response = client.get(
        "/records/admin/grid",
        params={
            "year": [2023],
            "beneficiary_country": ["Kenya"],
            "sort_by": "year",
            "sort_order": "desc",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    filter_response = client.get(
        "/records/admin/filter-options",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert grid_response.status_code == HTTPStatus.OK
    grid_payload = grid_response.json()
    assert grid_payload["total"] == 1
    assert grid_payload["records"][0]["year"] == 2023
    assert grid_payload["records"][0]["beneficiary_country"] == "Kenya"
    assert grid_payload["records"][0]["project_title"] == "Forest"

    assert filter_response.status_code == HTTPStatus.OK
    assert filter_response.json() == {
        "year": [2023, 2022],
        "project_title": ["Forest", "Solar"],
        "beneficiary_country": ["Brazil", "Kenya"],
        "funding_provider": ["Provider Alpha"],
        "source": ["OECD"],
        "source_url": ["https://oecd.org"],
        "financial_instrument": [],
        "sector": [],
        "sub_sector": [],
    }


@pytest.mark.asyncio
async def test_read_admin_record_filter_suggestions(
    client,
    token,
    seeded_records,
):
    response = client.get(
        "/records/admin/filter-suggestions",
        params={"column": "project_title", "search": "So", "limit": 10},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "column": "project_title",
        "values": ["Solar"],
        "limit": 10,
        "offset": 0,
        "has_more": False,
    }


@pytest.mark.asyncio
async def test_read_admin_record_filter_suggestions_respects_active_filters(
    client,
    token,
    seeded_records,
):
    response = client.get(
        "/records/admin/filter-suggestions",
        params={
            "column": "project_title",
            "beneficiary_country": ["Brazil"],
        },
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["values"] == ["Solar"]


@pytest.mark.asyncio
async def test_read_admin_record_filter_suggestions_invalid_column(
    client,
    token,
):
    response = client.get(
        "/records/admin/filter-suggestions",
        params={"column": "invalid_field"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["detail"] == "Invalid admin filter column"
