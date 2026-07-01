import pytest

from src.cfcgs_tracker.adapters.repositories.project_repository import (
    ProjectRepository,
)
from tests.conftest import (
    BeneficiaryCountryFactory,
    ClimateFinanceRecordFactory,
    ProjectFactory,
    SourceFactory,
)


@pytest.mark.asyncio
async def test_project_repository_filters_projects_for_records(session):
    country_a = BeneficiaryCountryFactory(name="Brazil")
    country_b = BeneficiaryCountryFactory(name="Chile")
    project_a = ProjectFactory(title="Alpha Project")
    project_b = ProjectFactory(title="Beta Project")
    source = SourceFactory(name="OECD", url="https://oecd.org")
    session.add_all([country_a, country_b, project_a, project_b, source])
    await session.flush()

    session.add_all(
        [
            ClimateFinanceRecordFactory(
                year=2024,
                project_id=project_a.id,
                beneficiary_country_id=country_a.id,
                source_id=source.id,
                adaptation_amount_usd_millions=10.0,
                mitigation_amount_usd_millions=0.0,
                both_objectives_amount_usd_millions=0.0,
            ),
            ClimateFinanceRecordFactory(
                year=2023,
                project_id=project_b.id,
                beneficiary_country_id=country_b.id,
                source_id=source.id,
                adaptation_amount_usd_millions=0.0,
                mitigation_amount_usd_millions=20.0,
                both_objectives_amount_usd_millions=0.0,
            ),
        ]
    )
    await session.commit()

    repository = ProjectRepository(session)

    result = await repository.list_paginated_for_records(
        search="Alpha",
        years=[2024],
        country_ids=[country_a.id],
        objective="adaptation",
        limit=10,
        offset=0,
    )

    assert result["total"] == 1
    assert result["has_more"] is False
    assert result["projects"] == [
        {"id": project_a.id, "name": "Alpha Project"}
    ]
