from enum import Enum
from http import HTTPStatus
from io import StringIO
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.cfcgs_tracker.domain.models import User
from src.cfcgs_tracker.entrypoints.api.dependencies import (
    get_current_import_operator,
    get_record_service,
)
from src.cfcgs_tracker.entrypoints.api.schemas.records import (
    AdminRecordFilterOptionsPublic,
    AdminRecordFilterSuggestionsPublic,
    AdminRecordList,
    AdminRecordPublic,
    AdminRecordUpdate,
    RecordCountryYearGridPublic,
    RecordCountryYearProjectList,
    RecordFilterOptionsPublic,
    RecordObjectiveTotalResponse,
    RecordSearchFilter,
    RecordSearchList,
    RecordSummaryResponse,
    RecordTimeSeriesResponse,
)
from src.cfcgs_tracker.service_layer.services.records import RecordService


class ObjectiveFilter(str, Enum):
    all = "all"
    adaptation = "adaptation"
    mitigation = "mitigation"
    both = "both"


class HeatmapView(str, Enum):
    country_year = "country_year"
    year_country = "year_country"


router = APIRouter(prefix="/records", tags=["records"])
RecordServiceDep = Annotated[RecordService, Depends(get_record_service)]
ImportOperator = Annotated[User, Depends(get_current_import_operator)]


@router.get("/years", response_model=list[int], status_code=HTTPStatus.OK)
async def read_record_years(service: RecordServiceDep):
    return await service.get_years()


@router.post(
    "/search", response_model=RecordSearchList, status_code=HTTPStatus.OK
)
async def search_records(
    filters: RecordSearchFilter,
    service: RecordServiceDep,
    limit: int = 40,
    offset: int = 0,
):
    records = await service.search_records(
        years=filters.years,
        country_ids=filters.countries,
        limit=limit,
        offset=offset,
    )
    return {"records": records}


@router.get("/export/{year}", response_class=StreamingResponse)
async def export_records_by_year(
    year: int,
    service: RecordServiceDep,
):
    rows = await service.export_records_by_year(year)
    output = StringIO()
    output.write(
        "ID,Year,Project,Funding Provider,Recipient Country,Source,Amount (USD millions)\n"
    )
    for row in rows:
        output.write(
            f'{row["id"]},{row["year"]},"{row["project"] or ""}","{row["funding_provider"] or ""}","{row["recipient_country"] or ""}","{row["source"] or ""}",{row["amount_usd_millions"] or 0}\n'
        )
    output.seek(0)
    headers = {
        "Content-Disposition": f'attachment; filename="records_{year}.csv"'
    }
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers=headers,
    )


@router.get(
    "/summary", response_model=RecordSummaryResponse, status_code=HTTPStatus.OK
)
async def read_record_summary(
    service: RecordServiceDep,
    years: list[int] | None = Query(None, alias="year"),
    country_ids: list[int] | None = Query(None, alias="country_id"),
    project_ids: list[int] | None = Query(None, alias="project_id"),
    objective: ObjectiveFilter = Query(ObjectiveFilter.all),
):
    return await service.get_summary(
        years=years,
        country_ids=country_ids,
        project_ids=project_ids,
        objective=objective.value,
    )


@router.get(
    "/aggregations/by-objective",
    response_model=RecordObjectiveTotalResponse,
    status_code=HTTPStatus.OK,
)
async def read_record_totals_by_objective(
    service: RecordServiceDep,
    years: list[int] | None = Query(None, alias="year"),
    country_ids: list[int] | None = Query(None, alias="country_id"),
):
    return await service.get_objective_aggregation(
        years=years,
        country_ids=country_ids,
    )


@router.get(
    "/aggregations/by-year",
    response_model=RecordTimeSeriesResponse,
    status_code=HTTPStatus.OK,
)
async def read_record_time_series(
    service: RecordServiceDep,
    years: list[int] | None = Query(None, alias="year"),
    country_ids: list[int] | None = Query(None, alias="country_id"),
):
    return await service.get_year_aggregation(
        years=years,
        country_ids=country_ids,
    )


@router.get(
    "/filter-options",
    response_model=RecordFilterOptionsPublic,
    status_code=HTTPStatus.OK,
)
async def read_record_filter_options(
    service: RecordServiceDep,
    years: list[int] | None = Query(None, alias="year"),
    country_ids: list[int] | None = Query(None, alias="country_id"),
    project_ids: list[int] | None = Query(None, alias="project_id"),
    objective: ObjectiveFilter = Query(ObjectiveFilter.all),
):
    return await service.get_filter_options(
        years=years,
        country_ids=country_ids,
        project_ids=project_ids,
        objective=objective.value,
    )


@router.get(
    "/aggregations/by-country-and-year",
    response_model=RecordCountryYearGridPublic,
    status_code=HTTPStatus.OK,
)
async def read_record_country_year_grid(
    service: RecordServiceDep,
    years: list[int] | None = Query(None, alias="year"),
    country_ids: list[int] | None = Query(None, alias="country_id"),
    project_ids: list[int] | None = Query(None, alias="project_id"),
    objective: ObjectiveFilter = Query(ObjectiveFilter.all),
    view: HeatmapView = Query(HeatmapView.country_year),
    row_offset: int = Query(0, ge=0),
    row_limit: int = Query(30, ge=1, le=200),
    column_offset: int = Query(0, ge=0),
    column_limit: int = Query(30, ge=1, le=200),
):
    return await service.get_country_year_grid(
        years=years,
        country_ids=country_ids,
        project_ids=project_ids,
        objective=objective.value,
        view=view.value,
        row_offset=row_offset,
        row_limit=row_limit,
        column_offset=column_offset,
        column_limit=column_limit,
    )


@router.get(
    "/projects/by-country-and-year",
    response_model=RecordCountryYearProjectList,
    status_code=HTTPStatus.OK,
)
async def read_record_projects_by_country_and_year(
    service: RecordServiceDep,
    year: int = Query(...),
    country_id: int = Query(...),
    project_ids: list[int] | None = Query(None, alias="project_id"),
    objective: ObjectiveFilter = Query(ObjectiveFilter.all),
    limit: int = Query(30, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    return await service.get_projects_by_country_and_year(
        year=year,
        country_id=country_id,
        project_ids=project_ids,
        objective=objective.value,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/admin/grid",
    response_model=AdminRecordList,
    status_code=HTTPStatus.OK,
)
async def read_admin_record_grid(
    service: RecordServiceDep,
    _current_user: ImportOperator,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: str | None = Query(None),
    sort_by: str = Query("year"),
    sort_order: str = Query("desc"),
    years: list[int] | None = Query(None, alias="year"),
    project_titles: list[str] | None = Query(None, alias="project_title"),
    beneficiary_countries: list[str] | None = Query(
        None,
        alias="beneficiary_country",
    ),
    funding_providers: list[str] | None = Query(
        None,
        alias="funding_provider",
    ),
    sources: list[str] | None = Query(None, alias="source"),
    source_urls: list[str] | None = Query(None, alias="source_url"),
    financial_instruments: list[str] | None = Query(
        None,
        alias="financial_instrument",
    ),
    sectors: list[str] | None = Query(None, alias="sector"),
    sub_sectors: list[str] | None = Query(None, alias="sub_sector"),
):
    filters = {
        "year": [str(year) for year in years or []],
        "project_title": project_titles or [],
        "beneficiary_country": beneficiary_countries or [],
        "funding_provider": funding_providers or [],
        "source": sources or [],
        "source_url": source_urls or [],
        "financial_instrument": financial_instruments or [],
        "sector": sectors or [],
        "sub_sector": sub_sectors or [],
    }
    return await service.get_admin_records(
        limit=limit,
        offset=offset,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
        filters=filters,
    )


@router.get(
    "/admin/filter-options",
    response_model=AdminRecordFilterOptionsPublic,
    status_code=HTTPStatus.OK,
)
async def read_admin_record_filter_options(
    service: RecordServiceDep,
    _current_user: ImportOperator,
    search: str | None = Query(None),
):
    return await service.get_admin_record_filter_options(search=search)


@router.get(
    "/admin/filter-suggestions",
    response_model=AdminRecordFilterSuggestionsPublic,
    status_code=HTTPStatus.OK,
)
async def read_admin_record_filter_suggestions(
    service: RecordServiceDep,
    _current_user: ImportOperator,
    column: str = Query(...),
    search: str | None = Query(None),
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
    years: list[int] | None = Query(None, alias="year"),
    project_titles: list[str] | None = Query(None, alias="project_title"),
    beneficiary_countries: list[str] | None = Query(
        None,
        alias="beneficiary_country",
    ),
    funding_providers: list[str] | None = Query(
        None,
        alias="funding_provider",
    ),
    sources: list[str] | None = Query(None, alias="source"),
    source_urls: list[str] | None = Query(None, alias="source_url"),
    financial_instruments: list[str] | None = Query(
        None,
        alias="financial_instrument",
    ),
    sectors: list[str] | None = Query(None, alias="sector"),
    sub_sectors: list[str] | None = Query(None, alias="sub_sector"),
):
    filters = {
        "year": [str(year) for year in years or []],
        "project_title": project_titles or [],
        "beneficiary_country": beneficiary_countries or [],
        "funding_provider": funding_providers or [],
        "source": sources or [],
        "source_url": source_urls or [],
        "financial_instrument": financial_instruments or [],
        "sector": sectors or [],
        "sub_sector": sub_sectors or [],
    }
    try:
        return await service.get_admin_record_filter_suggestions(
            column=column,
            search=search,
            limit=limit,
            offset=offset,
            filters=filters,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.patch(
    "/{record_id}",
    response_model=AdminRecordPublic,
    status_code=HTTPStatus.OK,
)
async def update_admin_record(
    record_id: int,
    payload: AdminRecordUpdate,
    service: RecordServiceDep,
    _current_user: ImportOperator,
):
    try:
        return await service.update_admin_record(
            record_id=record_id,
            payload=payload.model_dump(exclude_unset=True),
        )
    except ValueError as exc:
        detail = str(exc)
        status_code = (
            HTTPStatus.NOT_FOUND
            if detail == "Record not found"
            else HTTPStatus.BAD_REQUEST
        )
        raise HTTPException(status_code=status_code, detail=detail) from exc
