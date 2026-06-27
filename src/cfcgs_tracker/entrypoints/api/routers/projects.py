from enum import Enum
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from src.cfcgs_tracker.entrypoints.api.dependencies import get_record_service
from src.cfcgs_tracker.entrypoints.api.schemas.records import (
    PaginatedRecordProjectList,
)
from src.cfcgs_tracker.service_layer.services.records import RecordService


class ObjectiveFilter(str, Enum):
    all = "all"
    adaptation = "adaptation"
    mitigation = "mitigation"
    both = "both"


router = APIRouter(prefix="/projects", tags=["projects"])
RecordServiceDep = Annotated[RecordService, Depends(get_record_service)]


@router.get(
    "/records/paginated",
    response_model=PaginatedRecordProjectList,
    status_code=HTTPStatus.OK,
)
async def read_paginated_record_projects(
    service: RecordServiceDep,
    search: str | None = Query(None),
    years: list[int] | None = Query(None, alias="year"),
    country_ids: list[int] | None = Query(None, alias="country_id"),
    objective: ObjectiveFilter = Query(ObjectiveFilter.all),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    return await service.get_paginated_projects(
        search=search,
        years=years,
        country_ids=country_ids,
        objective=objective.value,
        limit=limit,
        offset=offset,
    )
