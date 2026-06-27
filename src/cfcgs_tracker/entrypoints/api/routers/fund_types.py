from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends

from src.cfcgs_tracker.entrypoints.api.dependencies import (
    get_reference_read_service,
)
from src.cfcgs_tracker.entrypoints.api.schemas.reference_data import (
    FundTypeList,
)
from src.cfcgs_tracker.service_layer.services.reference_reads import (
    ReferenceReadService,
)

router = APIRouter(prefix="/fund_types", tags=["fund_types"])
ReferenceReadServiceDep = Annotated[
    ReferenceReadService,
    Depends(get_reference_read_service),
]


@router.get("/", response_model=FundTypeList, status_code=HTTPStatus.OK)
async def read_fund_types(service: ReferenceReadServiceDep):
    fund_types = await service.get_fund_types()
    return {"fund_types": fund_types}
