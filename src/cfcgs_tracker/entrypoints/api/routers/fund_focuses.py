from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends

from src.cfcgs_tracker.entrypoints.api.dependencies import (
    get_reference_read_service,
)
from src.cfcgs_tracker.entrypoints.api.schemas.reference_data import (
    FundFocusList,
)
from src.cfcgs_tracker.service_layer.services.reference_reads import (
    ReferenceReadService,
)

router = APIRouter(prefix="/fund_focuses", tags=["fund_focuses"])
ReferenceReadServiceDep = Annotated[
    ReferenceReadService,
    Depends(get_reference_read_service),
]


@router.get("/", response_model=FundFocusList, status_code=HTTPStatus.OK)
async def read_fund_focuses(service: ReferenceReadServiceDep):
    fund_focuses = await service.get_fund_focuses()
    return {"fund_focuses": fund_focuses}
