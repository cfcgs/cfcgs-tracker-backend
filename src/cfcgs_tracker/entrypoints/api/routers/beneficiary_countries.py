from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends

from src.cfcgs_tracker.entrypoints.api.dependencies import (
    get_reference_read_service,
)
from src.cfcgs_tracker.entrypoints.api.schemas.reference_data import (
    CountryList,
)
from src.cfcgs_tracker.service_layer.services.reference_reads import (
    ReferenceReadService,
)

router = APIRouter(
    prefix="/beneficiary_countries",
    tags=["beneficiary_countries"],
)
ReferenceReadServiceDep = Annotated[
    ReferenceReadService,
    Depends(get_reference_read_service),
]


@router.get("/", response_model=CountryList, status_code=HTTPStatus.OK)
async def read_beneficiary_countries(service: ReferenceReadServiceDep):
    countries = await service.get_beneficiary_countries()
    return {"countries": countries}
