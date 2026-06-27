from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.cfcgs_tracker.entrypoints.api.dependencies import (
    get_current_import_operator,
    get_funding_provider_service,
)
from src.cfcgs_tracker.domain.exceptions import PermissionDeniedError
from src.cfcgs_tracker.domain.models import User
from src.cfcgs_tracker.entrypoints.api.schemas.funding_providers import (
    FundingProviderList,
    FundingProviderProfileUpdate,
    FundingProviderReadFilter,
    FundingProviderPublic,
    FundingProviderSummaryFilter,
    FundingProviderSummaryPublic,
)
from src.cfcgs_tracker.service_layer.services.funding_providers import (
    FundingProviderService,
)

router = APIRouter(
    prefix="/funding_providers",
    tags=["funding_providers"],
)
FundingProviderServiceDep = Annotated[
    FundingProviderService,
    Depends(get_funding_provider_service),
]
ImportOperator = Annotated[User, Depends(get_current_import_operator)]


@router.post(
    "/", response_model=FundingProviderList, status_code=HTTPStatus.OK
)
async def read_funding_providers(
    filters: FundingProviderReadFilter,
    service: FundingProviderServiceDep,
    limit: int = 40,
    offset: int = 0,
):
    return await service.get_funding_providers(
        fund_type_ids=filters.fund_types,
        fund_focus_ids=filters.fund_focuses,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/summary",
    response_model=FundingProviderSummaryPublic,
    status_code=HTTPStatus.OK,
)
async def read_funding_provider_summary(
    filters: FundingProviderSummaryFilter,
    service: FundingProviderServiceDep,
):
    return await service.get_funding_provider_summary(
        funding_provider_ids=filters.funding_providers,
        fund_type_ids=filters.fund_types,
        fund_focus_ids=filters.fund_focuses,
    )


@router.patch(
    "/{funding_provider_id}/profile",
    response_model=FundingProviderPublic,
    status_code=HTTPStatus.OK,
)
async def update_funding_provider_profile(
    funding_provider_id: int,
    payload: FundingProviderProfileUpdate,
    service: FundingProviderServiceDep,
    current_user: ImportOperator,
):
    try:
        return await service.update_funding_provider_profile(
            funding_provider_id=funding_provider_id,
            payload=payload.model_dump(exclude_unset=True),
            current_user=current_user,
        )
    except PermissionDeniedError as exc:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        detail = str(exc)
        status_code = (
            HTTPStatus.NOT_FOUND
            if detail
            in {
                "Funding provider not found",
                "Funding provider profile not found",
            }
            else HTTPStatus.BAD_REQUEST
        )
        raise HTTPException(status_code=status_code, detail=detail) from exc
