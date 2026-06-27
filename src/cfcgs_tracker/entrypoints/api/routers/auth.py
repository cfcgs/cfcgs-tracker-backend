from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from src.cfcgs_tracker.domain.exceptions import (
    InvalidCredentialsError,
    InvalidLoginError,
)
from src.cfcgs_tracker.domain.models import User
from src.cfcgs_tracker.entrypoints.api.dependencies import (
    get_auth_service,
    get_current_user,
)
from src.cfcgs_tracker.entrypoints.api.schemas.auth import (
    RefreshTokenRequest,
    Token,
)
from src.cfcgs_tracker.entrypoints.api.schemas.users import UserPublic
from src.cfcgs_tracker.service_layer.services.auth import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
OAuth2Form = Annotated[OAuth2PasswordRequestForm, Depends()]
CurrentUser = Annotated[User, Depends(get_current_user)]


@router.post("/token", response_model=Token)
async def login_for_access_token(
    service: AuthServiceDep,
    form_data: OAuth2Form,
):
    try:
        token_pair = await service.authenticate(
            username_or_email=form_data.username,
            password=form_data.password,
        )
    except InvalidLoginError as exc:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return {
        "access_token": token_pair["access_token"],
        "refresh_token": token_pair["refresh_token"],
        "token_type": "Bearer",
    }


@router.post("/refresh_token", response_model=Token)
async def refresh_access_token(
    payload: RefreshTokenRequest,
    service: AuthServiceDep,
):
    try:
        token_pair = await service.refresh_access_token(payload.refresh_token)
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return {
        "access_token": token_pair["access_token"],
        "refresh_token": token_pair["refresh_token"],
        "token_type": "Bearer",
    }


@router.get("/me", response_model=UserPublic, status_code=HTTPStatus.OK)
async def read_current_user(
    user: CurrentUser,
):
    return UserPublic.model_validate(user)
