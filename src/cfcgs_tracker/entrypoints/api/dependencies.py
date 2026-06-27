from http import HTTPStatus
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

from src.cfcgs_tracker.adapters.orm import get_session
from src.cfcgs_tracker.domain.exceptions import InvalidCredentialsError
from src.cfcgs_tracker.domain.models.user import User, UserRole
from src.cfcgs_tracker.service_layer.services.auth import AuthService
from src.cfcgs_tracker.service_layer.services.chatbot import ChatbotService
from src.cfcgs_tracker.service_layer.services.funding_providers import (
    FundingProviderService,
)
from src.cfcgs_tracker.service_layer.services.imports import ImportService
from src.cfcgs_tracker.service_layer.services.records import RecordService
from src.cfcgs_tracker.service_layer.services.reference_reads import (
    ReferenceReadService,
)
from src.cfcgs_tracker.service_layer.services.users import UserService
from src.cfcgs_tracker.service_layer.unit_of_work import (
    AbstractUnitOfWork,
    SqlAlchemyUnitOfWork,
)
from sqlalchemy.ext.asyncio import AsyncSession

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


async def get_uow(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AbstractUnitOfWork:
    async with SqlAlchemyUnitOfWork(session) as uow:
        yield uow


def get_user_service(
    uow: Annotated[AbstractUnitOfWork, Depends(get_uow)],
) -> UserService:
    return UserService(uow)


def get_auth_service(
    uow: Annotated[AbstractUnitOfWork, Depends(get_uow)],
) -> AuthService:
    return AuthService(uow)


def get_chatbot_service(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ChatbotService:
    return ChatbotService(session)


def get_import_service(
    uow: Annotated[AbstractUnitOfWork, Depends(get_uow)],
) -> ImportService:
    return ImportService(uow)


def get_reference_read_service(
    uow: Annotated[AbstractUnitOfWork, Depends(get_uow)],
) -> ReferenceReadService:
    return ReferenceReadService(uow)


def get_record_service(
    uow: Annotated[AbstractUnitOfWork, Depends(get_uow)],
) -> RecordService:
    return RecordService(uow)


def get_funding_provider_service(
    uow: Annotated[AbstractUnitOfWork, Depends(get_uow)],
) -> FundingProviderService:
    return FundingProviderService(uow)


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> User:
    try:
        return await auth_service.get_current_user(token)
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def _ensure_allowed_roles(
    current_user: User,
    *,
    allowed_roles: set[UserRole],
) -> User:
    if current_user.role not in allowed_roles:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user


def get_current_admin_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    return _ensure_allowed_roles(
        current_user,
        allowed_roles={UserRole.admin},
    )


def get_current_import_operator(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    return _ensure_allowed_roles(
        current_user,
        allowed_roles={UserRole.admin, UserRole.importer},
    )
