from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from src.cfcgs_tracker.domain.exceptions import (
    EmailAlreadyExistsError,
    UsernameAlreadyExistsError,
    PermissionDeniedError,
    UserNotFoundError,
    InvalidCurrentPasswordError,
    PasswordConfirmationMismatchError,
)
from src.cfcgs_tracker.entrypoints.api.dependencies import (
    get_current_admin_user,
    get_current_user,
    get_user_service,
)
from src.cfcgs_tracker.domain.models.user import User
from src.cfcgs_tracker.entrypoints.api.schemas.users import (
    UserCreate,
    PasswordVerificationRequest,
    UserList,
    UserPublic,
    UserRoleUpdate,
    UserUpdate,
)

from src.cfcgs_tracker.entrypoints.api.schemas.common import (
    Message,
    FilterPage,
)
from src.cfcgs_tracker.service_layer.services.users import UserService

router = APIRouter(prefix="/users", tags=["users"])
UserServiceDep = Annotated[UserService, Depends(get_user_service)]
CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(get_current_admin_user)]


@router.get("/{user_id}", status_code=HTTPStatus.OK, response_model=UserPublic)
async def read_user(
    user_id: int,
    service: UserServiceDep,
    current_user: CurrentUser,
):
    try:
        user_db = await service.get_user(user_id, current_user)
    except PermissionDeniedError as exc:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail=str(exc),
        ) from exc
    except UserNotFoundError as exc:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=str(exc),
        ) from exc

    return UserPublic.model_validate(user_db)


@router.get("/", status_code=HTTPStatus.OK, response_model=UserList)
async def read_users(
    service: UserServiceDep,
    _current_user: AdminUser,
    filter_users: Annotated[FilterPage, Query()],
):
    users = await service.get_users(
        limit=filter_users.limit,
        offset=filter_users.offset,
    )
    return {"users": users}


@router.post("/", status_code=HTTPStatus.CREATED, response_model=UserPublic)
async def create_user(
    user: UserCreate,
    service: UserServiceDep,
    _current_user: AdminUser,
):
    try:
        db_user = await service.create_user(
            User(
                username=user.username,
                email=user.email,
                password=user.password,
                role=user.role,
            )
        )
    except UsernameAlreadyExistsError as exception:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail=str(exception),
        )
    except EmailAlreadyExistsError as exception:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail=str(exception),
        )

    return db_user


@router.put(
    "/{user_id}",
    status_code=HTTPStatus.OK,
    response_model=UserPublic,
)
async def update_user(
    user_id: int,
    user: UserUpdate,
    service: UserServiceDep,
    current_user: CurrentUser,
):
    try:
        updated_user = await service.update_user(
            user_id=user_id,
            username=user.username,
            email=user.email,
            current_password=user.current_password,
            new_password=user.new_password,
            new_password_confirmation=user.new_password_confirmation,
            current_user=current_user,
        )
    except PermissionDeniedError as exc:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail=str(exc),
        ) from exc
    except UserNotFoundError as exc:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=str(exc),
        ) from exc
    except UsernameAlreadyExistsError as exc:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail=str(exc),
        ) from exc
    except EmailAlreadyExistsError as exc:
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT,
            detail=str(exc),
        ) from exc
    except InvalidCurrentPasswordError as exc:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail=str(exc),
        ) from exc
    except PasswordConfirmationMismatchError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return UserPublic.model_validate(updated_user)


@router.post(
    "/{user_id}/verify-password",
    status_code=HTTPStatus.NO_CONTENT,
)
async def verify_user_current_password(
    user_id: int,
    payload: PasswordVerificationRequest,
    service: UserServiceDep,
    current_user: CurrentUser,
):
    try:
        await service.verify_current_password(
            user_id=user_id,
            current_password=payload.current_password,
            current_user=current_user,
        )
    except PermissionDeniedError as exc:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail=str(exc),
        ) from exc
    except UserNotFoundError as exc:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=str(exc),
        ) from exc
    except InvalidCurrentPasswordError as exc:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail=str(exc),
        ) from exc


@router.patch(
    "/{user_id}/role",
    status_code=HTTPStatus.OK,
    response_model=UserPublic,
)
async def update_user_role(
    user_id: int,
    user_role_update: UserRoleUpdate,
    service: UserServiceDep,
    current_user: CurrentUser,
):
    try:
        updated_user = await service.update_user_role(
            user_id=user_id,
            role=user_role_update.role,
            current_user=current_user,
        )
    except PermissionDeniedError as exc:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail=str(exc),
        ) from exc
    except UserNotFoundError as exc:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=str(exc),
        ) from exc

    return UserPublic.model_validate(updated_user)


@router.delete("/{user_id}", status_code=HTTPStatus.OK, response_model=Message)
async def delete_user(
    user_id: int,
    service: UserServiceDep,
    current_user: CurrentUser,
):
    try:
        await service.delete_user(user_id, current_user)
    except PermissionDeniedError as exc:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail=str(exc),
        ) from exc
    except UserNotFoundError as exc:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=str(exc),
        ) from exc

    return {"message": "User deactivated"}
