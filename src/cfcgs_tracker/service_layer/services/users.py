from src.cfcgs_tracker.domain.exceptions import (
    EmailAlreadyExistsError,
    InvalidCurrentPasswordError,
    PasswordConfirmationMismatchError,
    PermissionDeniedError,
    UserNotFoundError,
    UsernameAlreadyExistsError,
)
from src.cfcgs_tracker.domain.models import User, UserRole
from src.cfcgs_tracker.service_layer.security import (
    get_password_hash,
    verify_password,
)
from src.cfcgs_tracker.service_layer.unit_of_work import AbstractUnitOfWork


class UserService:
    def __init__(self, uow: AbstractUnitOfWork) -> None:
        self.uow = uow

    async def _get_existing_active_user(self, user_id: int) -> User:
        user_db = await self.uow.users.get_by_id(user_id)
        if not user_db or not user_db.is_active:
            raise UserNotFoundError()
        return user_db

    async def get_user(
        self,
        user_id: int,
        current_user: User,
    ) -> User | None:
        if current_user.role != UserRole.admin and current_user.id != user_id:
            raise PermissionDeniedError()

        return await self._get_existing_active_user(user_id)

    async def get_users(self, limit: int, offset: int) -> list[User]:
        return await self.uow.users.list_paginated(limit, offset)

    async def create_user(self, user: User) -> User | None:
        user_by_username = await self.uow.users.get_by_username(user.username)
        if user_by_username:
            raise UsernameAlreadyExistsError()

        user_by_email = await self.uow.users.get_by_email(user.email)
        if user_by_email:
            raise EmailAlreadyExistsError()

        user.password = get_password_hash(user.password)
        await self.uow.users.add(user)
        await self.uow.commit()
        await self.uow.refresh(user)

        return user

    async def update_user(
        self,
        user_id: int,
        username: str,
        email: str,
        current_password: str | None,
        new_password: str | None,
        new_password_confirmation: str | None,
        current_user: User,
    ) -> User | None:
        if current_user.id != user_id:
            raise PermissionDeniedError()

        user_db = await self._get_existing_active_user(user_id)

        user_with_same_username = await self.uow.users.get_by_username(
            username
        )
        if user_with_same_username and user_with_same_username.id != user_id:
            raise UsernameAlreadyExistsError()

        user_with_same_email = await self.uow.users.get_by_email(email)
        if user_with_same_email and user_with_same_email.id != user_id:
            raise EmailAlreadyExistsError()

        user_db.username = username
        user_db.email = email

        if any(
            value is not None
            for value in (
                current_password,
                new_password,
                new_password_confirmation,
            )
        ):
            if not current_password or not verify_password(
                current_password,
                user_db.password,
            ):
                raise InvalidCurrentPasswordError()

            if not new_password or new_password != new_password_confirmation:
                raise PasswordConfirmationMismatchError()

            user_db.password = get_password_hash(new_password)

        await self.uow.commit()
        await self.uow.refresh(user_db)

        return user_db

    async def update_user_role(
        self,
        *,
        user_id: int,
        role: UserRole,
        current_user: User,
    ) -> User:
        if current_user.role != UserRole.admin:
            raise PermissionDeniedError()

        user_db = await self._get_existing_active_user(user_id)
        user_db.role = role

        await self.uow.commit()
        await self.uow.refresh(user_db)

        return user_db

    async def verify_current_password(
        self,
        *,
        user_id: int,
        current_password: str,
        current_user: User,
    ) -> None:
        if current_user.id != user_id:
            raise PermissionDeniedError()

        user_db = await self._get_existing_active_user(user_id)

        if not verify_password(current_password, user_db.password):
            raise InvalidCurrentPasswordError()

    async def delete_user(self, user_id: int, current_user: User) -> None:
        if current_user.role != UserRole.admin and current_user.id != user_id:
            raise PermissionDeniedError()

        user_db = await self._get_existing_active_user(user_id)
        user_db.is_active = False

        await self.uow.commit()
