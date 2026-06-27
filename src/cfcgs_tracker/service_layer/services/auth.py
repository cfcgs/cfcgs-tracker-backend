from jwt import DecodeError, ExpiredSignatureError, decode

from src.cfcgs_tracker.domain.exceptions import (
    InvalidCredentialsError,
    InvalidLoginError,
)
from src.cfcgs_tracker.domain.models.user import User
from src.cfcgs_tracker.service_layer.security import (
    create_access_token,
    create_refresh_token,
    verify_password,
)
from src.cfcgs_tracker.service_layer.unit_of_work import AbstractUnitOfWork
from src.cfcgs_tracker.settings import Settings


class AuthService:
    def __init__(self, uow: AbstractUnitOfWork) -> None:
        self.uow = uow

    async def get_current_user(self, token: str) -> User:
        payload = self._decode_token(token, expected_token_type="access")
        return await self._get_user_by_subject(payload["sub"])

    async def refresh_access_token(self, refresh_token: str) -> dict[str, str]:
        payload = self._decode_token(
            refresh_token,
            expected_token_type="refresh",
        )
        user = await self._get_user_by_subject(payload["sub"])
        return self._build_token_pair(user)

    def _decode_token(
        self,
        token: str,
        *,
        expected_token_type: str,
    ) -> dict:
        try:
            payload = decode(
                token,
                Settings().SECRET_KEY,
                algorithms=[Settings().ALGORITHM],
            )
            subject_email = payload["sub"]
            token_type = payload.get("token_type")
            if not subject_email or token_type != expected_token_type:
                raise InvalidCredentialsError()
        except DecodeError as exc:
            raise InvalidCredentialsError() from exc
        except ExpiredSignatureError as exc:
            raise InvalidCredentialsError() from exc

        return payload

    async def _get_user_by_subject(self, subject_email: str) -> User:
        user = await self.uow.users.get_by_email(subject_email)
        if not user or not user.is_active:
            raise InvalidCredentialsError()

        return user

    async def authenticate(
        self,
        username_or_email: str,
        password: str,
    ) -> dict[str, str]:
        user = await self.uow.users.get_by_email(username_or_email)

        if not user:
            user = await self.uow.users.get_by_username(username_or_email)

        if not user:
            raise InvalidLoginError()

        if not user.is_active:
            raise InvalidLoginError()

        if not verify_password(password, user.password):
            raise InvalidLoginError()

        return self._build_token_pair(user)

    def _build_token_pair(self, user: User) -> dict[str, str]:
        return {
            "access_token": create_access_token({"sub": user.email}),
            "refresh_token": create_refresh_token({"sub": user.email}),
        }
