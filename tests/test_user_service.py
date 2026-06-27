import pytest

from src.cfcgs_tracker.domain.exceptions import UserNotFoundError
from src.cfcgs_tracker.domain.models import User, UserRole
from src.cfcgs_tracker.service_layer.services.users import UserService
from src.cfcgs_tracker.service_layer.unit_of_work import AbstractUnitOfWork


class FakeUserRepository:
    def __init__(self) -> None:
        self._users_by_id: dict[int, User] = {}

    async def get_by_id(self, user_id: int) -> User | None:
        return self._users_by_id.get(user_id)

    async def get_by_username(self, username: str) -> User | None:
        for user in self._users_by_id.values():
            if user.username == username:
                return user
        return None

    async def get_by_email(self, email: str) -> User | None:
        for user in self._users_by_id.values():
            if user.email == email:
                return user
        return None


class FakeUnitOfWork(AbstractUnitOfWork):
    def __init__(self) -> None:
        self.users = FakeUserRepository()

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None

    async def refresh(self, instance) -> None:
        return None

    async def flush(self) -> None:
        return None

    def begin_nested(self):
        return None

    async def execute(self, statement, params=None):
        return None


@pytest.mark.asyncio
async def test_update_user_raises_not_found_when_current_user_matches_id():
    uow = FakeUnitOfWork()
    service = UserService(uow)

    current_user = User(
        username="alice",
        email="alice@example.com",
        password="secret",
        role=UserRole.importer,
    )
    current_user.id = 1

    with pytest.raises(UserNotFoundError, match="User not found."):
        await service.update_user(
            user_id=1,
            username="alice",
            email="alice@example.com",
            current_password=None,
            new_password=None,
            new_password_confirmation=None,
            current_user=current_user,
        )
