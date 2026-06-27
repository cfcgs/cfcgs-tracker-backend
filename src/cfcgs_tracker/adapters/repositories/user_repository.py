from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.domain.models.user import User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, user: User) -> None:
        self.session.add(user)

    async def get_by_id(self, user_id: int) -> User | None:
        return await self.session.scalar(
            select(User).where(User.id == user_id)
        )

    async def get_by_username(self, username: str) -> User | None:
        statement = select(User).where(User.username == username)
        return await self.session.scalar(statement)

    async def get_by_email(self, email: str) -> User | None:
        statement = select(User).where(User.email == email)
        return await self.session.scalar(statement)

    async def get_by_username_or_email(
        self,
        username: str,
        email: str,
    ) -> User | None:
        statement = select(User).where(
            (User.username == username) | (User.email == email)
        )
        return await self.session.scalar(statement)

    async def list_paginated(self, limit: int, offset: int) -> list[User]:
        statement = select(User).limit(limit).offset(offset)
        result = await self.session.scalars(statement)
        return list(result.all())

    async def delete(self, user: User) -> None:
        await self.session.delete(user)
