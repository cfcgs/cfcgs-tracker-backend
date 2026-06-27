from dataclasses import asdict

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.domain.models.user import User, UserRole


@pytest.mark.asyncio
async def test_create_user(session: AsyncSession, mock_db_time):
    with mock_db_time(model=User) as times:
        new_user = User(
            username="test",
            email="test@test",
            password="teste",
        )

        session.add(new_user)
        await session.commit()

        user = await session.scalar(
            select(User).where(User.username == "test")
        )

        assert asdict(user) == {
            "id": 1,
            "username": "test",
            "email": "test@test",
            "password": "teste",
            "role": UserRole.importer,
            "is_active": True,
            "import_jobs": [],
            "created_at": times[0],
            "updated_at": times[0],
        }


@pytest.mark.asyncio
async def test_update_user(session: AsyncSession, mock_db_time):
    with mock_db_time(model=User) as times:
        new_user = User(
            username="test",
            email="test@test",
            password="teste",
        )

        session.add(new_user)
        await session.commit()

        user_created = await session.scalar(
            select(User).where(User.username == "test")
        )

        assert asdict(user_created) == {
            "id": 1,
            "username": "test",
            "email": "test@test",
            "password": "teste",
            "role": UserRole.importer,
            "is_active": True,
            "import_jobs": [],
            "created_at": times[0],
            "updated_at": times[0],
        }

        user_created.username = "jordany"
        await session.commit()

        user_updated = await session.scalar(
            select(User).where(User.username == "jordany")
        )

        assert asdict(user_updated) == {
            "id": 1,
            "username": "jordany",
            "email": "test@test",
            "password": "teste",
            "role": UserRole.importer,
            "is_active": True,
            "import_jobs": [],
            "created_at": times[0],
            "updated_at": times[1],
        }
