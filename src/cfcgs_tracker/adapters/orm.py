from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.cfcgs_tracker.settings import Settings

engine = create_async_engine(Settings().DATABASE_URL)
session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncSession:  # pragma: no cover
    async with session_factory() as session:
        yield session
