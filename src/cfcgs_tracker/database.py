from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.cfcgs_tracker.settings import Settings

engine = create_engine(Settings().DATABASE_URL)


def get_session() -> Session:  # pragma: no cover
    with Session(engine) as session:
        yield session