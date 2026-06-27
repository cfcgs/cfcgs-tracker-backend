from datetime import datetime
from enum import Enum

from sqlalchemy import Enum as SqlEnum
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.cfcgs_tracker.domain.base import table_registry
from src.cfcgs_tracker.domain.models.import_job import ImportJob


class UserRole(str, Enum):
    admin = "admin"
    importer = "importer"


@table_registry.mapped_as_dataclass
class User:
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
    email: Mapped[str] = mapped_column(unique=True)
    password: Mapped[str]
    role: Mapped[UserRole] = mapped_column(
        SqlEnum(UserRole, name="user_role"),
        default=UserRole.importer,
        server_default=UserRole.importer.value,
    )
    is_active: Mapped[bool] = mapped_column(
        default=True,
        server_default="true",
    )
    created_at: Mapped[datetime] = mapped_column(
        init=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        init=False, server_default=func.now(), onupdate=func.now()
    )

    import_jobs: Mapped[list[ImportJob]] = relationship(
        init=False, cascade="all, delete-orphan", lazy="selectin"
    )
