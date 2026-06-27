from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from src.cfcgs_tracker.domain.base import table_registry


class FileType(str, Enum):
    csv = "csv"
    xlsx = "xlsx"


class ImportStatus(str, Enum):
    success = "success"
    processing = "processing"
    error = "error"


@table_registry.mapped_as_dataclass
class ImportJob:
    __tablename__ = "import_jobs"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    file_type: Mapped[FileType]
    file_name: Mapped[str]
    status: Mapped[ImportStatus]
    started_at: Mapped[datetime] = mapped_column(
        init=False, server_default=func.now()
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True, default=None
    )
    rows_inserted: Mapped[Optional[int]] = mapped_column(
        nullable=True, default=None
    )
    rows_received: Mapped[Optional[int]] = mapped_column(
        nullable=True, default=None
    )
    rows_updated: Mapped[Optional[int]] = mapped_column(
        nullable=True, default=None
    )
    rows_duplicated: Mapped[Optional[int]] = mapped_column(
        nullable=True, default=None
    )
    rows_failed: Mapped[Optional[int]] = mapped_column(
        nullable=True, default=None
    )
