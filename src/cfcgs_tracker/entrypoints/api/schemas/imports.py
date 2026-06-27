from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ImportJobPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_type: str
    file_name: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    rows_received: int | None
    rows_inserted: int | None
    rows_updated: int | None
    rows_duplicated: int | None
    rows_failed: int | None
    user_id: int
    user_email: str | None = None


class ImportJobList(BaseModel):
    import_jobs: list[ImportJobPublic]
