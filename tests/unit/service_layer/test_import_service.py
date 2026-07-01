import pytest

from src.cfcgs_tracker.domain.exceptions import PermissionDeniedError
from src.cfcgs_tracker.domain.models import ImportJob, User, UserRole
from src.cfcgs_tracker.domain.models.import_job import ImportStatus
from src.cfcgs_tracker.service_layer.services.imports import ImportService
from src.cfcgs_tracker.service_layer.unit_of_work import AbstractUnitOfWork
from tests.conftest import (
    ClimateFinanceImportRowFactory,
    FundImportRowFactory,
    build_csv_file,
    build_xlsx_file,
)


class _ImportJobsRepo:
    def __init__(self) -> None:
        self.rows = []

    async def add(self, import_job: ImportJob):
        self.rows.append(import_job)

    async def get_by_id(self, import_job_id: int):
        return next((row for row in self.rows if row.id == import_job_id), None)

    async def get_by_id_and_user_id(self, import_job_id: int, user_id: int):
        return next(
            (
                row
                for row in self.rows
                if row.id == import_job_id and row.user_id == user_id
            ),
            None,
        )

    async def list_paginated(self, limit: int, offset: int):
        return self.rows[offset : offset + limit]

    async def list_paginated_by_user_id(self, user_id: int, limit: int, offset: int):
        rows = [row for row in self.rows if row.user_id == user_id]
        return rows[offset : offset + limit]


class _UsersRepo:
    def __init__(self) -> None:
        self.by_id = {}

    async def get_by_id(self, user_id: int):
        return self.by_id.get(user_id)


class FakeImportUoW(AbstractUnitOfWork):
    def __init__(self) -> None:
        self.import_jobs = _ImportJobsRepo()
        self.users = _UsersRepo()
        self.commit_calls = 0
        self.refresh_calls = 0
        self.rollback_calls = 0
        self._next_id = 1

    async def commit(self) -> None:
        self.commit_calls += 1

    async def rollback(self) -> None:
        self.rollback_calls += 1

    async def refresh(self, instance) -> None:
        self.refresh_calls += 1
        if getattr(instance, "id", None) is None:
            instance.id = self._next_id
            self._next_id += 1

    async def flush(self) -> None:
        return None

    def begin_nested(self):
        return None

    async def execute(self, statement, params=None):
        return None


class FakeBatchImporter:
    def __init__(self, responses=None, error: Exception | None = None) -> None:
        self.responses = list(responses or [(0, 0, 0, 0)])
        self.error = error
        self.calls = []

    async def import_rows(self, rows):
        self.calls.append(rows)
        if self.error:
            raise self.error
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


def _build_user(user_id: int, role: UserRole = UserRole.importer) -> User:
    user = User(
        username=f"user-{user_id}",
        email=f"user-{user_id}@example.com",
        password="secret",
        role=role,
    )
    user.id = user_id
    return user


def test_parse_file_parses_workbook_rows():
    service = ImportService(FakeImportUoW())
    _, file_bytes, _ = build_xlsx_file(
        climate_rows=ClimateFinanceImportRowFactory.build_batch(2),
        fund_rows=FundImportRowFactory.build_batch(1),
    )

    payload = service.parse_file(
        file_name="records.xlsx",
        file_bytes=file_bytes,
    )

    assert len(payload.climate_finance_rows) == 2
    assert len(payload.fund_rows) == 1
    assert payload.rows_received == 3
    assert payload.rows_failed == 0


def test_parse_file_rejects_unknown_csv_headers():
    service = ImportService(FakeImportUoW())

    with pytest.raises(
        ValueError,
        match="Unsupported CSV headers",
    ):
        service.parse_file(
            file_name="invalid.csv",
            file_bytes=b"foo,bar\n1,2\n",
        )


@pytest.mark.asyncio
async def test_get_import_job_and_list_respect_user_scope():
    uow = FakeImportUoW()
    admin = _build_user(1, UserRole.admin)
    importer = _build_user(2, UserRole.importer)
    another_importer = _build_user(3, UserRole.importer)
    uow.users.by_id = {
        1: admin,
        2: importer,
        3: another_importer,
    }

    first_job = ImportJob(
        file_type="csv",
        file_name="a.csv",
        status=ImportStatus.success,
        user_id=2,
        rows_received=1,
        rows_inserted=1,
        rows_updated=0,
        rows_duplicated=0,
        rows_failed=0,
    )
    first_job.id = 1
    second_job = ImportJob(
        file_type="csv",
        file_name="b.csv",
        status=ImportStatus.success,
        user_id=3,
        rows_received=1,
        rows_inserted=1,
        rows_updated=0,
        rows_duplicated=0,
        rows_failed=0,
    )
    second_job.id = 2
    uow.import_jobs.rows.extend([first_job, second_job])
    service = ImportService(uow)

    admin_jobs = await service.get_import_jobs(admin, limit=10, offset=0)
    importer_jobs = await service.get_import_jobs(
        importer,
        limit=10,
        offset=0,
    )
    own_job = await service.get_import_job(1, importer)

    assert len(admin_jobs) == 2
    assert len(importer_jobs) == 1
    assert own_job["user_email"] == importer.email

    with pytest.raises(PermissionDeniedError):
        await service.get_import_job(2, importer)


@pytest.mark.asyncio
async def test_import_file_processes_climate_finance_csv_successfully():
    uow = FakeImportUoW()
    current_user = _build_user(1, UserRole.importer)
    uow.users.by_id[current_user.id] = current_user
    service = ImportService(uow)
    service.climate_finance_importer = FakeBatchImporter(
        responses=[(2, 0, 1, 0)]
    )

    _, file_bytes, _ = build_csv_file(
        file_kind="climate_finance",
        climate_rows=ClimateFinanceImportRowFactory.build_batch(2),
    )

    payload = await service.import_file(
        file_name="climate.csv",
        file_bytes=file_bytes,
        current_user=current_user,
    )

    assert payload["status"] == ImportStatus.success
    assert payload["rows_received"] == 2
    assert payload["rows_inserted"] == 2
    assert payload["rows_updated"] == 0
    assert payload["rows_duplicated"] == 1
    assert payload["rows_failed"] == 0
    assert payload["user_email"] == current_user.email


@pytest.mark.asyncio
async def test_import_file_marks_job_as_error_when_processing_fails():
    uow = FakeImportUoW()
    current_user = _build_user(1, UserRole.importer)
    uow.users.by_id[current_user.id] = current_user
    service = ImportService(uow)
    service.fund_status_importer = FakeBatchImporter(error=ValueError("boom"))

    _, file_bytes, _ = build_csv_file(
        file_kind="fund",
        fund_rows=FundImportRowFactory.build_batch(1),
    )

    with pytest.raises(ValueError, match="boom"):
        await service.import_file(
            file_name="funds.csv",
            file_bytes=file_bytes,
            current_user=current_user,
        )

    saved_job = uow.import_jobs.rows[0]
    assert saved_job.status == ImportStatus.error
    assert saved_job.finished_at is not None
    assert uow.commit_calls >= 2
