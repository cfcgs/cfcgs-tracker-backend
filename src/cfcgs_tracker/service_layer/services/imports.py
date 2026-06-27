from datetime import datetime
from itertools import chain
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from src.cfcgs_tracker.domain.exceptions import PermissionDeniedError
from src.cfcgs_tracker.domain.models import ImportJob, User
from src.cfcgs_tracker.domain.models.user import UserRole
from src.cfcgs_tracker.domain.models.import_job import FileType, ImportStatus
from src.cfcgs_tracker.adapters.csv_importer import iter_csv_row_batches
from src.cfcgs_tracker.adapters.xlsx_importer import iter_sheet_row_batches
from src.cfcgs_tracker.service_layer.import_schemas import (
    ClimateFinanceRowSchema,
    FundRowSchema,
    WorkbookImportSchema,
)
from src.cfcgs_tracker.service_layer.services.climate_finance_records import (
    ClimateFinanceRecordImportService,
)
from src.cfcgs_tracker.service_layer.services.provider_fund_profiles import (
    FundStatusImportService,
)
from src.cfcgs_tracker.service_layer.unit_of_work import AbstractUnitOfWork


class ImportService:
    CLIMATE_FINANCE_SHEET_INDEX = 3
    FUND_STATUS_SHEET_INDEX = 4
    BATCH_SIZE = 500
    CLIMATE_FINANCE_REQUIRED_HEADERS = {
        "year",
        "funding_provider",
        "source",
    }
    FUND_REQUIRED_HEADERS = {
        "fund",
        "fund_type",
        "fund_focus",
    }

    def __init__(self, uow: AbstractUnitOfWork) -> None:
        self.uow = uow
        self.climate_finance_importer = ClimateFinanceRecordImportService(uow)
        self.fund_status_importer = FundStatusImportService(uow)

    async def _serialize_import_job(self, import_job: ImportJob) -> dict:
        user = await self.uow.users.get_by_id(import_job.user_id)
        return {
            "id": import_job.id,
            "file_type": import_job.file_type,
            "file_name": import_job.file_name,
            "status": import_job.status,
            "started_at": import_job.started_at,
            "finished_at": import_job.finished_at,
            "rows_received": import_job.rows_received,
            "rows_inserted": import_job.rows_inserted,
            "rows_updated": import_job.rows_updated,
            "rows_duplicated": import_job.rows_duplicated,
            "rows_failed": import_job.rows_failed,
            "user_id": import_job.user_id,
            "user_email": user.email if user else None,
        }

    async def get_import_job(
        self,
        import_job_id: int,
        current_user: User,
    ) -> dict | None:
        if current_user.role == UserRole.admin:
            import_job = await self.uow.import_jobs.get_by_id(import_job_id)
            if not import_job:
                return None
            return await self._serialize_import_job(import_job)

        import_job = await self.uow.import_jobs.get_by_id_and_user_id(
            import_job_id,
            current_user.id,
        )
        if not import_job:
            raise PermissionDeniedError()
        return await self._serialize_import_job(import_job)

    async def get_import_jobs(
        self,
        current_user: User,
        limit: int,
        offset: int,
    ) -> list[dict]:
        if current_user.role == UserRole.admin:
            import_jobs = await self.uow.import_jobs.list_paginated(
                limit,
                offset,
            )
            return [
                await self._serialize_import_job(import_job)
                for import_job in import_jobs
            ]

        import_jobs = await self.uow.import_jobs.list_paginated_by_user_id(
            current_user.id,
            limit,
            offset,
        )
        return [
            await self._serialize_import_job(import_job)
            for import_job in import_jobs
        ]

    def _current_app_time(self) -> datetime:
        return datetime.now(tz=ZoneInfo("America/Belem")).replace(tzinfo=None)

    def _validate_climate_finance_batch(
        self,
        batch: list[dict],
        *,
        batch_number: int,
        rows_received: int,
        rows_failed: int,
    ) -> tuple[list[ClimateFinanceRowSchema], int, int]:
        validated_batch: list[ClimateFinanceRowSchema] = []

        for row in batch:
            rows_received += 1
            try:
                validated_batch.append(
                    ClimateFinanceRowSchema.model_validate(row)
                )
            except ValidationError as exc:
                rows_failed += 1
                print(
                    "[imports] invalid climate finance row "
                    f"(batch={batch_number}, row={rows_received}): "
                    f"{exc.errors()}"
                )

        return validated_batch, rows_received, rows_failed

    def _validate_fund_batch(
        self,
        batch: list[dict],
        *,
        batch_number: int,
        rows_received: int,
        rows_failed: int,
    ) -> tuple[list[FundRowSchema], int, int]:
        validated_batch: list[FundRowSchema] = []

        for row in batch:
            rows_received += 1
            try:
                validated_batch.append(FundRowSchema.model_validate(row))
            except ValidationError as exc:
                rows_failed += 1
                print(
                    "[imports] invalid fund status row "
                    f"(batch={batch_number}, row={rows_received}): "
                    f"{exc.errors()}"
                )

        return validated_batch, rows_received, rows_failed

    async def _persist_progress(self, import_job: ImportJob) -> None:
        await self.uow.commit()
        await self.uow.refresh(import_job)

    async def _process_climate_finance_batches(
        self,
        *,
        batches,
        import_job: ImportJob,
        rows_received_total: int,
        rows_failed_total: int,
        rows_inserted_total: int,
        rows_updated_total: int,
        rows_duplicated_total: int,
    ) -> tuple[int, int, int, int, int]:
        rows_received = 0
        rows_failed = 0
        rows_inserted = 0
        rows_updated = 0

        for batch_number, batch in enumerate(batches, start=1):
            previous_failed = rows_failed
            validated_batch, rows_received, rows_failed = (
                self._validate_climate_finance_batch(
                    batch,
                    batch_number=batch_number,
                    rows_received=rows_received,
                    rows_failed=rows_failed,
                )
            )
            parse_failed = rows_failed - previous_failed
            if not validated_batch:
                rows_received_total += len(batch)
                rows_failed_total += parse_failed
                import_job.rows_received = rows_received_total
                import_job.rows_failed = rows_failed_total
                import_job.rows_inserted = rows_inserted_total
                import_job.rows_updated = rows_updated_total
                import_job.rows_duplicated = rows_duplicated_total
                await self._persist_progress(import_job)
                continue

            (
                inserted,
                updated,
                duplicated,
                persist_failed,
            ) = await self.climate_finance_importer.import_rows(
                validated_batch
            )
            rows_inserted += inserted
            rows_updated += updated
            rows_duplicated_total += duplicated
            rows_failed += persist_failed

            rows_received_total += len(batch)
            rows_inserted_total += inserted
            rows_updated_total += updated
            rows_failed_total += parse_failed + persist_failed

            import_job.rows_received = rows_received_total
            import_job.rows_inserted = rows_inserted_total
            import_job.rows_updated = rows_updated_total
            import_job.rows_duplicated = rows_duplicated_total
            import_job.rows_failed = rows_failed_total
            await self._persist_progress(import_job)
            print(
                "[imports] climate finance persisted batch "
                f"{batch_number}: inserted={inserted}, updated={updated}, "
                f"duplicated={duplicated}, failed={persist_failed}"
            )

        return (
            rows_received_total,
            rows_failed_total,
            rows_inserted_total,
            rows_updated_total,
            rows_duplicated_total,
        )

    async def _process_fund_batches(
        self,
        *,
        batches,
        import_job: ImportJob,
        rows_received_total: int,
        rows_failed_total: int,
        rows_inserted_total: int,
        rows_updated_total: int,
        rows_duplicated_total: int,
    ) -> tuple[int, int, int, int, int]:
        rows_received = 0
        rows_failed = 0
        rows_inserted = 0
        rows_updated = 0

        for batch_number, batch in enumerate(batches, start=1):
            previous_failed = rows_failed
            validated_batch, rows_received, rows_failed = (
                self._validate_fund_batch(
                    batch,
                    batch_number=batch_number,
                    rows_received=rows_received,
                    rows_failed=rows_failed,
                )
            )
            parse_failed = rows_failed - previous_failed
            if not validated_batch:
                rows_received_total += len(batch)
                rows_failed_total += parse_failed
                import_job.rows_received = rows_received_total
                import_job.rows_failed = rows_failed_total
                import_job.rows_inserted = rows_inserted_total
                import_job.rows_updated = rows_updated_total
                import_job.rows_duplicated = rows_duplicated_total
                await self._persist_progress(import_job)
                continue

            (
                inserted,
                updated,
                duplicated,
                persist_failed,
            ) = await self.fund_status_importer.import_rows(validated_batch)
            rows_inserted += inserted
            rows_updated += updated
            rows_duplicated_total += duplicated
            rows_failed += persist_failed

            rows_received_total += len(batch)
            rows_inserted_total += inserted
            rows_updated_total += updated
            rows_failed_total += parse_failed + persist_failed

            import_job.rows_received = rows_received_total
            import_job.rows_inserted = rows_inserted_total
            import_job.rows_updated = rows_updated_total
            import_job.rows_duplicated = rows_duplicated_total
            import_job.rows_failed = rows_failed_total
            await self._persist_progress(import_job)
            print(
                "[imports] fund status persisted batch "
                f"{batch_number}: inserted={inserted}, updated={updated}, "
                f"duplicated={duplicated}, failed={persist_failed}"
            )

        return (
            rows_received_total,
            rows_failed_total,
            rows_inserted_total,
            rows_updated_total,
            rows_duplicated_total,
        )

    def parse_file(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
    ) -> WorkbookImportSchema:
        file_type = self._get_file_type(file_name)

        if file_type == FileType.xlsx:
            return self._parse_workbook(file_bytes)

        if file_type == FileType.csv:
            return self._parse_csv(file_bytes)

        raise ValueError("Unsupported file type. Use .csv or .xlsx.")

    def _get_file_type(self, file_name: str) -> FileType:
        suffix = Path(file_name).suffix.lower()

        if suffix == ".xlsx":
            return FileType.xlsx

        if suffix == ".csv":
            return FileType.csv

        raise ValueError("Unsupported file type. Use .csv or .xlsx.")

    def _parse_workbook(self, file_bytes: bytes) -> WorkbookImportSchema:
        climate_finance_rows, climate_rows_received, climate_rows_failed = (
            self._parse_climate_finance_sheet(file_bytes)
        )
        fund_rows, fund_rows_received, fund_rows_failed = (
            self._parse_fund_status_sheet(file_bytes)
        )

        return WorkbookImportSchema(
            climate_finance_rows=climate_finance_rows,
            fund_rows=fund_rows,
            rows_received=climate_rows_received + fund_rows_received,
            rows_failed=climate_rows_failed + fund_rows_failed,
        )

    def _parse_csv(self, file_bytes: bytes) -> WorkbookImportSchema:
        first_batch = next(
            iter_csv_row_batches(
                file_bytes,
                batch_size=self.BATCH_SIZE,
            ),
            None,
        )
        if not first_batch:
            return WorkbookImportSchema(
                climate_finance_rows=[],
                fund_rows=[],
            )

        headers = set(first_batch[0].keys())

        if self.CLIMATE_FINANCE_REQUIRED_HEADERS.issubset(headers):
            (
                climate_finance_rows,
                climate_rows_received,
                climate_rows_failed,
            ) = self._parse_climate_finance_csv(file_bytes)
            return WorkbookImportSchema(
                climate_finance_rows=climate_finance_rows,
                fund_rows=[],
                rows_received=climate_rows_received,
                rows_failed=climate_rows_failed,
            )

        if self.FUND_REQUIRED_HEADERS.issubset(headers):
            fund_rows, fund_rows_received, fund_rows_failed = (
                self._parse_fund_status_csv(file_bytes)
            )
            return WorkbookImportSchema(
                climate_finance_rows=[],
                fund_rows=fund_rows,
                rows_received=fund_rows_received,
                rows_failed=fund_rows_failed,
            )

        raise ValueError(
            "Unsupported CSV headers. Could not identify climate finance "
            "or fund status layout."
        )

    def _parse_climate_finance_sheet(
        self,
        file_bytes: bytes,
    ) -> tuple[list[ClimateFinanceRowSchema], int, int]:
        parsed_rows: list[ClimateFinanceRowSchema] = []
        rows_received = 0
        rows_failed = 0

        for batch_number, batch in enumerate(
            iter_sheet_row_batches(
                file_bytes,
                sheet_index=self.CLIMATE_FINANCE_SHEET_INDEX,
                batch_size=self.BATCH_SIZE,
            ),
            start=1,
        ):
            validated_batch: list[ClimateFinanceRowSchema] = []
            for row in batch:
                rows_received += 1
                try:
                    validated_batch.append(
                        ClimateFinanceRowSchema.model_validate(row)
                    )
                except ValidationError as exc:
                    rows_failed += 1
                    print(
                        "[imports] invalid climate finance row "
                        f"(batch={batch_number}, row={rows_received}): "
                        f"{exc.errors()}"
                    )
            parsed_rows.extend(validated_batch)
            print(
                "[imports] climate finance sheet batch "
                f"{batch_number}: +{len(validated_batch)} rows "
                f"(received={rows_received}, failed={rows_failed})"
            )

        print(
            "[imports] climate finance sheet completed "
            f"with received={rows_received}, valid={len(parsed_rows)}, "
            f"failed={rows_failed}"
        )
        return parsed_rows, rows_received, rows_failed

    def _parse_fund_status_sheet(
        self,
        file_bytes: bytes,
    ) -> tuple[list[FundRowSchema], int, int]:
        parsed_rows: list[FundRowSchema] = []
        rows_received = 0
        rows_failed = 0

        for batch_number, batch in enumerate(
            iter_sheet_row_batches(
                file_bytes,
                sheet_index=self.FUND_STATUS_SHEET_INDEX,
                batch_size=self.BATCH_SIZE,
            ),
            start=1,
        ):
            validated_batch: list[FundRowSchema] = []
            for row in batch:
                rows_received += 1
                try:
                    validated_batch.append(FundRowSchema.model_validate(row))
                except ValidationError as exc:
                    rows_failed += 1
                    print(
                        "[imports] invalid fund status row "
                        f"(batch={batch_number}, row={rows_received}): "
                        f"{exc.errors()}"
                    )
            parsed_rows.extend(validated_batch)
            print(
                "[imports] fund status sheet batch "
                f"{batch_number}: +{len(validated_batch)} rows "
                f"(received={rows_received}, failed={rows_failed})"
            )

        print(
            "[imports] fund status sheet completed "
            f"with received={rows_received}, valid={len(parsed_rows)}, "
            f"failed={rows_failed}"
        )
        return parsed_rows, rows_received, rows_failed

    def _parse_climate_finance_csv(
        self,
        file_bytes: bytes,
    ) -> tuple[list[ClimateFinanceRowSchema], int, int]:
        parsed_rows: list[ClimateFinanceRowSchema] = []
        rows_received = 0
        rows_failed = 0

        for batch_number, batch in enumerate(
            iter_csv_row_batches(
                file_bytes,
                batch_size=self.BATCH_SIZE,
            ),
            start=1,
        ):
            validated_batch: list[ClimateFinanceRowSchema] = []
            for row in batch:
                rows_received += 1
                try:
                    validated_batch.append(
                        ClimateFinanceRowSchema.model_validate(row)
                    )
                except ValidationError as exc:
                    rows_failed += 1
                    print(
                        "[imports] invalid climate finance csv row "
                        f"(batch={batch_number}, row={rows_received}): "
                        f"{exc.errors()}"
                    )
            parsed_rows.extend(validated_batch)
            print(
                "[imports] climate finance csv batch "
                f"{batch_number}: +{len(validated_batch)} rows "
                f"(received={rows_received}, failed={rows_failed})"
            )

        print(
            "[imports] climate finance csv completed with "
            f"received={rows_received}, valid={len(parsed_rows)}, "
            f"failed={rows_failed}"
        )
        return parsed_rows, rows_received, rows_failed

    def _parse_fund_status_csv(
        self,
        file_bytes: bytes,
    ) -> tuple[list[FundRowSchema], int, int]:
        parsed_rows: list[FundRowSchema] = []
        rows_received = 0
        rows_failed = 0

        for batch_number, batch in enumerate(
            iter_csv_row_batches(
                file_bytes,
                batch_size=self.BATCH_SIZE,
            ),
            start=1,
        ):
            validated_batch: list[FundRowSchema] = []
            for row in batch:
                rows_received += 1
                try:
                    validated_batch.append(FundRowSchema.model_validate(row))
                except ValidationError as exc:
                    rows_failed += 1
                    print(
                        "[imports] invalid fund status csv row "
                        f"(batch={batch_number}, row={rows_received}): "
                        f"{exc.errors()}"
                    )
            parsed_rows.extend(validated_batch)
            print(
                "[imports] fund status csv batch "
                f"{batch_number}: +{len(validated_batch)} rows "
                f"(received={rows_received}, failed={rows_failed})"
            )

        print(
            "[imports] fund status csv completed with "
            f"received={rows_received}, valid={len(parsed_rows)}, "
            f"failed={rows_failed}"
        )
        return parsed_rows, rows_received, rows_failed

    async def import_file(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        current_user: User,
    ) -> dict:
        import_job = ImportJob(
            file_type=self._get_file_type(file_name),
            file_name=file_name,
            status=ImportStatus.processing,
            user_id=current_user.id,
            rows_received=0,
            rows_inserted=0,
            rows_updated=0,
            rows_duplicated=0,
            rows_failed=0,
        )
        import_job.started_at = self._current_app_time()
        await self.uow.import_jobs.add(import_job)
        await self.uow.commit()
        await self.uow.refresh(import_job)

        try:
            rows_received_total = 0
            rows_failed_total = 0
            rows_inserted_total = 0
            rows_updated_total = 0
            rows_duplicated_total = 0

            file_type = self._get_file_type(file_name)
            if file_type == FileType.xlsx:
                (
                    rows_received_total,
                    rows_failed_total,
                    rows_inserted_total,
                    rows_updated_total,
                    rows_duplicated_total,
                ) = await self._process_climate_finance_batches(
                    batches=iter_sheet_row_batches(
                        file_bytes,
                        sheet_index=self.CLIMATE_FINANCE_SHEET_INDEX,
                        batch_size=self.BATCH_SIZE,
                    ),
                    import_job=import_job,
                    rows_received_total=rows_received_total,
                    rows_failed_total=rows_failed_total,
                    rows_inserted_total=rows_inserted_total,
                    rows_updated_total=rows_updated_total,
                    rows_duplicated_total=rows_duplicated_total,
                )
                (
                    rows_received_total,
                    rows_failed_total,
                    rows_inserted_total,
                    rows_updated_total,
                    rows_duplicated_total,
                ) = await self._process_fund_batches(
                    batches=iter_sheet_row_batches(
                        file_bytes,
                        sheet_index=self.FUND_STATUS_SHEET_INDEX,
                        batch_size=self.BATCH_SIZE,
                    ),
                    import_job=import_job,
                    rows_received_total=rows_received_total,
                    rows_failed_total=rows_failed_total,
                    rows_inserted_total=rows_inserted_total,
                    rows_updated_total=rows_updated_total,
                    rows_duplicated_total=rows_duplicated_total,
                )
            elif file_type == FileType.csv:
                iterator = iter_csv_row_batches(
                    file_bytes,
                    batch_size=self.BATCH_SIZE,
                )
                first_batch = next(iterator, None)
                if first_batch:
                    headers = set(first_batch[0].keys())
                    if self.CLIMATE_FINANCE_REQUIRED_HEADERS.issubset(headers):
                        (
                            rows_received_total,
                            rows_failed_total,
                            rows_inserted_total,
                            rows_updated_total,
                            rows_duplicated_total,
                        ) = await self._process_climate_finance_batches(
                            batches=chain([first_batch], iterator),
                            import_job=import_job,
                            rows_received_total=rows_received_total,
                            rows_failed_total=rows_failed_total,
                            rows_inserted_total=rows_inserted_total,
                            rows_updated_total=rows_updated_total,
                            rows_duplicated_total=rows_duplicated_total,
                        )
                    elif self.FUND_REQUIRED_HEADERS.issubset(headers):
                        (
                            rows_received_total,
                            rows_failed_total,
                            rows_inserted_total,
                            rows_updated_total,
                            rows_duplicated_total,
                        ) = await self._process_fund_batches(
                            batches=chain([first_batch], iterator),
                            import_job=import_job,
                            rows_received_total=rows_received_total,
                            rows_failed_total=rows_failed_total,
                            rows_inserted_total=rows_inserted_total,
                            rows_updated_total=rows_updated_total,
                            rows_duplicated_total=rows_duplicated_total,
                        )
                    else:
                        raise ValueError(
                            "Unsupported CSV headers. Could not identify "
                            "climate finance or fund status layout."
                        )
            else:
                raise ValueError("Unsupported file type. Use .csv or .xlsx.")

            import_job.status = ImportStatus.success
            import_job.finished_at = self._current_app_time()
            import_job.rows_duplicated = rows_duplicated_total
            await self.uow.commit()
            await self.uow.refresh(import_job)
            return await self._serialize_import_job(import_job)
        except Exception:
            import_job.status = ImportStatus.error
            import_job.finished_at = self._current_app_time()
            await self.uow.commit()
            raise
