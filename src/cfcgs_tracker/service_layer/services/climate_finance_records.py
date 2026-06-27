from collections.abc import Iterable
import hashlib
import json

from sqlalchemy import bindparam, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import (
    DBAPIError,
    InterfaceError,
    OperationalError,
    PendingRollbackError,
)

from src.cfcgs_tracker.domain.models import (
    BeneficiaryCountry,
    ClimateFinanceRecord,
    FinancialInstrument,
    FundingProvider,
    Project,
    Sector,
    Source,
    SubSector,
)
from src.cfcgs_tracker.service_layer.import_schemas import (
    ClimateFinanceRowSchema,
)
from src.cfcgs_tracker.service_layer.unit_of_work import AbstractUnitOfWork


class ClimateFinanceRecordImportService:
    def __init__(self, uow: AbstractUnitOfWork) -> None:
        self.uow = uow

    def _log_climate_action(
        self,
        *,
        action: str,
        row_hash: str,
        target_hash: str | None = None,
        record_id: int | None = None,
    ) -> None:
        print(
            "[imports] climate finance action "
            f"(action={action}, record_id={record_id}, "
            f"source_row_hash={row_hash}, target_hash={target_hash})"
        )

    async def _fetch_name_id_map(
        self,
        model,
        field_name: str,
        values: set[str],
    ) -> dict[str, int]:
        if not values:
            return {}

        column = getattr(model, field_name)
        result = await self.uow.execute(
            select(model.id, column).where(column.in_(values))
        )
        return {value: item_id for item_id, value in result.all()}

    async def _ensure_named_dimension(
        self,
        model,
        field_name: str,
        values: Iterable[str | None],
    ) -> dict[str, int]:
        normalized_values = {
            value for value in values if value not in (None, "")
        }
        if not normalized_values:
            return {}

        existing = await self._fetch_name_id_map(
            model,
            field_name,
            normalized_values,
        )
        missing = normalized_values - set(existing)

        if missing:
            column = getattr(model.__table__.c, field_name)
            statement = insert(model.__table__).values(
                [{field_name: value} for value in missing]
            )
            statement = statement.on_conflict_do_nothing(
                index_elements=[column.name]
            )
            await self.uow.execute(statement)
            existing = await self._fetch_name_id_map(
                model,
                field_name,
                normalized_values,
            )

        return existing

    async def _ensure_sources(
        self,
        rows: list[ClimateFinanceRowSchema],
    ) -> dict[tuple[str, str], int]:
        source_pairs = {(row.source, row.source_url) for row in rows}
        if not source_pairs:
            return {}

        source_names = {name for name, _ in source_pairs}
        statement = insert(Source.__table__).values(
            [{"name": name, "url": url} for name, url in source_pairs]
        )
        statement = statement.on_conflict_do_update(
            index_elements=[Source.__table__.c.name],
            set_={"url": statement.excluded.url},
        )
        await self.uow.execute(statement)

        result = await self.uow.execute(
            select(Source.id, Source.name, Source.url).where(
                Source.name.in_(source_names)
            )
        )
        return {(name, url): item_id for item_id, name, url in result.all()}

    async def _ensure_sub_sectors(
        self,
        rows: list[ClimateFinanceRowSchema],
        sector_ids: dict[str, int],
    ) -> dict[tuple[str, int], int]:
        pairs = {
            (row.sub_sector, sector_ids[row.sector])
            for row in rows
            if row.sub_sector and row.sector and row.sector in sector_ids
        }
        if not pairs:
            return {}

        statement = insert(SubSector.__table__).values(
            [
                {"name": name, "sector_id": sector_id}
                for name, sector_id in pairs
            ]
        )
        statement = statement.on_conflict_do_nothing(
            index_elements=["name", "sector_id"]
        )
        await self.uow.execute(statement)

        sector_id_values = {sector_id for _, sector_id in pairs}
        sub_sector_names = {name for name, _ in pairs}
        result = await self.uow.execute(
            select(SubSector.id, SubSector.name, SubSector.sector_id).where(
                SubSector.name.in_(sub_sector_names),
                SubSector.sector_id.in_(sector_id_values),
            )
        )
        return {
            (name, sector_id): item_id
            for item_id, name, sector_id in result.all()
        }

    async def _load_reference_maps(
        self,
        rows: list[ClimateFinanceRowSchema],
    ) -> dict[str, dict]:
        sector_ids = await self._ensure_named_dimension(
            Sector,
            "name",
            [row.sector for row in rows],
        )
        return {
            "projects": await self._ensure_named_dimension(
                Project,
                "title",
                [row.project_title for row in rows],
            ),
            "providers": await self._ensure_named_dimension(
                FundingProvider,
                "name",
                [row.funding_provider for row in rows],
            ),
            "instruments": await self._ensure_named_dimension(
                FinancialInstrument,
                "name",
                [row.financial_instrument for row in rows],
            ),
            "sectors": sector_ids,
            "countries": await self._ensure_named_dimension(
                BeneficiaryCountry,
                "name",
                [row.beneficiary_country for row in rows],
            ),
            "sources": await self._ensure_sources(rows),
            "sub_sectors": await self._ensure_sub_sectors(rows, sector_ids),
        }

    def _build_source_row_hash(self, row: ClimateFinanceRowSchema) -> str:
        canonical_row = {
            "year": row.year,
            "project_title": row.project_title,
            "beneficiary_country": row.beneficiary_country,
            "provider_type": row.provider_type,
            "funding_provider": row.funding_provider,
            "approved_amount_usd_millions": row.approved_amount_usd_millions,
            "climate_finance_amount_usd_millions": (
                row.climate_finance_amount_usd_millions
            ),
            "adaptation_amount_usd_millions": (
                row.adaptation_climate_finance_amount_usd_millions
            ),
            "mitigation_amount_usd_millions": (
                row.mitigation_climate_finance_amount_usd_millions
            ),
            "both_objectives_amount_usd_millions": (
                row.both_climate_objectives_finance_amount_usd_millions
            ),
            "sector": row.sector,
            "sub_sector": row.sub_sector,
            "financial_instrument": row.financial_instrument,
            "source": row.source,
            "source_url": row.source_url,
        }
        serialized = json.dumps(
            canonical_row,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _build_record_payload(
        self,
        *,
        row: ClimateFinanceRowSchema,
        reference_ids: dict[str, dict],
    ) -> dict:
        sector_id = (
            reference_ids["sectors"].get(row.sector) if row.sector else None
        )
        sub_sector_id = (
            reference_ids["sub_sectors"].get((row.sub_sector, sector_id))
            if row.sub_sector and sector_id
            else None
        )
        return {
            "year": row.year,
            "source_row_hash": self._build_source_row_hash(row),
            "project_id": (
                reference_ids["projects"].get(row.project_title)
                if row.project_title
                else None
            ),
            "funding_provider_id": (
                reference_ids["providers"].get(row.funding_provider)
                if row.funding_provider
                else None
            ),
            "source_id": reference_ids["sources"][
                (row.source, row.source_url)
            ],
            "financial_instrument_id": (
                reference_ids["instruments"].get(row.financial_instrument)
                if row.financial_instrument
                else None
            ),
            "sector_id": sector_id,
            "sub_sector_id": sub_sector_id,
            "beneficiary_country_id": (
                reference_ids["countries"].get(row.beneficiary_country)
                if row.beneficiary_country
                else None
            ),
            "approved_amount_usd_millions": row.approved_amount_usd_millions,
            "climate_finance_amount_usd_millions": (
                row.climate_finance_amount_usd_millions
            ),
            "adaptation_amount_usd_millions": (
                row.adaptation_climate_finance_amount_usd_millions
            ),
            "mitigation_amount_usd_millions": (
                row.mitigation_climate_finance_amount_usd_millions
            ),
            "both_objectives_amount_usd_millions": (
                row.both_climate_objectives_finance_amount_usd_millions
            ),
            "hash_updated": row.hash_updated,
        }

    async def _fetch_existing_records_by_hash(
        self,
        hashes: set[str],
    ) -> dict[str, int]:
        if not hashes:
            return {}

        result = await self.uow.execute(
            select(
                ClimateFinanceRecord.id,
                ClimateFinanceRecord.source_row_hash,
            ).where(ClimateFinanceRecord.source_row_hash.in_(hashes))
        )
        return {row_hash: record_id for record_id, row_hash in result.all()}

    async def _bulk_persist(
        self,
        payloads: list[dict],
    ) -> tuple[int, int, int]:
        unique_payloads: dict[tuple[str | None, str], dict] = {}
        duplicate_rows = 0
        for payload in payloads:
            dedupe_key = (
                payload.get("hash_updated"),
                payload["source_row_hash"],
            )
            if dedupe_key in unique_payloads:
                duplicate_rows += 1
                self._log_climate_action(
                    action="duplicate_hash_in_same_batch",
                    row_hash=payload["source_row_hash"],
                    target_hash=payload.get("hash_updated"),
                )
            unique_payloads[dedupe_key] = payload

        normalized_payloads = list(unique_payloads.values())
        source_hashes = {
            payload["source_row_hash"] for payload in normalized_payloads
        }
        update_hashes = {
            payload["hash_updated"]
            for payload in normalized_payloads
            if payload.get("hash_updated")
        }
        existing_by_source_hash = await self._fetch_existing_records_by_hash(
            source_hashes
        )
        existing_by_update_hash = await self._fetch_existing_records_by_hash(
            update_hashes
        )

        inserts: list[dict] = []
        updates: list[dict] = []

        for payload in normalized_payloads:
            row_hash = payload["source_row_hash"]
            update_hash = payload.pop("hash_updated", None)

            if update_hash:
                target_id = existing_by_update_hash.get(update_hash)
                if target_id is None:
                    raise ValueError(
                        "reported hash_updated was not found in the database."
                    )

                conflicting_id = existing_by_source_hash.get(row_hash)
                if conflicting_id is not None and conflicting_id != target_id:
                    duplicate_rows += 1
                    self._log_climate_action(
                        action="duplicate_hash_conflict_on_update",
                        row_hash=row_hash,
                        target_hash=update_hash,
                        record_id=conflicting_id,
                    )
                    continue

                self._log_climate_action(
                    action="update_by_hash",
                    row_hash=row_hash,
                    target_hash=update_hash,
                    record_id=target_id,
                )
                updates.append({"record_id": target_id, **payload})
                continue

            if row_hash in existing_by_source_hash:
                duplicate_rows += 1
                self._log_climate_action(
                    action="duplicate_hash_in_database",
                    row_hash=row_hash,
                    record_id=existing_by_source_hash[row_hash],
                )
                continue

            inserts.append(payload)

        if inserts:
            await self.uow.execute(
                insert(ClimateFinanceRecord.__table__), inserts
            )

        if updates:
            await self.uow.execute(
                update(ClimateFinanceRecord.__table__)
                .where(
                    ClimateFinanceRecord.__table__.c.id
                    == bindparam("record_id")
                )
                .values(
                    year=bindparam("year"),
                    source_row_hash=bindparam("source_row_hash"),
                    project_id=bindparam("project_id"),
                    funding_provider_id=bindparam("funding_provider_id"),
                    source_id=bindparam("source_id"),
                    financial_instrument_id=bindparam(
                        "financial_instrument_id"
                    ),
                    sector_id=bindparam("sector_id"),
                    sub_sector_id=bindparam("sub_sector_id"),
                    beneficiary_country_id=bindparam("beneficiary_country_id"),
                    approved_amount_usd_millions=bindparam(
                        "approved_amount_usd_millions"
                    ),
                    climate_finance_amount_usd_millions=bindparam(
                        "climate_finance_amount_usd_millions"
                    ),
                    adaptation_amount_usd_millions=bindparam(
                        "adaptation_amount_usd_millions"
                    ),
                    mitigation_amount_usd_millions=bindparam(
                        "mitigation_amount_usd_millions"
                    ),
                    both_objectives_amount_usd_millions=bindparam(
                        "both_objectives_amount_usd_millions"
                    ),
                ),
                updates,
            )

        return len(inserts), len(updates), duplicate_rows

    async def _handle_fatal_error(self, exc: Exception) -> None:
        if isinstance(
            exc,
            (PendingRollbackError, InterfaceError, OperationalError),
        ):
            await self.uow.rollback()
            raise exc

        if isinstance(exc, DBAPIError) and exc.connection_invalidated:
            await self.uow.rollback()
            raise exc

    async def _persist_single_row(
        self,
        row: ClimateFinanceRowSchema,
    ) -> tuple[int, int, int]:
        reference_ids = await self._load_reference_maps([row])
        payload = self._build_record_payload(
            row=row,
            reference_ids=reference_ids,
        )
        return await self._bulk_persist([payload])

    async def import_rows(
        self,
        rows: list[ClimateFinanceRowSchema],
    ) -> tuple[int, int, int, int]:
        if not rows:
            return 0, 0, 0, 0

        inserted = 0
        updated = 0
        duplicated = 0
        failed = 0

        try:
            reference_ids = await self._load_reference_maps(rows)
            payloads = [
                self._build_record_payload(
                    row=row,
                    reference_ids=reference_ids,
                )
                for row in rows
            ]
            inserted, updated, duplicated = await self._bulk_persist(payloads)
            return inserted, updated, duplicated, failed
        except Exception as exc:
            await self._handle_fatal_error(exc)
            await self.uow.rollback()
            print(
                "[imports] climate finance batch failed, "
                "falling back to row-by-row persistence: "
                f"{exc}"
            )

        for row_number, row in enumerate(rows, start=1):
            try:
                (
                    batch_inserted,
                    batch_updated,
                    batch_duplicated,
                ) = await self._persist_single_row(row)
                inserted += batch_inserted
                updated += batch_updated
                duplicated += batch_duplicated
            except Exception as exc:
                await self._handle_fatal_error(exc)
                await self.uow.rollback()
                failed += 1
                print(
                    "[imports] failed to persist climate finance row "
                    f"(row={row_number}): {exc}"
                )

        return inserted, updated, duplicated, failed
