from sqlalchemy import bindparam, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import (
    DBAPIError,
    InterfaceError,
    OperationalError,
    PendingRollbackError,
)

from src.cfcgs_tracker.domain.models import (
    FundFocus,
    FundType,
    FundingProvider,
    ProviderFundProfile,
)
from src.cfcgs_tracker.service_layer.import_schemas import FundRowSchema
from src.cfcgs_tracker.service_layer.unit_of_work import AbstractUnitOfWork


class FundStatusImportService:
    def __init__(self, uow: AbstractUnitOfWork) -> None:
        self.uow = uow

    def _log_fund_action(
        self,
        *,
        action: str,
        funding_provider_id: int,
        fund_name: str,
    ) -> None:
        print(
            "[imports] fund profile action "
            f"(action={action}, funding_provider_id={funding_provider_id}, "
            f"fund={fund_name})"
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
        values: list[str | None],
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
            statement = insert(model.__table__).values(
                [{field_name: value} for value in missing]
            )
            statement = statement.on_conflict_do_nothing(
                index_elements=[field_name]
            )
            await self.uow.execute(statement)
            existing = await self._fetch_name_id_map(
                model,
                field_name,
                normalized_values,
            )

        return existing

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

    async def _bulk_persist(
        self,
        rows: list[FundRowSchema],
    ) -> tuple[int, int, int]:
        unique_rows_by_fund: dict[str, FundRowSchema] = {}
        duplicated = 0
        for row in rows:
            if row.fund in unique_rows_by_fund:
                duplicated += 1
                print(
                    "[imports] fund profile action "
                    f"(action=duplicate_fund_in_same_batch, fund={row.fund})"
                )
            unique_rows_by_fund[row.fund] = row

        rows = list(unique_rows_by_fund.values())

        provider_ids = await self._ensure_named_dimension(
            FundingProvider,
            "name",
            [row.fund for row in rows],
        )
        fund_type_ids = await self._ensure_named_dimension(
            FundType,
            "name",
            [row.fund_type for row in rows],
        )
        fund_focus_ids = await self._ensure_named_dimension(
            FundFocus,
            "name",
            [row.fund_focus for row in rows],
        )

        payloads = []
        for row in rows:
            provider_id = provider_ids.get(row.fund)
            if provider_id is None:
                raise ValueError("Funding provider could not be resolved.")

            payloads.append(
                {
                    "b_funding_provider_id": provider_id,
                    "funding_provider_id": provider_id,
                    "fund_type_id": (
                        fund_type_ids.get(row.fund_type)
                        if row.fund_type
                        else None
                    ),
                    "fund_focus_id": (
                        fund_focus_ids.get(row.fund_focus)
                        if row.fund_focus
                        else None
                    ),
                    "pledge": row.pledge_usd_mn,
                    "deposit": row.deposit_usd_mn,
                    "approval": row.approval_usd_mn,
                    "disbursement": row.disbursement_usd_mn,
                    "projects_approved": row.number_of_projects_approved,
                }
            )

        provider_id_values = {
            payload["funding_provider_id"] for payload in payloads
        }
        result = await self.uow.execute(
            select(
                ProviderFundProfile.funding_provider_id,
                ProviderFundProfile.fund_type_id,
                ProviderFundProfile.fund_focus_id,
                ProviderFundProfile.pledge,
                ProviderFundProfile.deposit,
                ProviderFundProfile.approval,
                ProviderFundProfile.disbursement,
                ProviderFundProfile.projects_approved,
            ).where(
                ProviderFundProfile.funding_provider_id.in_(provider_id_values)
            )
        )
        existing_profiles = {
            funding_provider_id: {
                "fund_type_id": fund_type_id,
                "fund_focus_id": fund_focus_id,
                "pledge": pledge,
                "deposit": deposit,
                "approval": approval,
                "disbursement": disbursement,
                "projects_approved": projects_approved,
            }
            for (
                funding_provider_id,
                fund_type_id,
                fund_focus_id,
                pledge,
                deposit,
                approval,
                disbursement,
                projects_approved,
            ) in result.all()
        }
        existing_provider_ids = set(existing_profiles)

        inserts = [
            payload
            for payload in payloads
            if payload["funding_provider_id"] not in existing_provider_ids
        ]
        updates = []

        provider_names_by_id = {
            provider_id: fund_name
            for fund_name, provider_id in provider_ids.items()
        }
        for payload in payloads:
            funding_provider_id = payload["funding_provider_id"]
            if funding_provider_id not in existing_provider_ids:
                continue

            current_profile = existing_profiles[funding_provider_id]
            comparable_payload = {
                "fund_type_id": payload["fund_type_id"],
                "fund_focus_id": payload["fund_focus_id"],
                "pledge": payload["pledge"],
                "deposit": payload["deposit"],
                "approval": payload["approval"],
                "disbursement": payload["disbursement"],
                "projects_approved": payload["projects_approved"],
            }

            if comparable_payload == current_profile:
                duplicated += 1
                self._log_fund_action(
                    action="duplicate_existing_fund_profile",
                    funding_provider_id=funding_provider_id,
                    fund_name=provider_names_by_id[funding_provider_id],
                )
                continue

            self._log_fund_action(
                action="update_by_fund_name",
                funding_provider_id=funding_provider_id,
                fund_name=provider_names_by_id[funding_provider_id],
            )
            updates.append(payload)

        if inserts:
            await self.uow.execute(
                insert(ProviderFundProfile.__table__), inserts
            )

        if updates:
            await self.uow.execute(
                update(ProviderFundProfile.__table__)
                .where(
                    ProviderFundProfile.__table__.c.funding_provider_id
                    == bindparam("b_funding_provider_id")
                )
                .values(
                    fund_type_id=bindparam("fund_type_id"),
                    fund_focus_id=bindparam("fund_focus_id"),
                    pledge=bindparam("pledge"),
                    deposit=bindparam("deposit"),
                    approval=bindparam("approval"),
                    disbursement=bindparam("disbursement"),
                    projects_approved=bindparam("projects_approved"),
                ),
                updates,
            )

        return len(inserts), len(updates), duplicated

    async def import_rows(
        self,
        rows: list[FundRowSchema],
    ) -> tuple[int, int, int, int]:
        if not rows:
            return 0, 0, 0, 0

        inserted = 0
        updated = 0
        duplicated = 0
        failed = 0

        try:
            inserted, updated, duplicated = await self._bulk_persist(rows)
            return inserted, updated, duplicated, failed
        except Exception as exc:
            await self._handle_fatal_error(exc)
            await self.uow.rollback()
            print(
                "[imports] fund status batch failed, "
                "falling back to row-by-row persistence: "
                f"{exc}"
            )

        for row_number, row in enumerate(rows, start=1):
            try:
                (
                    batch_inserted,
                    batch_updated,
                    batch_duplicated,
                ) = await self._bulk_persist([row])
                inserted += batch_inserted
                updated += batch_updated
                duplicated += batch_duplicated
            except Exception as exc:
                await self._handle_fatal_error(exc)
                await self.uow.rollback()
                failed += 1
                print(
                    "[imports] failed to persist fund status row "
                    f"(row={row_number}): {exc}"
                )

        return inserted, updated, duplicated, failed
