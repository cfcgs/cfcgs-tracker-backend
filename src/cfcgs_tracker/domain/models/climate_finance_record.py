from typing import Optional

from sqlalchemy import ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column

from src.cfcgs_tracker.domain.base import table_registry


@table_registry.mapped_as_dataclass
class ClimateFinanceRecord:
    __tablename__ = "climate_finance_records"
    __table_args__ = (
        Index(
            "uq_climate_finance_records_source_row_hash",
            "source_row_hash",
            unique=True,
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    year: Mapped[int]
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    source_row_hash: Mapped[str] = mapped_column(nullable=False)

    approved_amount_usd_millions: Mapped[Optional[float]] = mapped_column(
        nullable=True, default=None
    )
    climate_finance_amount_usd_millions: Mapped[Optional[float]] = (
        mapped_column(nullable=True, default=None)
    )
    adaptation_amount_usd_millions: Mapped[Optional[float]] = mapped_column(
        nullable=True, default=None
    )
    mitigation_amount_usd_millions: Mapped[Optional[float]] = mapped_column(
        nullable=True, default=None
    )
    both_objectives_amount_usd_millions: Mapped[Optional[float]] = (
        mapped_column(nullable=True, default=None)
    )

    project_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("projects.id"), nullable=True, default=None
    )
    funding_provider_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("funding_providers.id"), nullable=True, default=None
    )
    financial_instrument_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("financial_instruments.id"), nullable=True, default=None
    )
    sector_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("sectors.id"), nullable=True, default=None
    )
    sub_sector_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("sub_sectors.id"), nullable=True, default=None
    )
    beneficiary_country_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("beneficiary_countries.id"), nullable=True, default=None
    )
