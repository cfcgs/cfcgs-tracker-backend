from typing import Optional

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from src.cfcgs_tracker.domain.base import table_registry


@table_registry.mapped_as_dataclass
class ProviderFundProfile:
    __tablename__ = "provider_fund_profiles"

    funding_provider_id: Mapped[int] = mapped_column(
        ForeignKey("funding_providers.id"), primary_key=True
    )

    fund_type_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("fund_types.id"), nullable=True, default=None
    )
    fund_focus_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("fund_focuses.id"), nullable=True, default=None
    )
    pledge: Mapped[Optional[float]] = mapped_column(
        nullable=True, default=None
    )
    deposit: Mapped[Optional[float]] = mapped_column(
        nullable=True, default=None
    )
    approval: Mapped[Optional[float]] = mapped_column(
        nullable=True, default=None
    )
    disbursement: Mapped[Optional[float]] = mapped_column(
        nullable=True, default=None
    )
    projects_approved: Mapped[Optional[int]] = mapped_column(
        nullable=True, default=None
    )
