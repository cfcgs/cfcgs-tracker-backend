from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.cfcgs_tracker.domain.models.climate_finance_record import (
    ClimateFinanceRecord,
)
from src.cfcgs_tracker.domain.base import table_registry


@table_registry.mapped_as_dataclass
class SubSector:
    __tablename__ = "sub_sectors"
    __table_args__ = (
        UniqueConstraint(
            "name",
            "sector_id",
            name="uq_sub_sectors_name_sector_id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    name: Mapped[str]

    sector_id: Mapped[int] = mapped_column(ForeignKey("sectors.id"))
    climate_finance_records: Mapped[list[ClimateFinanceRecord]] = relationship(
        init=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )
