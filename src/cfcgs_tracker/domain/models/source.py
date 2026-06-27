from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.cfcgs_tracker.domain.models.climate_finance_record import (
    ClimateFinanceRecord,
)
from src.cfcgs_tracker.domain.base import table_registry


@table_registry.mapped_as_dataclass
class Source:
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    name: Mapped[str] = mapped_column(unique=True)
    url: Mapped[str] = mapped_column(unique=True)

    climate_finance_records: Mapped[list[ClimateFinanceRecord]] = relationship(
        init=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )
