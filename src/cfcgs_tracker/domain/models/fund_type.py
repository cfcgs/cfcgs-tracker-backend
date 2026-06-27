from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.cfcgs_tracker.domain.models.provider_fund_profile import (
    ProviderFundProfile,
)
from src.cfcgs_tracker.domain.base import table_registry


@table_registry.mapped_as_dataclass
class FundType:
    __tablename__ = "fund_types"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)

    provider_fund_profiles: Mapped[list[ProviderFundProfile]] = relationship(
        init=False, cascade="all, delete-orphan", lazy="selectin"
    )
