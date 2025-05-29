from sqlalchemy import ForeignKey
from sqlalchemy.orm import registry, Mapped, mapped_column, relationship

table_registry = registry()

@table_registry.mapped_as_dataclass
class Fund:
    __tablename__ = 'funds'


    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    fund_name: Mapped[str] = mapped_column(unique=True)
    fund_type_id: Mapped[int] = mapped_column(ForeignKey('fund_types.id'), default=None)
    fund_focus_id: Mapped[int] = mapped_column(ForeignKey('fund_focuses.id'), default=None)
    pledge: Mapped[float] = mapped_column(nullable=True, default=None)
    deposit: Mapped[float] = mapped_column(nullable=True, default=None)
    approval: Mapped[float] = mapped_column(nullable=True, default=None)
    disbursement: Mapped[float] = mapped_column(nullable=True, default=None)
    projects_approved: Mapped[int] = mapped_column(nullable=True, default=None)

    fund_type: Mapped["FundType"] = relationship(back_populates="funds", init=False)
    fund_focus: Mapped["FundFocus"] = relationship(back_populates="funds", init=False)



@table_registry.mapped_as_dataclass
class FundType:
    __tablename__ = 'fund_types'

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    name: Mapped[str] = mapped_column(unique=True)

    funds: Mapped[list["Fund"]] = relationship(back_populates="fund_type", default_factory=list)


@table_registry.mapped_as_dataclass
class FundFocus:
    __tablename__ = 'fund_focuses'
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    name: Mapped[str] = mapped_column(unique=True)

    funds: Mapped[list["Fund"]] = relationship(back_populates="fund_focus", default_factory=list)
