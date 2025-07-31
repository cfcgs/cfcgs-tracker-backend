from typing import Optional

from sqlalchemy import ForeignKey
from sqlalchemy.orm import registry, Mapped, mapped_column, relationship

table_registry = registry()


@table_registry.mapped_as_dataclass
class Fund:
    __tablename__ = "funds"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    fund_name: Mapped[str] = mapped_column(unique=True)
    fund_type_id: Mapped[int] = mapped_column(
        ForeignKey("fund_types.id"), default=None
    )
    fund_focus_id: Mapped[int] = mapped_column(
        ForeignKey("fund_focuses.id"), default=None
    )
    pledge: Mapped[float] = mapped_column(nullable=True, default=None)
    deposit: Mapped[float] = mapped_column(nullable=True, default=None)
    approval: Mapped[float] = mapped_column(nullable=True, default=None)
    disbursement: Mapped[float] = mapped_column(nullable=True, default=None)
    projects_approved: Mapped[int] = mapped_column(nullable=True, default=None)

    fund_type: Mapped["FundType"] = relationship(
        back_populates="funds", init=False
    )
    fund_focus: Mapped["FundFocus"] = relationship(
        back_populates="funds", init=False
    )
    fund_projects: Mapped[list["Project"]] = relationship(
        back_populates="fund", default_factory=list
    )


@table_registry.mapped_as_dataclass
class FundType:
    __tablename__ = "fund_types"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    name: Mapped[str] = mapped_column(unique=True)

    funds: Mapped[list["Fund"]] = relationship(
        back_populates="fund_type", default_factory=list
    )


@table_registry.mapped_as_dataclass
class FundFocus:
    __tablename__ = "fund_focuses"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    name: Mapped[str] = mapped_column(unique=True)

    funds: Mapped[list["Fund"]] = relationship(
        back_populates="fund_focus", default_factory=list
    )


@table_registry.mapped_as_dataclass
class Project:
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    name: Mapped[str] = mapped_column(unique=True)
    fund_id: Mapped[int] = mapped_column(
        ForeignKey("funds.id"), default=None, nullable=True
    )
    country_id: Mapped[int] = mapped_column(
        ForeignKey("countries.id"), default=None
    )

    fund: Mapped["Fund"] = relationship(
        back_populates="fund_projects", init=False
    )
    country: Mapped["Country"] = relationship(
        back_populates="fund_projects", init=False
    )
    commitments: Mapped[list["Commitment"]] = relationship(
        back_populates="project", default_factory=list
    )


@table_registry.mapped_as_dataclass
class Region:
    __tablename__ = "regions"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    name: Mapped[str] = mapped_column(unique=True)

    countries: Mapped[list["Country"]] = relationship(
        back_populates="region", default_factory=list
    )


@table_registry.mapped_as_dataclass
class Country:
    __tablename__ = "countries"

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    name: Mapped[str] = mapped_column(unique=True)
    region_id: Mapped[int] = mapped_column(
        ForeignKey("regions.id"), nullable=True, default=None
    )

    region: Mapped[Region] = relationship(
        back_populates="countries", init=False
    )
    fund_projects: Mapped[list["Project"]] = relationship(
        back_populates="country", default_factory=list
    )

    commitments_as_recipient: Mapped[list["Commitment"]] = relationship(
        back_populates="recipient_country",
        foreign_keys="[Commitment.recipient_country_id]",
        default_factory=list
    )


@table_registry.mapped_as_dataclass
class FundingEntity:
    __tablename__ = "funding_entities"  # A nova tabela genérica

    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    name: Mapped[str] = mapped_column(unique=True, nullable=False)

    # Relações para navegar de uma entidade para os compromissos onde ela atuou
    commitments_as_provider: Mapped[list["Commitment"]] = relationship(
        back_populates="provider",
        foreign_keys="[Commitment.provider_id]",
        default_factory=list,
    )
    commitments_as_channel: Mapped[list["Commitment"]] = relationship(
        back_populates="channel",
        foreign_keys="[Commitment.channel_id]",
        default_factory=list,
    )


@table_registry.mapped_as_dataclass
class Commitment:
    __tablename__ = "commitments"
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    year: Mapped[int] = mapped_column(nullable=False)
    amount_usd_thousand: Mapped[float] = mapped_column(nullable=False)
    adaptation_amount_usd_thousand: Mapped[Optional[float]] = mapped_column(nullable=True)
    mitigation_amount_usd_thousand: Mapped[Optional[float]] = mapped_column(nullable=True)
    recipient_country_id: Mapped[int] = mapped_column(
        ForeignKey("countries.id"), default=None,
    )
    provider_id: Mapped[int] = mapped_column(
        ForeignKey("funding_entities.id"), default=None
    )
    channel_id: Mapped[int] = mapped_column(
        ForeignKey("funding_entities.id"), nullable=True, default=None
    )
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id"), nullable=True, default=None
    )

    project: Mapped["Project"] = relationship(
        back_populates="commitments", init=False
    )
    provider: Mapped["FundingEntity"] = relationship(
        back_populates="commitments_as_provider",
        foreign_keys=[provider_id],
        init=False,
    )
    channel: Mapped["FundingEntity"] = relationship(
        back_populates="commitments_as_channel",
        foreign_keys=[channel_id],
        init=False,
    )

    recipient_country: Mapped["Country"] = relationship(
        back_populates="commitments_as_recipient",
        foreign_keys=[recipient_country_id],
        init=False
    )
