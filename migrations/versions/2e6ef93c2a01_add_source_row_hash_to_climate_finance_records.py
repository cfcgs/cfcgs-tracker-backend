"""add source row hash to climate finance records

Revision ID: 2e6ef93c2a01
Revises: b91d1b5fb2b7
Create Date: 2026-06-23 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2e6ef93c2a01"
down_revision: Union[str, None] = "b91d1b5fb2b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "climate_finance_records",
        sa.Column("source_row_hash", sa.String(), nullable=True),
    )
    op.execute(
        """
        DROP INDEX IF EXISTS uq_climate_finance_records_natural_key
        """
    )
    op.create_index(
        "uq_climate_finance_records_source_row_hash",
        "climate_finance_records",
        ["source_row_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "uq_climate_finance_records_source_row_hash",
        table_name="climate_finance_records",
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
        uq_climate_finance_records_natural_key
        ON climate_finance_records (
            year,
            project_id,
            funding_provider_id,
            source_id,
            beneficiary_country_id,
            sector_id,
            sub_sector_id,
            financial_instrument_id
        ) NULLS NOT DISTINCT
        """
    )
    op.drop_column("climate_finance_records", "source_row_hash")
