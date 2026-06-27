"""add natural key index to climate finance records

Revision ID: b91d1b5fb2b7
Revises: 5d5a4b0a8d22
Create Date: 2026-06-23 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "b91d1b5fb2b7"
down_revision: Union[str, None] = "5d5a4b0a8d22"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
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


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS uq_climate_finance_records_natural_key
        """
    )
