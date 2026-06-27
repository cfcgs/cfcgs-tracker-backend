"""fix sub sector unique constraint

Revision ID: 5d5a4b0a8d22
Revises: c8b66f400d8a
Create Date: 2026-06-22 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "5d5a4b0a8d22"
down_revision: Union[str, None] = "c8b66f400d8a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(
        op.f("sub_sectors_name_key"),
        "sub_sectors",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_sub_sectors_name_sector_id",
        "sub_sectors",
        ["name", "sector_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_sub_sectors_name_sector_id",
        "sub_sectors",
        type_="unique",
    )
    op.create_unique_constraint(
        op.f("sub_sectors_name_key"),
        "sub_sectors",
        ["name"],
    )
