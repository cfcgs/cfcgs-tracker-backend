"""add rows duplicated to import jobs

Revision ID: 9f44f9d113c1
Revises: 2e6ef93c2a01
Create Date: 2026-06-23 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9f44f9d113c1"
down_revision: Union[str, None] = "2e6ef93c2a01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "import_jobs",
        sa.Column("rows_duplicated", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("import_jobs", "rows_duplicated")
