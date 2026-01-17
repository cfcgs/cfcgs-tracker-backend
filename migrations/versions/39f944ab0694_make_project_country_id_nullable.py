"""Make Project.country_id nullable

Revision ID: 39f944ab0694
Revises: cd73046f0307
Create Date: 2025-11-01 21:13:48.385388

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '39f944ab0694'
down_revision: Union[str, None] = 'cd73046f0307'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column('projects', 'country_id',
               existing_type=sa.INTEGER(),
               nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column('projects', 'country_id',
               existing_type=sa.INTEGER(),
               nullable=False)
    # ### end Alembic commands ###
