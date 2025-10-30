"""Add indexes and detailed commitments view for chatbot performance

Revision ID: cd73046f0307
Revises: a33ce1dd3cc1
Create Date: 2025-10-28 12:07:51.718232

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cd73046f0307'
down_revision: Union[str, None] = 'a33ce1dd3cc1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # ### Novos comandos para índices e view ###
    print("Creating indexes for performance...")

    # --- Índices na tabela Projects ---
    op.create_index(op.f('ix_projects_country_id'), 'projects', ['country_id'], unique=False)
    op.create_index(op.f('ix_projects_fund_id'), 'projects', ['fund_id'], unique=False)

    # --- Índice na tabela Countries ---
    op.create_index(op.f('ix_countries_region_id'), 'countries', ['region_id'], unique=False)

    print("Creating view_commitments_detailed...")
    # --- Criação da View ---
    op.execute("""
    CREATE OR REPLACE VIEW view_commitments_detailed AS
    SELECT 
        c.id AS commitment_id,
        c.year,
        c.amount_usd_thousand,
        c.adaptation_amount_usd_thousand,
        c.mitigation_amount_usd_thousand,
        c.overlap_amount_usd_thousand,
        p.id AS project_id,
        p.name AS project_name,
        co.id AS country_id,
        co.name AS country_name,
        r.id AS region_id,
        r.name AS region_name,
        prov.id AS provider_id,
        prov.name AS provider_name,
        chan.id AS channel_id,
        chan.name AS channel_name,
        f.id AS fund_id,
        f.fund_name AS fund_name,
        ft.name AS fund_type_name,
        ff.name AS fund_focus_name
    FROM 
        commitments c
    LEFT JOIN 
        projects p ON c.project_id = p.id
    LEFT JOIN 
        countries co ON c.recipient_country_id = co.id -- Junta pelo recipient_country_id do commitment
    LEFT JOIN 
        regions r ON co.region_id = r.id
    LEFT JOIN 
        funding_entities prov ON c.provider_id = prov.id
    LEFT JOIN 
        funding_entities chan ON c.channel_id = chan.id
    LEFT JOIN 
        funds f ON p.fund_id = f.id -- Junta fundos através do projeto
    LEFT JOIN
        fund_types ft ON f.fund_type_id = ft.id
    LEFT JOIN
        fund_focuses ff ON f.fund_focus_id = ff.id;
    """)
    print("Columns, indexes, and view created successfully.")
    # ### Fim dos novos comandos ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### Novos comandos para remover índices e view (executar primeiro) ###
    print("Dropping view_commitments_detailed...")
    # --- Remoção da View ---
    op.execute("DROP VIEW IF EXISTS view_commitments_detailed;")

    print("Dropping indexes...")
    # --- Remoção dos Índices (ordem reversa da criação é uma boa prática) ---
    op.drop_index(op.f('ix_countries_region_id'), table_name='countries')
    op.drop_index(op.f('ix_projects_fund_id'), table_name='projects')
    op.drop_index(op.f('ix_projects_country_id'), table_name='projects')
    print("Indexes and view dropped.")
    # ### Fim dos novos comandos ###