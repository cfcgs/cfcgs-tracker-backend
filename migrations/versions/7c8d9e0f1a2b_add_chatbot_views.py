"""add chatbot views

Revision ID: 7c8d9e0f1a2b
Revises: 6a7b8c9d0e1f
Create Date: 2026-06-27 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "7c8d9e0f1a2b"
down_revision: Union[str, None] = "6a7b8c9d0e1f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


CLIMATE_FINANCE_RECORDS_VIEW = """
CREATE VIEW view_climate_finance_records_detailed AS
SELECT
    cfr.id AS record_id,
    cfr.source_row_hash AS source_row_hash,
    cfr.year AS year,
    p.id AS project_id,
    p.title AS project_title,
    bc.id AS beneficiary_country_id,
    bc.name AS beneficiary_country_name,
    fp.id AS funding_provider_id,
    fp.name AS funding_provider_name,
    s.id AS source_id,
    s.name AS source_name,
    s.url AS source_url,
    fi.id AS financial_instrument_id,
    fi.name AS financial_instrument_name,
    sec.id AS sector_id,
    sec.name AS sector_name,
    ssec.id AS sub_sector_id,
    ssec.name AS sub_sector_name,
    cfr.approved_amount_usd_millions AS approved_amount_usd_millions,
    cfr.climate_finance_amount_usd_millions AS climate_finance_amount_usd_millions,
    cfr.adaptation_amount_usd_millions AS adaptation_amount_usd_millions,
    cfr.mitigation_amount_usd_millions AS mitigation_amount_usd_millions,
    cfr.both_objectives_amount_usd_millions AS both_objectives_amount_usd_millions
FROM climate_finance_records cfr
LEFT JOIN projects p
    ON p.id = cfr.project_id
LEFT JOIN beneficiary_countries bc
    ON bc.id = cfr.beneficiary_country_id
LEFT JOIN funding_providers fp
    ON fp.id = cfr.funding_provider_id
JOIN sources s
    ON s.id = cfr.source_id
LEFT JOIN financial_instruments fi
    ON fi.id = cfr.financial_instrument_id
LEFT JOIN sectors sec
    ON sec.id = cfr.sector_id
LEFT JOIN sub_sectors ssec
    ON ssec.id = cfr.sub_sector_id
"""


PROVIDER_FUND_PROFILES_VIEW = """
CREATE VIEW view_provider_fund_profiles_detailed AS
SELECT
    pfp.funding_provider_id AS funding_provider_id,
    fp.name AS funding_provider_name,
    pfp.fund_type_id AS fund_type_id,
    ft.name AS fund_type_name,
    pfp.fund_focus_id AS fund_focus_id,
    ff.name AS fund_focus_name,
    pfp.pledge AS pledge,
    pfp.deposit AS deposit,
    pfp.approval AS approval,
    pfp.disbursement AS disbursement,
    pfp.projects_approved AS projects_approved
FROM provider_fund_profiles pfp
JOIN funding_providers fp
    ON fp.id = pfp.funding_provider_id
LEFT JOIN fund_types ft
    ON ft.id = pfp.fund_type_id
LEFT JOIN fund_focuses ff
    ON ff.id = pfp.fund_focus_id
"""


def upgrade() -> None:
    op.execute(CLIMATE_FINANCE_RECORDS_VIEW)
    op.execute(PROVIDER_FUND_PROFILES_VIEW)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS view_provider_fund_profiles_detailed")
    op.execute("DROP VIEW IF EXISTS view_climate_finance_records_detailed")
