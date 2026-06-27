"""update chatbot views with total amount

Revision ID: 8d9e0f1a2b3c
Revises: 7c8d9e0f1a2b
Create Date: 2026-06-27 00:00:01.000000

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "8d9e0f1a2b3c"
down_revision: Union[str, None] = "7c8d9e0f1a2b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


RECREATE_CLIMATE_FINANCE_RECORDS_VIEW = """
CREATE OR REPLACE VIEW view_climate_finance_records_detailed AS
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
    cfr.both_objectives_amount_usd_millions AS both_objectives_amount_usd_millions,
    COALESCE(
        cfr.climate_finance_amount_usd_millions,
        COALESCE(cfr.adaptation_amount_usd_millions, 0)
        + COALESCE(cfr.mitigation_amount_usd_millions, 0)
        + COALESCE(cfr.both_objectives_amount_usd_millions, 0),
        0
    ) AS total_amount_usd_millions
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


PREVIOUS_CLIMATE_FINANCE_RECORDS_VIEW = """
CREATE OR REPLACE VIEW view_climate_finance_records_detailed AS
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


def upgrade() -> None:
    op.execute(RECREATE_CLIMATE_FINANCE_RECORDS_VIEW)


def downgrade() -> None:
    op.execute(PREVIOUS_CLIMATE_FINANCE_RECORDS_VIEW)
