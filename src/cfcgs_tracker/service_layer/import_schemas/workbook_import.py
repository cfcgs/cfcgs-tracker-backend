from pydantic import BaseModel, ConfigDict

from src.cfcgs_tracker.service_layer.import_schemas.climate_finance_row import (
    ClimateFinanceRowSchema,
)
from src.cfcgs_tracker.service_layer.import_schemas.fund_row import (
    FundRowSchema,
)


class WorkbookImportSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    climate_finance_rows: list[ClimateFinanceRowSchema]
    fund_rows: list[FundRowSchema]
    rows_received: int = 0
    rows_failed: int = 0
