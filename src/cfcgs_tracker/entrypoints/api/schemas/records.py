from pydantic import BaseModel


class SourceReferencePublic(BaseModel):
    label: str
    url: str


class RecordSearchFilter(BaseModel):
    years: list[int] | None = None
    countries: list[int] | None = None


class RecordSearchPublic(BaseModel):
    id: int
    year: int
    amount_usd_millions: float
    funding_provider: str | None = None
    recipient_country: str | None = None
    project: str | None = None
    source: str | None = None


class RecordSearchList(BaseModel):
    records: list[RecordSearchPublic]


class RecordSummaryPublic(BaseModel):
    total_projects: int
    total_funded_countries: int
    total_countries: int
    total_amount: float
    total_adaptation: float
    total_mitigation: float
    total_overlap: float


class RecordSummaryResponse(BaseModel):
    total_projects: int
    total_funded_countries: int
    total_countries: int
    total_amount: float
    total_adaptation: float
    total_mitigation: float
    total_overlap: float
    sources: list[SourceReferencePublic]


class RecordObjectiveTotalPublic(BaseModel):
    year: int
    total_adaptation: float
    total_mitigation: float
    total_overlap: float


class RecordObjectiveTotalList(BaseModel):
    totals: list[RecordObjectiveTotalPublic]


class RecordObjectiveTotalResponse(BaseModel):
    totals: list[RecordObjectiveTotalPublic]
    sources: list[SourceReferencePublic]


class RecordTimeSeriesPoint(BaseModel):
    year: int
    amount: float


class RecordTimeSeriesPublic(BaseModel):
    name: str
    data: list[RecordTimeSeriesPoint]


class RecordTimeSeriesList(BaseModel):
    series: list[RecordTimeSeriesPublic]


class RecordTimeSeriesResponse(BaseModel):
    series: list[RecordTimeSeriesPublic]
    sources: list[SourceReferencePublic]


class HeatmapCountryPublic(BaseModel):
    id: int
    name: str
    region: str | None = None


class HeatmapProjectPublic(BaseModel):
    id: int
    name: str


class RecordFilterOptionsPublic(BaseModel):
    years: list[int]
    countries: list[HeatmapCountryPublic]
    projects: list[HeatmapProjectPublic]
    objectives: list[str]


class HeatmapAxisTotalPublic(BaseModel):
    label: str
    total_amount: float
    project_count: int
    percent_of_total: float
    country_id: int | None = None
    year: int | None = None


class HeatmapCellPublic(BaseModel):
    year: int
    country_id: int
    country_name: str
    row_label: str
    column_label: str
    total_amount: float
    adaptation_exclusive: float
    mitigation_exclusive: float
    overlap: float
    project_count: int
    percent_of_total: float
    percent_of_row: float
    percent_of_column: float


class RecordCountryYearGridPublic(BaseModel):
    view: str
    rows: list[str]
    columns: list[str]
    row_totals: list[HeatmapAxisTotalPublic]
    column_totals: list[HeatmapAxisTotalPublic]
    cells: list[HeatmapCellPublic]
    grand_total: float
    grand_total_projects: int
    row_count: int
    column_count: int
    row_offset: int
    column_offset: int
    row_limit: int
    column_limit: int
    sources: list[SourceReferencePublic]


class RecordOverviewPublic(BaseModel):
    years: list[int]
    countries: list[HeatmapCountryPublic]
    projects: list[HeatmapProjectPublic]
    objectives: list[str]
    summary: RecordSummaryResponse
    grid: RecordCountryYearGridPublic


class RecordCountryYearProjectPublic(BaseModel):
    id: int
    name: str
    objective: str
    total_amount: float
    adaptation_exclusive: float
    mitigation_exclusive: float
    overlap: float


class RecordCountryYearProjectList(BaseModel):
    total: int
    has_more: bool
    projects: list[RecordCountryYearProjectPublic]


class PaginatedRecordProjectPublic(BaseModel):
    id: int
    name: str


class PaginatedRecordProjectList(BaseModel):
    projects: list[PaginatedRecordProjectPublic]
    total: int
    limit: int
    offset: int
    has_more: bool


class AdminRecordPublic(BaseModel):
    id: int
    year: int
    source_row_hash: str
    project_title: str | None = None
    beneficiary_country: str | None = None
    funding_provider: str | None = None
    source: str | None = None
    source_url: str | None = None
    financial_instrument: str | None = None
    sector: str | None = None
    sub_sector: str | None = None
    approved_amount_usd_millions: float | None = None
    climate_finance_amount_usd_millions: float | None = None
    adaptation_amount_usd_millions: float | None = None
    mitigation_amount_usd_millions: float | None = None
    both_objectives_amount_usd_millions: float | None = None


class AdminRecordList(BaseModel):
    records: list[AdminRecordPublic]
    total: int
    limit: int
    offset: int
    has_more: bool


class AdminRecordFilterOptionsPublic(BaseModel):
    year: list[int]
    project_title: list[str]
    beneficiary_country: list[str]
    funding_provider: list[str]
    source: list[str]
    source_url: list[str]
    financial_instrument: list[str]
    sector: list[str]
    sub_sector: list[str]


class AdminRecordFilterSuggestionsPublic(BaseModel):
    column: str
    values: list[str | int]
    limit: int
    offset: int
    has_more: bool


class AdminRecordUpdate(BaseModel):
    year: int | None = None
    project_title: str | None = None
    beneficiary_country: str | None = None
    funding_provider: str | None = None
    source: str | None = None
    source_url: str | None = None
    financial_instrument: str | None = None
    sector: str | None = None
    sub_sector: str | None = None
    approved_amount_usd_millions: float | None = None
    climate_finance_amount_usd_millions: float | None = None
    adaptation_amount_usd_millions: float | None = None
    mitigation_amount_usd_millions: float | None = None
    both_objectives_amount_usd_millions: float | None = None
