from typing import Optional, List, Dict, Any

from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    message: str


class FundFocusSchema(BaseModel):
    id: int
    name: str


class FundFocusUpdateSchema(BaseModel):
    name: Optional[str] = None


class FundFocusList(BaseModel):
    fund_focuses: List[FundFocusSchema]


class FundTypeSchema(BaseModel):
    id: int
    name: str


class FundTypeUpdateSchema(BaseModel):
    name: Optional[str] = None


class FundTypeList(BaseModel):
    fund_types: List[FundTypeSchema]


class FundSchema(BaseModel):
    id: int
    fund_name: str
    fund_type: Optional[str]
    fund_focus: Optional[str]
    pledge: Optional[float]
    deposit: Optional[float]
    approval: Optional[float]
    disbursement: Optional[float]
    projects_approved: Optional[int]

    model_config = ConfigDict(from_attributes=True)


class FundUpdateSchema(BaseModel):
    fund_name: Optional[str] = None
    fund_type: Optional[str] = None
    fund_focus: Optional[str] = None
    pledge: Optional[float] = None
    deposit: Optional[float] = None
    approval: Optional[float] = None
    disbursement: Optional[float] = None
    projects_approved: Optional[int] = None


class FundStatusResponse(BaseModel):
    total_pledge: float
    total_deposit: float
    total_approval: float


class FundList(BaseModel):
    funds: list[FundSchema]


class FundStatusFilter(BaseModel):
    funds: Optional[List[int]] = None
    fund_types: Optional[List[int]] = None
    fund_focuses: Optional[List[int]] = None


class FundDataFilter(BaseModel):
    fund_types: Optional[List[int]] = None
    fund_focuses: Optional[List[int]] = None


class FundProjectDataFilter(BaseModel):
    funds: Optional[List[int]] = None
    countries: Optional[List[int]] = None
    regions: Optional[List[int]] = None


class FundProjectSchema(BaseModel):
    id: int
    name: str
    fund_name: Optional[str] = None
    country_name: Optional[str] = None
    region: Optional[str] = None


class FundProjectList(BaseModel):
    fund_projects: list[FundProjectSchema]


class CommitmentDataFilter(BaseModel):
    years: Optional[List[int]] = None
    countries: Optional[List[int]] = None


class CommitmentDataSchema(BaseModel):
    id: int
    year: int
    amount_usd_thousand: float
    channel_of_delivery: Optional[str]
    provider_country: Optional[str]
    recipient_country: Optional[str]
    project: Optional[str]


class CommitmentList(BaseModel):
    commitments: List[CommitmentDataSchema]


class RegionSchema(BaseModel):
    id: int
    name: str


class RegionList(BaseModel):
    regions: List[RegionSchema]


class CountrySchema(BaseModel):
    id: int
    name: str
    region: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class CountryList(BaseModel):
    countries: List[CountrySchema]


class ObjectiveDataFilter(BaseModel):
    years: Optional[List[int]] = None
    recipient_countries: Optional[List[int]] = None # Filtraremos por ID do país


class ObjectiveTotalSchema(BaseModel):
    year: int
    total_adaptation: float
    total_mitigation: float
    total_overlap: float


class ObjectiveTotalsList(BaseModel):
    totals: List[ObjectiveTotalSchema]


class TimeSeriesDataPoint(BaseModel):
    year: int
    amount: float


class TimeSeriesData(BaseModel):
    name: str  # Nome da série (ex: "Financiamento Total Agregado" ou "Brasil")
    data: List[TimeSeriesDataPoint]


class TimeSeriesResponse(BaseModel):
    series: List[TimeSeriesData]

class ChatQuery(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    page: int = 1
    page_size: int = 10
    confirm_pagination: bool = False
    disambiguation_choice: Optional[Dict[str, str]] = None


class PaginationResult(BaseModel):
    page: int
    page_size: int
    total_rows: int
    has_more: bool
    rows: List[Dict[str, Any]]


class ChatSource(BaseModel):
    name: str
    url: str


class DisambiguationOption(BaseModel):
    name: str
    kind: str


class DisambiguationPayload(BaseModel):
    message: str
    options: List[DisambiguationOption]
    mode: str = "select"


class ChatResponse(BaseModel):
    answer: str
    needs_pagination_confirmation: bool = False
    pagination: Optional[PaginationResult] = None
    sources: Optional[List[ChatSource]] = None
    disambiguation: Optional[DisambiguationPayload] = None


class KpiResponseSchema(BaseModel):
    """Retorna os KPIs principais (Nº de Projetos, Nº de Países)."""
    total_projects: int
    total_funded_countries: int


class HeatmapCellSchema(BaseModel):
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


class HeatmapAxisTotalSchema(BaseModel):
    label: str
    total_amount: float
    project_count: int
    percent_of_total: float
    country_id: Optional[int] = None
    year: Optional[int] = None


class HeatmapResponseSchema(BaseModel):
    view: str
    rows: List[str]
    columns: List[str]
    row_totals: List[HeatmapAxisTotalSchema]
    column_totals: List[HeatmapAxisTotalSchema]
    cells: List[HeatmapCellSchema]
    grand_total: float
    grand_total_projects: int
    row_count: int
    column_count: int
    row_offset: int
    column_offset: int
    row_limit: int
    column_limit: int


class HeatmapProjectSchema(BaseModel):
    id: int
    name: str
    objective: str
    total_amount: float
    adaptation_exclusive: float
    mitigation_exclusive: float
    overlap: float


class HeatmapProjectsResponseSchema(BaseModel):
    total: int
    has_more: bool
    projects: List[HeatmapProjectSchema]


class HeatmapKpiResponseSchema(BaseModel):
    total_projects: int
    total_countries: int
    total_amount: float
    total_adaptation: float
    total_mitigation: float
    total_overlap: float

class ProjectSimple(BaseModel):
    id: int
    name: str

class CountrySimple(BaseModel):
    id: int
    name: str

class ProjectListResponse(BaseModel):
    projects: List[ProjectSimple]

class PaginatedProjectResponse(BaseModel):
    projects: List[ProjectSimple]
    total: int
    has_more: bool

class HeatmapFilterOptionsSchema(BaseModel):
    years: List[int]
    countries: List[CountrySimple]
    projects: List[ProjectSimple]
    objectives: List[str]
