from typing import Optional, List

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
    years: Optional[List[int]] = (None,)
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