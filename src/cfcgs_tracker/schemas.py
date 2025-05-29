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
