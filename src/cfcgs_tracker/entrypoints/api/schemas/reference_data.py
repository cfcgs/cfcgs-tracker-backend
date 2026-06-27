from pydantic import BaseModel, ConfigDict


class FundTypePublic(BaseModel):
    id: int
    name: str

    model_config = ConfigDict(from_attributes=True)


class FundTypeList(BaseModel):
    fund_types: list[FundTypePublic]


class FundFocusPublic(BaseModel):
    id: int
    name: str

    model_config = ConfigDict(from_attributes=True)


class FundFocusList(BaseModel):
    fund_focuses: list[FundFocusPublic]


class CountryPublic(BaseModel):
    id: int
    name: str
    region: str | None = None

    model_config = ConfigDict(from_attributes=True)


class CountryList(BaseModel):
    countries: list[CountryPublic]
