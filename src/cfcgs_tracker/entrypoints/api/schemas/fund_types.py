from pydantic import BaseModel


class FundTypePublic(BaseModel):
    id: int
    name: str
