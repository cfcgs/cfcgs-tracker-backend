from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.cfcgs_tracker.database import get_session
from src.cfcgs_tracker.schemas import FundTypeList, Message, FundTypeSchema, FundTypeUpdateSchema
from src.services.fund_service import get_fund_types, delete_fund_type_by_id, update_fund_type_by_id

router = APIRouter(prefix="/fund_types", tags=["fund_types"])
T_Session = Annotated[Session, Depends(get_session)]

@router.get("/", response_model=FundTypeList, status_code=HTTPStatus.OK)
def read_fund_types(session: T_Session):
    result = get_fund_types(session)
    return {
        "fund_types": result
    }

@router.delete("/{type_id}", response_model=Message, status_code=HTTPStatus.OK)
def delete_fund_type(
    type_id: int,
    session: T_Session,
):
    result = delete_fund_type_by_id(session, type_id)
    if not result:
        raise HTTPException(status_code=404, detail="Fund type not found")
    return {"message": "Fund type deleted successfully"}

@router.patch("/{type_id}", response_model=FundTypeSchema, status_code=HTTPStatus.OK)
def update_fund_type(
    type_id: int,
    type_data: FundTypeUpdateSchema,
    session: T_Session,
):
    fund_type = update_fund_type_by_id(session, type_id, type_data)
    if not fund_type:
        raise HTTPException(status_code=404, detail="Fund type not found")
    return fund_type