from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.cfcgs_tracker.database.database import get_session
from src.cfcgs_tracker.schemas import (
    FundFocusList,
    Message,
    FundFocusSchema,
    FundFocusUpdateSchema,
)
from src.services.fund_service import (
    get_fund_focuses,
    delete_fund_focus_by_id,
    update_fund_focus_by_id,
)

router = APIRouter(prefix="/fund_focuses", tags=["fund_focuses"])

T_Session = Annotated[Session, Depends(get_session)]


@router.get("/", response_model=FundFocusList, status_code=HTTPStatus.OK)
def read_fund_focuses(session: T_Session):
    result = get_fund_focuses(session)
    return {"fund_focuses": result}


@router.delete(
    "/{focus_id}", response_model=Message, status_code=HTTPStatus.OK
)
def delete_fund_focus(
    focus_id: int,
    session: T_Session,
):
    result = delete_fund_focus_by_id(session, focus_id)
    if not result:
        raise HTTPException(status_code=404, detail="Fund focus not found")
    return {"message": "Fund focus deleted successfully"}


@router.patch(
    "/{focus_id}", response_model=FundFocusSchema, status_code=HTTPStatus.OK
)
def update_fund_focus(
    focus_id: int,
    focus_data: FundFocusUpdateSchema,
    session: T_Session,
):
    fund_focus = update_fund_focus_by_id(session, focus_id, focus_data)
    if not fund_focus:
        raise HTTPException(status_code=404, detail="Fund focus not found")
    return fund_focus
