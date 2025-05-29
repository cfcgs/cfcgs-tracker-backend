from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from src.cfcgs_tracker.database import get_session
from src.cfcgs_tracker.schemas import FundList, FundDataFilter, Message, FundStatusResponse, FundStatusFilter, \
    FundSchema, FundUpdateSchema
from src.services.fund_service import get_funds_data, insert_funds_from_df, get_fund_status, delete_fund_by_id, \
    update_fund_by_id
from src.utils.parser import read_file

router = APIRouter(prefix="/funds", tags=["funds"])

T_Session = Annotated[Session, Depends(get_session)]

@router.post("/", response_model=FundList, status_code=HTTPStatus.OK)
def read_funds(
        session: T_Session,
        filters: FundDataFilter,
        limit: int = 40,
        offset: int = 0,
):
    funds_data = get_funds_data(session, filters, limit, offset)
    return {
        "funds": funds_data,
    }

@router.post("/upload", response_model=Message, status_code=HTTPStatus.CREATED)
async def upload_fund_file(
    session: T_Session,
    file: UploadFile = File(...),
):
    filename = file.filename.lower()
    if filename.endswith(".csv"):
        file_type = "csv"
    elif filename.endswith(".xlsx"):
        file_type = "xlsx"
    else:
        raise HTTPException(status_code=400, detail="Only .csv and .xlsx are supported.")

    try:
        dataframe = read_file(file, file_type)
        insert_funds_from_df(session, dataframe)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"message": "File processed and data inserted successfully"}

@router.post("/status", response_model=FundStatusResponse, status_code=HTTPStatus.OK)
def status_of_funds(
    session: T_Session,
    filters: FundStatusFilter,
):
    result = get_fund_status(session, filters)
    return {
        "total_pledge": 0 if not result else result.total_pledge or 0,
        "total_deposit": 0 if not result else result.total_deposit or 0,
        "total_approval": 0 if not result else result.total_approval or 0,
    }

@router.delete("/{fund_id}", response_model=Message, status_code=HTTPStatus.OK)
def delete_fund(
    fund_id: int,
    session: T_Session,
):
    result = delete_fund_by_id(session, fund_id)
    if not result:
        raise HTTPException(status_code=404, detail="Fund not found")
    return {"message": "Fund deleted successfully"}

@router.patch("/{fund_id}", response_model=FundSchema, status_code=HTTPStatus.OK)
def update_fund(
    fund_id: int,
    fund_data: FundUpdateSchema,
    session: T_Session,
):
    fund = update_fund_by_id(session, fund_id, fund_data)
    if not fund:
        raise HTTPException(status_code=404, detail="Fund not found")
    return fund