from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from src.cfcgs_tracker.database.database import get_session
from src.cfcgs_tracker.schemas import (
    CommitmentDataFilter,
    CommitmentList,
    Message, ObjectiveTotalsList, ObjectiveDataFilter,
)
from src.services.fund_service import (
    get_commitments_data,
    insert_commitments_from_df, get_totals_by_objective,
)
from src.utils.parser import read_file

router = APIRouter(prefix="/commitments", tags=["commitments"])

T_Session = Annotated[Session, Depends(get_session)]


@router.post("/", response_model=CommitmentList)
def read_commitments(
    session: T_Session,
    filters: CommitmentDataFilter,
    limit: int = 40,
    offset: int = 0,
):
    commitments_data = get_commitments_data(session, filters, limit, offset)
    return {
        "commitments": commitments_data,
    }


@router.post("/upload", response_model=Message, status_code=HTTPStatus.CREATED)
def upload_commitment_file(
    session: T_Session,
    file: UploadFile = File(...),
):
    filename = file.filename.lower()
    if filename.endswith(".csv"):
        file_type = "csv"
    elif filename.endswith(".xlsx"):
        file_type = "xlsx"
    else:
        raise HTTPException(
            status_code=400, detail="Only .csv and .xlsx are supported."
        )

    try:
        dataframe = read_file(file, file_type, upload_type=3)
        print(dataframe)
        insert_commitments_from_df(session, dataframe)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"message": "File processed and data inserted successfully"}

@router.post("/totals_by_objective", response_model=ObjectiveTotalsList)
def read_totals_by_objective(session: T_Session, filters: ObjectiveDataFilter):
    totals_data = get_totals_by_objective(session, filters)
    return {"totals": totals_data}