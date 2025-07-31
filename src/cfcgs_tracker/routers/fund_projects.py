from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from src.cfcgs_tracker.database.database import get_session
from src.cfcgs_tracker.schemas import (
    FundProjectList,
    FundProjectDataFilter,
    Message,
)
from src.services.fund_service import (
    get_fund_projects_data,
    insert_fund_project_from_df,
)
from src.utils.parser import read_file

router = APIRouter(prefix="/fund_projects", tags=["fund_projects"])

T_Session = Annotated[Session, Depends(get_session)]


@router.post("/", response_model=FundProjectList, status_code=HTTPStatus.OK)
def read_fund_projects(
    session: T_Session,
    filters: FundProjectDataFilter,
    limit: int = 40,
    offset: int = 0,
):
    fund_projects_data = get_fund_projects_data(
        session, filters, limit, offset
    )
    return {
        "fund_projects": fund_projects_data,
    }


@router.post("/upload", response_model=Message, status_code=HTTPStatus.OK)
async def upload_fund_project_file(
    session: T_Session, file: UploadFile = File(...)
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
        dataframe = read_file(file, file_type, upload_type=2)
        insert_fund_project_from_df(session, dataframe)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"message": "File processed and data inserted successfully"}
