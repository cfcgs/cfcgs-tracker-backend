from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.cfcgs_tracker.database.database import get_session
from src.cfcgs_tracker.schemas import RegionList
from src.services.fund_service import get_regions

router = APIRouter(prefix="/regions", tags=["regions"])

T_Session = Annotated[Session, Depends(get_session)]


@router.get("/", response_model=RegionList, status_code=HTTPStatus.OK)
def read_regions(session: T_Session):
    result = get_regions(session)
    return {"regions": result}
