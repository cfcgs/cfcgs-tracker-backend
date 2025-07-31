from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.cfcgs_tracker.database.database import get_session
from src.cfcgs_tracker.schemas import CountryList
from src.services.fund_service import get_countries, get_recipient_countries

router = APIRouter(prefix="/countries", tags=["countries"])

T_Session = Annotated[Session, Depends(get_session)]

@router.get("/", response_model=CountryList, status_code=HTTPStatus.OK)
def read_countries(session: T_Session):
    result = get_countries(session)
    return {"countries": result}

@router.get("/recipients", response_model=CountryList, status_code=HTTPStatus.OK)
def read_recipient_countries(session: T_Session):
    """Retorna apenas os países que são RECEPTORES de financiamento."""
    result = get_recipient_countries(session)
    return {"countries": result}