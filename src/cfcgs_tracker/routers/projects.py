# src/cfcgs_tracker/routers/projects.py
from http import HTTPStatus
from typing import Annotated, List, Dict, Any, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from src.cfcgs_tracker.database.database import get_session
from src.cfcgs_tracker.schemas import ProjectListResponse, PaginatedProjectResponse
from src.cfcgs_tracker.routers.commitments import ObjectiveFilter
from src.services.fund_service import get_commitment_projects, get_paginated_commitment_projects

router = APIRouter(prefix="/projects", tags=["projects"])

T_Session = Annotated[Session, Depends(get_session)]

@router.get("/commitments", response_model=ProjectListResponse, status_code=HTTPStatus.OK)
def read_commitment_projects(session: T_Session):
    """
    Retorna uma lista de todos os projetos únicos que
    receberam pelo menos um compromisso.
    """
    result = get_commitment_projects(session) # Retorna List[dict]
    return {"projects": result}

@router.get(
    "/commitments/paginated",
    response_model=PaginatedProjectResponse,
    status_code=HTTPStatus.OK
)
async def read_paginated_commitment_projects(
    session: T_Session,
    search: Optional[str] = Query(None, description="Termo de busca para o nome do projeto"),
    years: Optional[List[int]] = Query(
        None, alias="year", description="Filtra por um ou mais anos"
    ),
    country_ids: Optional[List[int]] = Query(
        None, alias="country_id", description="Filtra por um ou mais IDs de paises"
    ),
    objective: ObjectiveFilter = Query(
        ObjectiveFilter.all,
        description="Filtra por objetivo do financiamento (padrao: 'all')",
    ),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Retorna uma lista paginada de projetos que receberam pelo menos um compromisso.
    """
    result = await get_paginated_commitment_projects(
        session,
        search=search,
        limit=limit,
        offset=offset,
        filter_years=years,
        filter_country_ids=country_ids,
        objective=objective.value,
    )
    return result
