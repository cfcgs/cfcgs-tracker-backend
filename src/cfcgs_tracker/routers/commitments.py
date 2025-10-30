from enum import Enum
from http import HTTPStatus
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.cfcgs_tracker.database.database import get_session
from src.cfcgs_tracker.schemas import (
    CommitmentDataFilter,
    CommitmentList,
    Message, ObjectiveTotalsList, ObjectiveDataFilter, TimeSeriesResponse,
    PaginatedSankeyResponseSchema, KpiResponseSchema,
)
from src.services.fund_service import (
    get_commitments_data,
    insert_commitments_from_df, get_totals_by_objective, get_commitment_time_series, get_distinct_commitment_years,
    stream_commitments_csv, get_paginated_sankey_data, get_dashboard_kpis,
)
from src.utils.parser import read_file

router = APIRouter(prefix="/commitments", tags=["commitments"])

T_Session = Annotated[Session, Depends(get_session)]


class ObjectiveFilter(str, Enum):
    all = "all"
    adaptation = "adaptation"
    mitigation = "mitigation"
    both = "both"


class SankeyView(str, Enum):
    project_country_year = "project_country_year"
    project_year_country = "project_year_country"

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

@router.post("/time_series", response_model=TimeSeriesResponse)
def read_commitment_time_series(session: T_Session, filters: CommitmentDataFilter):
    """Retorna dados agregados para o gráfico de evolução de financiamento."""
    time_series_data = get_commitment_time_series(session, filters)
    return {"series": time_series_data}

@router.get("/years", response_model=List[int])
def read_distinct_commitment_years(session: T_Session):
    """Retorna uma lista de anos únicos em que ocorreram compromissos."""
    years = get_distinct_commitment_years(session)
    return years

@router.get("/export/{year}", response_class=StreamingResponse)
def export_commitments_by_year(year: int, session: T_Session):
    """
    Gera e faz o streaming de um arquivo CSV com todos os compromissos de um ano.
    """
    file_name = f"commitments_{year}.csv"
    headers = {
        "Content-Disposition": f"attachment; filename=\"{file_name}\""
    }

    # Retorna uma resposta de streaming que chama nosso gerador
    return StreamingResponse(
        stream_commitments_csv(session, year),
        media_type="text/csv",
        headers=headers
    )

@router.get(
    "/kpis",
    response_model=KpiResponseSchema,
    status_code=HTTPStatus.OK
)
def get_kpis(session: T_Session):
    """
    Retorna os KPIs principais (Nº de Projetos, Nº de Países).
    """
    try:
        return get_dashboard_kpis(session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/sankey_data",
    response_model=PaginatedSankeyResponseSchema,
    status_code=HTTPStatus.OK,
)
def get_sankey_diagram_data(
    session: T_Session,
    # --- Filtros ---
    years: Optional[List[int]] = Query(
        None, alias="year", description="Filtra por um ou mais anos"
    ),
    country_ids: Optional[List[int]] = Query(
        None, alias="country_id", description="Filtra por um ou mais IDs de países"
    ),
    project_ids: Optional[List[int]] = Query(
        None, alias="project_id", description="Filtra por um ou mais IDs de projetos"
    ),
    objective: ObjectiveFilter = Query(
        ObjectiveFilter.all,
        description="Filtra por objetivo do financiamento (padrão: 'all')",
    ),
    # --- Paginação ---
    limit: int = Query(
        5, description="Nº de projetos por página", ge=1, le=20
    ),
    offset: int = Query(0, description="Nº de projetos a pular", ge=0),
    # --- Controle de View ---
    view: SankeyView = Query(
        SankeyView.project_country_year,
        description="Define a ordem dos nós: 'project_country_year' (Padrão) ou 'project_year_country'",
    ),
):
    """
    Retorna os dados [from, to, weight, tooltip] para o diagrama Sankey.
    Os dados são PAGINADOS com base no ranking de 'total_amount' dos PROJETOS.
    Permite duas visões: Projeto->País->Ano ou Projeto->Ano->País.
    """
    try:
        sankey_page_data = get_paginated_sankey_data(
            session,
            filter_years=years,
            filter_country_ids=country_ids,
            filter_project_ids=project_ids,
            objective=objective.value,
            limit=limit,
            offset=offset,
            view=view.value,
        )
        return sankey_page_data
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ocorreu um erro ao buscar os dados: {str(e)}"
        )