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
    HeatmapResponseSchema, HeatmapProjectsResponseSchema, HeatmapKpiResponseSchema, KpiResponseSchema,
    HeatmapFilterOptionsSchema,
)
from src.services.fund_service import (
    get_commitments_data,
    insert_commitments_from_df, get_totals_by_objective, get_commitment_time_series, get_distinct_commitment_years,
    stream_commitments_csv, get_heatmap_data, get_heatmap_cell_projects,
    get_dashboard_kpis, get_heatmap_kpis,
    get_heatmap_filter_options,
)
from src.utils.parser import read_file

router = APIRouter(prefix="/commitments", tags=["commitments"])

T_Session = Annotated[Session, Depends(get_session)]


class ObjectiveFilter(str, Enum):
    all = "all"
    adaptation = "adaptation"
    mitigation = "mitigation"
    both = "both"


class HeatmapView(str, Enum):
    country_year = "country_year"
    year_country = "year_country"

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
    "/heatmap_data",
    response_model=HeatmapResponseSchema,
    status_code=HTTPStatus.OK,
)
def get_heatmap_diagram_data(
    session: T_Session,
    years: Optional[List[int]] = Query(
        None, alias="year", description="Filtra por um ou mais anos"
    ),
    country_ids: Optional[List[int]] = Query(
        None, alias="country_id", description="Filtra por um ou mais IDs de paises"
    ),
    project_ids: Optional[List[int]] = Query(
        None, alias="project_id", description="Filtra por um ou mais IDs de projetos"
    ),
    objective: ObjectiveFilter = Query(
        ObjectiveFilter.all,
        description="Filtra por objetivo do financiamento (padrao: 'all')",
    ),
    view: HeatmapView = Query(
        HeatmapView.country_year,
        description="Define a ordem dos eixos: 'country_year' (Padrao) ou 'year_country'",
    ),
    row_offset: int = Query(0, description="Deslocamento de linhas", ge=0),
    row_limit: int = Query(30, description="Quantidade de linhas por janela", ge=1, le=200),
    column_offset: int = Query(0, description="Deslocamento de colunas", ge=0),
    column_limit: int = Query(30, description="Quantidade de colunas por janela", ge=1, le=200),
):
    """
    Retorna dados agregados para o heatmap (ano x pais), com totais e percentuais.
    """
    try:
        heatmap_data = get_heatmap_data(
            session,
            filter_years=years,
            filter_country_ids=country_ids,
            filter_project_ids=project_ids,
            objective=objective.value,
            view=view.value,
            row_offset=row_offset,
            row_limit=row_limit,
            column_offset=column_offset,
            column_limit=column_limit,
        )
        return heatmap_data
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ocorreu um erro ao buscar os dados: {str(e)}"
        )


@router.get(
    "/heatmap_kpis",
    response_model=HeatmapKpiResponseSchema,
    status_code=HTTPStatus.OK,
)
def get_heatmap_kpi_data(
    session: T_Session,
    years: Optional[List[int]] = Query(
        None, alias="year", description="Filtra por um ou mais anos"
    ),
    country_ids: Optional[List[int]] = Query(
        None, alias="country_id", description="Filtra por um ou mais IDs de paises"
    ),
    project_ids: Optional[List[int]] = Query(
        None, alias="project_id", description="Filtra por um ou mais IDs de projetos"
    ),
    objective: ObjectiveFilter = Query(
        ObjectiveFilter.all,
        description="Filtra por objetivo do financiamento (padrao: 'all')",
    ),
):
    """
    Retorna KPIs agregados para os filtros do heatmap.
    """
    try:
        return get_heatmap_kpis(
            session,
            filter_years=years,
            filter_country_ids=country_ids,
            filter_project_ids=project_ids,
            objective=objective.value,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ocorreu um erro ao buscar os KPIs: {str(e)}"
        )


@router.get(
    "/heatmap_projects",
    response_model=HeatmapProjectsResponseSchema,
    status_code=HTTPStatus.OK,
)
def get_heatmap_projects(
    session: T_Session,
    year: int = Query(..., description="Ano da celula selecionada"),
    country_id: int = Query(..., description="ID do pais da celula selecionada"),
    objective: ObjectiveFilter = Query(
        ObjectiveFilter.all,
        description="Filtra por objetivo do financiamento (padrao: 'all')",
    ),
    limit: int = Query(20, description="Nº de projetos por pagina", ge=1, le=200),
    offset: int = Query(0, description="Nº de projetos a pular", ge=0),
):
    """
    Retorna projetos paginados para uma celula (ano x pais).
    """
    try:
        return get_heatmap_cell_projects(
            session,
            year=year,
            country_id=country_id,
            objective=objective.value,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ocorreu um erro ao buscar os dados: {str(e)}"
        )


@router.get(
    "/heatmap_filters",
    response_model=HeatmapFilterOptionsSchema,
    status_code=HTTPStatus.OK,
)
def get_heatmap_filter_options_data(
    session: T_Session,
    years: Optional[List[int]] = Query(
        None, alias="year", description="Filtra por um ou mais anos"
    ),
    country_ids: Optional[List[int]] = Query(
        None, alias="country_id", description="Filtra por um ou mais IDs de paises"
    ),
    project_ids: Optional[List[int]] = Query(
        None, alias="project_id", description="Filtra por um ou mais IDs de projetos"
    ),
    objective: ObjectiveFilter = Query(
        ObjectiveFilter.all,
        description="Filtra por objetivo do financiamento (padrao: 'all')",
    ),
):
    """
    Retorna opções de filtros do heatmap com base nos filtros atuais.
    """
    try:
        return get_heatmap_filter_options(
            session,
            filter_years=years,
            filter_country_ids=country_ids,
            filter_project_ids=project_ids,
            objective=objective.value,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Ocorreu um erro ao buscar filtros: {str(e)}"
        )
