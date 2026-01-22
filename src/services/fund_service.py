import math
import threading
import time
from typing import Optional, List, Dict, Any

from fastapi import HTTPException
from sqlalchemy import func, select, desc
from sqlalchemy.orm import Session, joinedload
from src.cfcgs_tracker.models import (
    Fund,
    FundType,
    FundFocus,
    Project,
    Country,
    Region,
    Commitment,
    FundingEntity,
)
import pandas as pd

from src.cfcgs_tracker.schemas import (
    FundSchema,
    FundStatusFilter,
    FundDataFilter,
    FundTypeSchema,
    FundFocusSchema,
    FundTypeUpdateSchema,
    FundUpdateSchema,
    FundFocusUpdateSchema,
    FundProjectDataFilter,
    FundProjectSchema,
    CommitmentDataFilter,
    CommitmentDataSchema, CountrySchema, ObjectiveDataFilter, ObjectiveTotalSchema,
    KpiResponseSchema,
)
from src.utils.parser import safe_float, safe_int

HEATMAP_CACHE_TTL_SECONDS = 300
_HEATMAP_CACHE_LOCK = threading.Lock()
_HEATMAP_METADATA_CACHE: Dict[tuple, Dict[str, Any]] = {}


def invalidate_heatmap_cache() -> None:
    with _HEATMAP_CACHE_LOCK:
        _HEATMAP_METADATA_CACHE.clear()


def _make_heatmap_cache_key(
    filter_years: Optional[List[int]],
    filter_country_ids: Optional[List[int]],
    filter_project_ids: Optional[List[int]],
    objective: str,
) -> tuple:
    def _normalize(values: Optional[List[int]]) -> Optional[tuple]:
        if not values:
            return None
        return tuple(sorted(values))

    return (
        objective or "all",
        _normalize(filter_years),
        _normalize(filter_country_ids),
        _normalize(filter_project_ids),
    )


def _get_heatmap_cached_metadata(
    db: Session,
    filter_years: Optional[List[int]],
    filter_country_ids: Optional[List[int]],
    filter_project_ids: Optional[List[int]],
    objective: str,
) -> Dict[str, Any]:
    cache_key = _make_heatmap_cache_key(
        filter_years, filter_country_ids, filter_project_ids, objective
    )
    now = time.time()
    with _HEATMAP_CACHE_LOCK:
        cached = _HEATMAP_METADATA_CACHE.get(cache_key)
        if cached and now - cached["ts"] <= HEATMAP_CACHE_TTL_SECONDS:
            return cached["data"]

    objective_filtered_cte = _build_objective_filtered_commits_cte(
        filter_years=filter_years,
        filter_country_ids=filter_country_ids,
        filter_project_ids=filter_project_ids,
        objective=objective,
    )

    totals_query = select(
        func.coalesce(func.sum(objective_filtered_cte.c.sum_total), 0).label(
            "grand_total"
        ),
        func.coalesce(
            func.count(objective_filtered_cte.c.project_id.distinct()), 0
        ).label("grand_total_projects"),
    )
    totals = db.execute(totals_query).one()
    grand_total = float(totals.grand_total or 0)
    grand_total_projects = int(totals.grand_total_projects or 0)

    totals_by_country_query = (
        select(
            objective_filtered_cte.c.country_id,
            objective_filtered_cte.c.country_name,
            func.sum(objective_filtered_cte.c.sum_total).label("total_amount"),
            func.count(objective_filtered_cte.c.project_id.distinct()).label(
                "project_count"
            ),
        )
        .group_by(
            objective_filtered_cte.c.country_id,
            objective_filtered_cte.c.country_name,
        )
        .order_by(objective_filtered_cte.c.country_name)
    )
    totals_by_year_query = (
        select(
            objective_filtered_cte.c.year,
            func.sum(objective_filtered_cte.c.sum_total).label("total_amount"),
            func.count(objective_filtered_cte.c.project_id.distinct()).label(
                "project_count"
            ),
        )
        .group_by(objective_filtered_cte.c.year)
        .order_by(objective_filtered_cte.c.year)
    )

    totals_by_country = []
    for row in db.execute(totals_by_country_query).all():
        totals_by_country.append(
            {
                "country_id": int(row.country_id),
                "country_name": row.country_name,
                "total_amount": float(row.total_amount or 0),
                "project_count": int(row.project_count or 0),
            }
        )

    totals_by_year = []
    for row in db.execute(totals_by_year_query).all():
        totals_by_year.append(
            {
                "year": int(row.year),
                "total_amount": float(row.total_amount or 0),
                "project_count": int(row.project_count or 0),
            }
        )

    payload = {
        "grand_total": grand_total,
        "grand_total_projects": grand_total_projects,
        "totals_by_country": totals_by_country,
        "totals_by_year": totals_by_year,
    }

    with _HEATMAP_CACHE_LOCK:
        _HEATMAP_METADATA_CACHE[cache_key] = {"ts": now, "data": payload}

    return payload


def insert_funds_from_df(db: Session, df: pd.DataFrame):
    """
    Insere dados de fundos no banco a partir de um DataFrame, criando tipos e focos se necessário.

    Parâmetros:
    - db: Sessão do banco de dados.
    - df: DataFrame com os dados dos fundos.
    """
    for _, row in df.iterrows():
        # Buscar ou criar fund_type
        fund_type_name = row.get("fund_type")
        fund_type = db.query(FundType).filter_by(name=fund_type_name).first()
        if not fund_type:
            fund_type = FundType(name=fund_type_name)
            db.add(fund_type)
            db.flush()  # obtém o ID sem fazer commit

        # Buscar ou criar fund_focus
        fund_focus_name = row.get("fund_focus")
        fund_focus = (
            db.query(FundFocus).filter_by(name=fund_focus_name).first()
        )
        if not fund_focus:
            fund_focus = FundFocus(name=fund_focus_name)
            db.add(fund_focus)
            db.flush()

        # Buscar ou criar fund
        fund_name = row.get("fund")
        fund = db.query(Fund).filter_by(fund_name=fund_name).first()
        if not fund:
            fund = Fund(
                fund_name=fund_name,
                fund_type_id=fund_type.id,
                fund_focus_id=fund_focus.id,
                pledge=safe_float(row.get("pledge", 0)),
                deposit=safe_float(row.get("deposit", 0)),
                approval=safe_float(row.get("approval", 0)),
                disbursement=safe_float(row.get("disbursement", 0)),
                projects_approved=safe_int(row.get("projects_approved", 0)),
            )
            db.add(fund)
    db.commit()


def get_fund_status(db: Session, filters: FundStatusFilter):
    """
    Retorna a soma de pledge, deposit e approval dos fundos conforme filtros aplicados.

    Parâmetros:
    - db: Sessão do banco de dados.
    - filters: Filtros de fundos, tipos e focos.

    Retorna:
    - Linha com totais ou None.
    """
    query = db.query(
        func.sum(Fund.pledge).label("total_pledge"),
        func.sum(Fund.deposit).label("total_deposit"),
        func.sum(Fund.approval).label("total_approval"),
    )
    if filters.funds:
        query = query.filter(Fund.id.in_(filters.funds))
    if filters.fund_types:
        query = query.filter(Fund.fund_type_id.in_(filters.fund_types))
    if filters.fund_focuses:
        query = query.filter(Fund.fund_focus_id.in_(filters.fund_focuses))

    return query.one_or_none()


def get_funds_data(
    db: Session, filters: FundDataFilter, limit: int, offset: int
):
    """
    Retorna lista de fundos com filtros e paginação aplicados.

    Parâmetros:
    - db: Sessão do banco de dados.
    - filters: Filtros de tipo e foco.
    - limit: Máximo de resultados.
    - offset: Deslocamento (para paginação).

    Retorna:
    - Lista de FundSchema.
    """
    query = (
        select(Fund)
        .options(joinedload(Fund.fund_type), joinedload(Fund.fund_focus))
        .limit(limit)
        .offset(offset)
    )

    if filters.fund_types:
        query = query.filter(Fund.fund_type_id.in_(filters.fund_types))
    if filters.fund_focuses:
        query = query.filter(Fund.fund_focus_id.in_(filters.fund_focuses))

    result = db.scalars(query).all()

    funds_data = [
        FundSchema(
            id=fund.id,
            fund_name=fund.fund_name,
            fund_type=fund.fund_type.name if fund.fund_type else None,
            fund_focus=fund.fund_focus.name if fund.fund_focus else None,
            pledge=fund.pledge,
            deposit=fund.deposit,
            approval=fund.approval,
            disbursement=fund.disbursement,
            projects_approved=fund.projects_approved,
        )
        for fund in result
    ]

    return funds_data


def get_fund_types(db: Session):
    """Retorna todos os tipos de fundos cadastrados"""
    result = db.scalars(select(FundType).order_by(FundType.id)).all()
    return result


def get_fund_focuses(db: Session):
    """Retorna todos os focos dos fundos cadastrados"""
    result = db.scalars(select(FundFocus).order_by(FundFocus.id)).all()
    return result


def delete_fund_by_id(db: Session, fund_id: int) -> bool:
    """
    Deleta um fundo pelo ID.

    Parâmetros:
    - db: Sessão do banco de dados
    - fund_id: ID do fundo a ser deletado

    Retorna:
    - True se deletado, False se não encontrado
    """
    fund = db.get(Fund, fund_id)
    if not fund:
        return False

    db.delete(fund)
    db.commit()
    return True


def update_fund_by_id(
    db: Session, fund_id: int, fund_data: FundUpdateSchema
) -> Optional[FundSchema]:
    """
    Atualiza um fundo pelo ID.

    Parâmetros:
    - db: Sessão do banco de dados
    - fund_id: ID do fundo a ser atualizado
    - fund_data: Dados para atualização

    Retorna:
    - FundSchema atualizado ou None se não encontrado
    """
    fund = db.scalars(
        select(Fund)
        .where(Fund.id == fund_id)
        .options(joinedload(Fund.fund_type), joinedload(Fund.fund_focus))
    ).one_or_none()
    if not fund:
        return None

    # Atualiza os campos
    if fund_data.fund_name is not None:
        fund.fund_name = fund_data.fund_name
    if fund_data.pledge is not None:
        fund.pledge = fund_data.pledge
    if fund_data.deposit is not None:
        fund.deposit = fund_data.deposit
    if fund_data.approval is not None:
        fund.approval = fund_data.approval
    if fund_data.disbursement is not None:
        fund.disbursement = fund_data.disbursement
    if fund_data.projects_approved is not None:
        fund.projects_approved = fund_data.projects_approved

    # Atualiza relações se necessário
    if fund_data.fund_type:
        fund_type = (
            db.query(FundType).filter_by(name=fund_data.fund_type).first()
        )
        if fund_type:
            fund.fund_type_id = fund_type.id

    if fund_data.fund_focus:
        fund_focus = (
            db.query(FundFocus).filter_by(name=fund_data.fund_focus).first()
        )
        if fund_focus:
            fund.fund_focus_id = fund_focus.id

    db.commit()
    db.refresh(fund)

    fund_data_dict = {
        "id": getattr(fund, "id"),
        "fund_name": getattr(fund, "fund_name"),
        "fund_type": fund.fund_type.name if fund.fund_type else None,
        "fund_focus": fund.fund_focus.name if fund.fund_focus else None,
        "pledge": getattr(fund, "pledge"),
        "deposit": getattr(fund, "deposit"),
        "approval": getattr(fund, "approval"),
        "disbursement": getattr(fund, "disbursement"),
        "projects_approved": getattr(fund, "projects_approved"),
    }

    return FundSchema(**fund_data_dict)


def delete_fund_type_by_id(db: Session, type_id: int) -> bool:
    """
    Deleta um tipo de fundo pelo ID, se não estiver em uso.

    Parâmetros:
    - db: Sessão do banco de dados
    - type_id: ID do tipo a ser deletado

    Retorna:
    - True se deletado, False se não encontrado ou em uso
    """
    fund_type = db.get(FundType, type_id)
    if not fund_type:
        return False

    # Verifica se há fundos usando este tipo
    if db.query(Fund).filter_by(fund_type_id=type_id).count() > 0:
        raise HTTPException(
            status_code=400, detail="Cannot delete fund type in use by funds"
        )

    db.delete(fund_type)
    db.commit()
    return True


def update_fund_type_by_id(
    db: Session, type_id: int, type_data: FundTypeUpdateSchema
) -> Optional[FundTypeSchema]:
    """
    Atualiza um tipo de fundo pelo ID.

    Parâmetros:
    - db: Sessão do banco de dados
    - type_id: ID do tipo a ser atualizado
    - type_data: Dados para atualização

    Retorna:
    - FundTypeSchema atualizado ou None se não encontrado
    """
    fund_type = db.get(FundType, type_id)
    if not fund_type:
        return None

    fund_type.name = type_data.name
    db.commit()
    db.refresh(fund_type)

    return FundTypeSchema(id=fund_type.id, name=fund_type.name)


def delete_fund_focus_by_id(db: Session, focus_id: int) -> bool:
    """
    Deleta um foco de fundo pelo ID, se não estiver em uso.

    Parâmetros:
    - db: Sessão do banco de dados
    - focus_id: ID do foco a ser deletado

    Retorna:
    - True se deletado, False se não encontrado ou em uso
    """
    fund_focus = db.get(FundFocus, focus_id)
    if not fund_focus:
        return False

    # Verifica se há fundos usando este foco
    if db.query(Fund).filter_by(fund_focus_id=focus_id).count() > 0:
        raise HTTPException(
            status_code=400, detail="Cannot delete fund focus in use by funds"
        )

    db.delete(fund_focus)
    db.commit()
    return True


def update_fund_focus_by_id(
    db: Session, focus_id: int, focus_data: FundFocusUpdateSchema
) -> Optional[FundFocusSchema]:
    """
    Atualiza um foco de fundo pelo ID.

    Parâmetros:
    - db: Sessão do banco de dados
    - focus_id: ID do foco a ser atualizado
    - focus_data: Dados para atualização

    Retorna:
    - FundFocusSchema atualizado ou None se não encontrado
    """
    fund_focus = db.get(FundFocus, focus_id)
    if not fund_focus:
        return None

    fund_focus.name = focus_data.name
    db.commit()
    db.refresh(fund_focus)

    return FundFocusSchema(id=fund_focus.id, name=fund_focus.name)


def get_fund_projects_data(
    db: Session, filters: FundProjectDataFilter, limit: int, offset: int
):
    query = select(Project).options(
        joinedload(Project.country),
        joinedload(Project.fund),
    )

    if filters.funds:
        query = query.where(Project.fund_id.in_(filters.funds))

    if filters.countries:
        query = query.where(Project.country_id.in_(filters.countries))

    if filters.regions and filters.regions is not None:
        query = (
            query.join(Project.country)
            .join(Country.region)
            .where(Region.id.in_(filters.regions))
        )

    query = query.limit(limit).offset(offset)

    orm_projects = db.scalars(query).all()

    # --- TRANSFORMAÇÃO MANUAL E EXPLÍCITA ---
    # Converte cada objeto SQLAlchemy no schema Pydantic desejado
    response_schemas = []
    for project in orm_projects:
        schema_instance = FundProjectSchema(
            id=project.id,
            name=project.name,
            fund_name=project.fund.fund_name if project.fund else None,
            country_name=project.country.name if project.country else None,
            region=project.country.region.name
            if project.country and project.country.region
            else None,
        )
        response_schemas.append(schema_instance)

    return response_schemas


def insert_fund_project_from_df(db: Session, df: pd.DataFrame):
    try:
        # Cache para otimização
        existing_projects = {p.name: p for p in db.query(Project).all()}
        existing_countries = {c.name: c for c in db.query(Country).all()}
        existing_funds = {f.fund_name: f for f in db.query(Fund).all()}

        for _, row in df.iterrows():
            project_name = row.get("name_of_project")
            country_name = row.get("country")
            fund_name = row.get("fund")

            if (
                pd.isna(project_name)
                or pd.isna(country_name)
                or pd.isna(fund_name)
            ):
                continue

            # Se o projeto já existe, pule
            if project_name in existing_projects:
                continue

            # O fundo DEVE existir para projetos dos fundos
            fund = existing_funds.get(fund_name)
            if not fund:
                raise ValueError(
                    f"Fundo '{fund_name}' não encontrado para o projeto '{project_name}'."
                )

            # Get or Create para o País
            country = existing_countries.get(country_name)
            if not country:
                country = Country(name=country_name)
                db.add(country)
                db.flush()
                existing_countries[country_name] = country

            # Cria o novo projeto com a associação ao Fundo
            new_project = Project(name=project_name)
            new_project.country = country
            new_project.fund = fund
            db.add(new_project)
            existing_projects[project_name] = new_project

        db.commit()
        invalidate_heatmap_cache()

    except Exception as e:
        db.rollback()
        raise e


def get_or_create_country(
    name: str, db: Session, countries_cache: dict[str, Country]
) -> Country:
    """
    Busca um país pelo nome no cache/banco. Se não existir, cria um novo,
    associa a região (se encontrada) e o adiciona ao cache.
    """
    country = countries_cache.get(name)
    if not country:
        # Cria a nova instância do país
        country = Country(name=name)
        db.add(country)
        db.flush()
        # Atualiza o cache para que o novo país possa ser reutilizado no mesmo loop
        countries_cache[name] = country

    return country


def get_or_create_funding_entity(
    name: str, db: Session, entities_cache: dict[str, FundingEntity]
) -> FundingEntity:
    """
    Busca uma entidade financeira pelo nome no cache/banco.
    Se não existir, cria uma nova e a adiciona ao cache.
    """
    # Retorna None se o nome for inválido para não criar entidades vazias
    if not name or pd.isna(name):
        return None

    entity = entities_cache.get(name)
    if not entity:
        entity = FundingEntity(name=name)
        db.add(entity)
        db.flush()
        entities_cache[name] = entity  # Atualiza o cache com a nova entidade
    return entity

def insert_commitments_from_df(db: Session, df: pd.DataFrame):
    """Insere compromissos (da planilha CRDF), populando as tabelas de apoio."""
    try:
        # Cache de dados existentes
        existing_projects = {p.name: p for p in db.query(Project).all()}
        existing_entities = {c.name: c for c in db.query(FundingEntity).all()}
        existing_countries = {c.name: c for c in db.query(Country).all()}

        for _, row in df.iterrows():
            project_title = row.get("project_title")
            recipient_name = row.get("recipient")
            provider_name = row.get("provider")
            channel_name = row.get("channel_of_delivery")
            year = row.get("year")
            amount_str = row.get(
                "climate-related_development_finance_-_commitment_-_2023_usd_thousand"
            )
            adaptation_amount = safe_float(
                row.get("adaptation-related_development_finance_-_commitment_-_2023_usd_thousand"))
            mitigation_amount = safe_float(
                row.get("mitigation-related_development_finance_-_commitment_-_2023_usd_thousand"))
            overlap_amount = safe_float(
                row.get("overlap_-_commitment_-_2023_usd_thousand"))

            if any(
                pd.isna(val)
                for val in [recipient_name, provider_name, year, amount_str]
            ):
                continue

            provider_entity = get_or_create_funding_entity(
                provider_name, db, existing_entities
            )
            channel_entity = get_or_create_funding_entity(
                channel_name, db, existing_entities
            )

            recipient_country = get_or_create_country(
                recipient_name, db, existing_countries
            )

            # Get or Create - Projeto (com fund_id NULO)
            project = None  # O valor padrão para o projeto é None

            # Só executa a lógica de projeto se um título foi fornecido na planilha
            if pd.notna(project_title):
                project = existing_projects.get(project_title)
                if not project:
                    project = Project(name=project_title)
                    db.add(project)
                    db.flush()
                    existing_projects[project_title] = project

            # Criar o registro de Compromisso
            new_commitment = Commitment(
                year=int(year),
                amount_usd_thousand=safe_float(amount_str),
                adaptation_amount_usd_thousand=adaptation_amount,
                mitigation_amount_usd_thousand=mitigation_amount,
                overlap_amount_usd_thousand=overlap_amount
            )
            new_commitment.project = project
            new_commitment.channel = channel_entity
            new_commitment.provider = provider_entity
            new_commitment.recipient_country = recipient_country
            db.add(new_commitment)

        db.commit()

    except Exception as e:
        db.rollback()
        raise e

def get_commitments_data(
    db: Session, filters: CommitmentDataFilter, limit: int, offset: int
):
    """
    Busca e retorna uma lista de compromissos financeiros com base em filtros,
    com paginação e dados de relacionamentos formatados.
    """
    # 1. Construção da Query Base
    # Começamos selecionando a entidade principal (Commitment) e já pedimos
    # para carregar os dados relacionados para evitar múltiplas queries.
    query = select(Commitment).options(
        joinedload(Commitment.channel),
        joinedload(Commitment.provider),
        joinedload(Commitment.project),
    )

    # 2. Aplicação dos Filtros
    # Filtro por lista de anos
    if filters.years:
        query = query.where(Commitment.year.in_(filters.years))

    # Filtro por lista de nomes de países (provedores)
    if filters.countries:
        # Precisamos fazer um JOIN com a tabela Country para filtrar pelo nome
        query = query.join(Commitment.recipient_country).where(
            Country.id.in_(filters.countries)
        )

    # 3. Paginação
    query = query.limit(limit).offset(offset)

    # Executa a query para obter os objetos SQLAlchemy
    orm_commitments = db.scalars(query).all()

    # 4. Transformação Manual para o Schema de Resposta
    response_schemas = []
    for commitment in orm_commitments:
        schema_instance = CommitmentDataSchema(
            id=commitment.id,
            year=commitment.year,  # Converte o inteiro para string, como definido no schema
            amount_usd_thousand=commitment.amount_usd_thousand,
            # Acessa os nomes através dos objetos carregados pelo joinedload
            channel_of_delivery=commitment.channel.name
            if commitment.channel
            else None,
            provider_country=commitment.provider.name
            if commitment.provider
            else None,
            recipient_country=commitment.recipient_country.name
            if commitment.recipient_country
            else None,
            project=commitment.project.name if commitment.project else None,
        )
        response_schemas.append(schema_instance)

    return response_schemas


def get_regions(db: Session):
    """Retorna todos as regiões cadastrados"""
    result = db.scalars(select(Region).order_by(Region.id)).all()
    return result


def get_countries(db: Session):
    """
    Retorna todos os países cadastrados, formatados como CountrySchema.
    """
    # 1. Query otimizada para carregar a relação com Region (eager loading)
    query = select(Country).options(joinedload(Country.region)).order_by(Country.id)

    orm_countries = db.scalars(query).all()

    # 2. Transforma cada objeto SQLAlchemy no schema Pydantic correspondente
    response_schemas = []
    for country in orm_countries:
        schema_instance = CountrySchema(
            id=country.id,
            name=country.name,
            # Extrai o nome do objeto de relacionamento, com segurança
            region=country.region.name if country.region else None
        )
        response_schemas.append(schema_instance)

    return response_schemas


def get_recipient_countries(db: Session) -> list[CountrySchema]:
    """
    Retorna apenas os países que receberam pelo menos um compromisso financeiro,
    formatados como CountrySchema.
    """
    # A mágica está no .where(Country.commitments_as_recipient.any())
    # Isso filtra apenas os países que têm alguma entrada na relação 'commitments_as_recipient'
    query = (
        select(Country)
        .where(Country.commitments_as_recipient.any())
        .options(joinedload(Country.region))
        .order_by(Country.name)  # Ordena por nome para o dropdown
    )

    orm_countries = db.scalars(query).all()

    # Transforma os resultados para o schema Pydantic
    response_schemas = [
        CountrySchema(
            id=country.id,
            name=country.name,
            region=country.region.name if country.region else None
        )
        for country in orm_countries
    ]

    return response_schemas


def get_totals_by_objective(db: Session, filters: ObjectiveDataFilter):
    query = (
        select(
            Commitment.year,
            func.sum(func.coalesce(Commitment.adaptation_amount_usd_thousand, 0)).label("total_adaptation"),
            func.sum(func.coalesce(Commitment.mitigation_amount_usd_thousand, 0)).label("total_mitigation"),
            func.sum(func.coalesce(Commitment.overlap_amount_usd_thousand, 0)).label("total_overlap")
        )
        .group_by(Commitment.year)
        .order_by(Commitment.year)
    )

    if filters.years:
        query = query.where(Commitment.year.in_(filters.years))

    if filters.recipient_countries:
        query = query.where(Commitment.recipient_country_id.in_(filters.recipient_countries))

    result = db.execute(query).all()
    print(result)
    response_data = []
    for row in result:
        total_adaptation = row.total_adaptation
        total_mitigation = row.total_mitigation
        total_overlap = row.total_overlap

        response_data.append(
            ObjectiveTotalSchema(
                year=row.year,
                total_adaptation=0 if total_adaptation is None or math.isnan(total_adaptation) else total_adaptation,
                total_mitigation=0 if total_mitigation is None or math.isnan(total_mitigation) else total_mitigation,
                total_overlap=0 if total_overlap is None or math.isnan(total_overlap) else total_overlap,
            )
        )

    return response_data


def get_distinct_commitment_years(db: Session) -> list[int]:
    """Retorna uma lista de anos únicos da tabela de commitments, em ordem decrescente."""
    result = db.query(Commitment.year).distinct().order_by(Commitment.year.desc()).all()
    # O resultado é uma lista de tuplas (ex: [(2023,), (2022,)]), então extraímos o primeiro elemento.
    return [year[0] for year in result]


def get_commitment_time_series(db: Session, filters: CommitmentDataFilter):
    """
    Busca e agrega os dados de compromissos por ano e país receptor,
    retornando apenas os dados necessários para o gráfico de linhas.
    """
    # Query base para agregar
    query = (
        select(
            Commitment.year,
            Country.name.label("country_name"),
            func.sum(Commitment.amount_usd_thousand).label("total_amount")
        )
        .join(Commitment.project)
        .join(Commitment.recipient_country)
        .group_by(Commitment.year, Country.name)
    )

    if filters.years:
        query = query.where(Commitment.year.in_(filters.years))
    if filters.countries:
        query = query.where(Country.id.in_(filters.countries))

    result = db.execute(query).all()

    # Estrutura os dados para o frontend
    series_map = {}
    total_map = {}

    for row in result:
        # Adiciona ao total
        total_map[row.year] = total_map.get(row.year, 0) + row.total_amount

        # Adiciona à série do país
        if row.country_name not in series_map:
            series_map[row.country_name] = {}
        series_map[row.country_name][row.year] = row.total_amount

    # Formata a série "Total Agregado"
    total_series_data = [{"year": year, "amount": amount} for year, amount in total_map.items()]

    # Formata as séries por país
    country_series = [
        {"name": country, "data": [{"year": year, "amount": amount} for year, amount in data.items()]}
        for country, data in series_map.items()
    ]

    # Se nenhum país específico foi filtrado, retorne apenas o total.
    # Caso contrário, retorne as séries dos países.

    if not filters.countries:
        return [{"name": "Financiamento Total Agregado", "data": total_series_data}]
    else:
        return country_series


# src/services/fund_service.py
import csv
from io import StringIO

def stream_commitments_csv(db: Session, year: int):
    """
    Gera as linhas de um arquivo CSV para os compromissos de um ano específico.
    """
    output = StringIO()
    writer = csv.writer(output)

    # 1. Escreve o cabeçalho
    headers = ['ID', 'Year', 'Project', 'Provider', 'Channel', 'Recipient', 'Amount (USD K)']
    writer.writerow(headers)
    yield output.getvalue()
    output.seek(0)
    output.truncate(0)

    # 2. Busca os dados do banco
    commitments = db.query(Commitment).filter(Commitment.year == year).options(
        joinedload(Commitment.project).joinedload(Project.country),
        joinedload(Commitment.provider),
        joinedload(Commitment.channel),
        joinedload(Commitment.recipient_country)
    ).all()

    # 3. Escreve os dados linha por linha
    for c in commitments:
        row = [
            c.id,
            c.year,
            c.project.name if c.project else None,
            c.provider.name if c.provider else None,
            c.channel.name if c.channel else None,
            c.recipient_country.name if c.recipient_country else None,
            c.amount_usd_thousand
        ]
        writer.writerow(row)
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)


def get_dashboard_kpis(db: Session) -> KpiResponseSchema:
    """Retorna os KPIs principais do dashboard."""

    # KPI 1: Total de projetos (que têm financiamento)
    total_projects_query = (
        select(func.count(Project.id.distinct()))
        .join(Commitment, Project.id == Commitment.project_id)
        .filter(Commitment.amount_usd_thousand > 0)
    )
    total_projects = db.execute(total_projects_query).scalar_one_or_none() or 0

    total_recipient_countries_query = (
        select(func.count(Commitment.recipient_country_id.distinct()))
        .filter(Commitment.amount_usd_thousand > 0)
    )
    total_funded_countries = (
            db.execute(total_recipient_countries_query).scalar_one_or_none() or 0
    )

    return KpiResponseSchema(
        total_projects=total_projects,
        total_funded_countries=total_funded_countries,
    )


def _build_objective_filtered_commits_cte(
    filter_years: Optional[List[int]],
    filter_country_ids: Optional[List[int]],
    filter_project_ids: Optional[List[int]],
    objective: str,
):
    col_ada_raw = func.coalesce(Commitment.adaptation_amount_usd_thousand, 0)
    col_mit_raw = func.coalesce(Commitment.mitigation_amount_usd_thousand, 0)
    col_over_raw = func.coalesce(Commitment.overlap_amount_usd_thousand, 0)

    col_ada_ex_raw = (col_ada_raw - col_over_raw).label("adaptation_exclusive")
    col_mit_ex_raw = (col_mit_raw - col_over_raw).label("mitigation_exclusive")
    col_total_raw = (col_ada_ex_raw + col_mit_ex_raw + col_over_raw).label(
        "total_amount"
    )

    base_commits_query = (
        select(
            Project.id.label("project_id"),
            Project.name.label("project_name"),
            Country.id.label("country_id"),
            Country.name.label("country_name"),
            Commitment.year.label("year"),
            func.sum(col_ada_ex_raw).label("sum_ada_ex"),
            func.sum(col_mit_ex_raw).label("sum_mit_ex"),
            func.sum(col_over_raw).label("sum_overlap"),
            func.sum(col_total_raw).label("sum_total"),
        )
        .join(Project, Project.id == Commitment.project_id)
        .join(Country, Commitment.recipient_country_id == Country.id)
        .filter(Commitment.id.isnot(None))
        .filter(col_total_raw > 0)
    )

    if filter_years:
        base_commits_query = base_commits_query.filter(
            Commitment.year.in_(filter_years)
        )
    if filter_project_ids:
        base_commits_query = base_commits_query.filter(
            Project.id.in_(filter_project_ids)
        )
    if filter_country_ids:
        base_commits_query = base_commits_query.filter(
            Commitment.recipient_country_id.in_(filter_country_ids)
        )

    base_commits_query = base_commits_query.group_by(
        Project.id,
        Project.name,
        Country.id,
        Country.name,
        Commitment.year,
    )

    aggregated_commits_cte = base_commits_query.cte("aggregated_commits")

    objective_filtered_query = select(aggregated_commits_cte)

    if objective == "adaptation":
        objective_filtered_query = objective_filtered_query.filter(
            aggregated_commits_cte.c.sum_ada_ex > 0,
            aggregated_commits_cte.c.sum_mit_ex == 0,
            aggregated_commits_cte.c.sum_overlap == 0,
        )
    elif objective == "mitigation":
        objective_filtered_query = objective_filtered_query.filter(
            aggregated_commits_cte.c.sum_mit_ex > 0,
            aggregated_commits_cte.c.sum_ada_ex == 0,
            aggregated_commits_cte.c.sum_overlap == 0,
        )
    elif objective == "both":
        objective_filtered_query = objective_filtered_query.filter(
            aggregated_commits_cte.c.sum_overlap > 0
        )

    return objective_filtered_query.cte("objective_filtered_commits")


def _classify_project_objective(
    adaptation_exclusive: float,
    mitigation_exclusive: float,
    overlap: float,
) -> str:
    if overlap > 0 or (adaptation_exclusive > 0 and mitigation_exclusive > 0):
        return "both"
    if adaptation_exclusive > 0:
        return "adaptation"
    if mitigation_exclusive > 0:
        return "mitigation"
    return "unknown"


def get_heatmap_data(
    db: Session,
    filter_years: Optional[List[int]],
    filter_country_ids: Optional[List[int]],
    filter_project_ids: Optional[List[int]],
    objective: str,
    view: str,
    row_offset: int = 0,
    row_limit: int = 30,
    column_offset: int = 0,
    column_limit: int = 30,
) -> Dict[str, Any]:
    """
    Retorna dados agregados por ano e pais para um heatmap.
    """
    metadata = _get_heatmap_cached_metadata(
        db=db,
        filter_years=filter_years,
        filter_country_ids=filter_country_ids,
        filter_project_ids=filter_project_ids,
        objective=objective,
    )
    grand_total = float(metadata["grand_total"] or 0)
    grand_total_projects = int(metadata["grand_total_projects"] or 0)
    totals_by_country = metadata["totals_by_country"]
    totals_by_year = metadata["totals_by_year"]

    if grand_total == 0:
        return {
            "view": view,
            "rows": [],
            "columns": [],
            "row_totals": [],
            "column_totals": [],
            "cells": [],
            "grand_total": 0.0,
            "grand_total_projects": 0,
            "row_count": 0,
            "column_count": 0,
            "row_offset": row_offset,
            "column_offset": column_offset,
            "row_limit": row_limit,
            "column_limit": column_limit,
        }

    if view == "country_year":
        row_count = len(totals_by_country)
        column_count = len(totals_by_year)
    elif view == "year_country":
        row_count = len(totals_by_year)
        column_count = len(totals_by_country)
    else:
        raise HTTPException(status_code=400, detail="Visualizacao (view) invalida.")

    if row_count == 0 or column_count == 0:
        return {
            "view": view,
            "rows": [],
            "columns": [],
            "row_totals": [],
            "column_totals": [],
            "cells": [],
            "grand_total": grand_total,
            "grand_total_projects": grand_total_projects,
            "row_count": row_count,
            "column_count": column_count,
            "row_offset": row_offset,
            "column_offset": column_offset,
            "row_limit": row_limit,
            "column_limit": column_limit,
        }

    row_offset = max(row_offset, 0)
    column_offset = max(column_offset, 0)
    row_limit = max(row_limit, 1)
    column_limit = max(column_limit, 1)

    if row_offset >= row_count:
        row_offset = max(row_count - row_limit, 0)
    if column_offset >= column_count:
        column_offset = max(column_count - column_limit, 0)

    if view == "country_year":
        row_totals_results = totals_by_country[row_offset:row_offset + row_limit]
        column_totals_results = totals_by_year[column_offset:column_offset + column_limit]
    else:
        row_totals_results = totals_by_year[row_offset:row_offset + row_limit]
        column_totals_results = totals_by_country[column_offset:column_offset + column_limit]

    if view == "country_year":
        row_keys = [row["country_id"] for row in row_totals_results]
        column_keys = [row["year"] for row in column_totals_results]
    else:
        row_keys = [row["year"] for row in row_totals_results]
        column_keys = [row["country_id"] for row in column_totals_results]

    row_totals = []
    column_totals = []
    row_total_map = {}
    column_total_map = {}

    for row in row_totals_results:
        if view == "country_year":
            label = row["country_name"]
            row_key = row["country_id"]
            year = None
            country_id = row["country_id"]
        else:
            label = str(row["year"])
            row_key = row["year"]
            year = row["year"]
            country_id = None

        total_amount = float(row["total_amount"] or 0)
        project_count = int(row["project_count"] or 0)
        percent_of_total = (total_amount / grand_total) * 100 if grand_total else 0

        row_totals.append(
            {
                "label": label,
                "total_amount": total_amount,
                "project_count": project_count,
                "percent_of_total": percent_of_total,
                "country_id": country_id,
                "year": year,
            }
        )
        row_total_map[row_key] = total_amount

    for row in column_totals_results:
        if view == "country_year":
            label = str(row["year"])
            column_key = row["year"]
            year = row["year"]
            country_id = None
        else:
            label = row["country_name"]
            column_key = row["country_id"]
            year = None
            country_id = row["country_id"]

        total_amount = float(row["total_amount"] or 0)
        project_count = int(row["project_count"] or 0)
        percent_of_total = (total_amount / grand_total) * 100 if grand_total else 0

        column_totals.append(
            {
                "label": label,
                "total_amount": total_amount,
                "project_count": project_count,
                "percent_of_total": percent_of_total,
                "country_id": country_id,
                "year": year,
            }
        )
        column_total_map[column_key] = total_amount

    rows = [row["label"] for row in row_totals]
    columns = [column["label"] for column in column_totals]

    if not row_keys or not column_keys:
        return {
            "view": view,
            "rows": rows,
            "columns": columns,
            "row_totals": row_totals,
            "column_totals": column_totals,
            "cells": [],
            "grand_total": grand_total,
            "grand_total_projects": grand_total_projects,
            "row_count": row_count,
            "column_count": column_count,
            "row_offset": row_offset,
            "column_offset": column_offset,
            "row_limit": row_limit,
            "column_limit": column_limit,
        }

    objective_filtered_cte = _build_objective_filtered_commits_cte(
        filter_years=filter_years,
        filter_country_ids=filter_country_ids,
        filter_project_ids=filter_project_ids,
        objective=objective,
    )

    cell_query = (
        select(
            objective_filtered_cte.c.country_id,
            objective_filtered_cte.c.country_name,
            objective_filtered_cte.c.year,
            func.sum(objective_filtered_cte.c.sum_ada_ex).label(
                "adaptation_exclusive"
            ),
            func.sum(objective_filtered_cte.c.sum_mit_ex).label(
                "mitigation_exclusive"
            ),
            func.sum(objective_filtered_cte.c.sum_overlap).label("overlap"),
            func.sum(objective_filtered_cte.c.sum_total).label("total_amount"),
            func.count(objective_filtered_cte.c.project_id.distinct()).label(
                "project_count"
            ),
        )
        .group_by(
            objective_filtered_cte.c.country_id,
            objective_filtered_cte.c.country_name,
            objective_filtered_cte.c.year,
        )
    )

    if view == "country_year":
        cell_query = cell_query.filter(
            objective_filtered_cte.c.country_id.in_(row_keys),
            objective_filtered_cte.c.year.in_(column_keys),
        )
    else:
        cell_query = cell_query.filter(
            objective_filtered_cte.c.year.in_(row_keys),
            objective_filtered_cte.c.country_id.in_(column_keys),
        )

    cell_results = db.execute(cell_query).all()

    cells = []
    for row in cell_results:
        if view == "country_year":
            row_label = row.country_name
            column_label = str(row.year)
            row_key = row.country_id
            column_key = row.year
        else:
            row_label = str(row.year)
            column_label = row.country_name
            row_key = row.year
            column_key = row.country_id

        total_amount = float(row.total_amount or 0)
        row_total_amount = row_total_map.get(row_key, 0) or 0
        column_total_amount = column_total_map.get(column_key, 0) or 0

        cells.append(
            {
                "year": int(row.year),
                "country_id": int(row.country_id),
                "country_name": row.country_name,
                "row_label": row_label,
                "column_label": column_label,
                "total_amount": total_amount,
                "adaptation_exclusive": float(row.adaptation_exclusive or 0),
                "mitigation_exclusive": float(row.mitigation_exclusive or 0),
                "overlap": float(row.overlap or 0),
                "project_count": int(row.project_count or 0),
                "percent_of_total": (total_amount / grand_total) * 100
                if grand_total
                else 0,
                "percent_of_row": (total_amount / row_total_amount) * 100
                if row_total_amount
                else 0,
                "percent_of_column": (total_amount / column_total_amount) * 100
                if column_total_amount
                else 0,
            }
        )

    return {
        "view": view,
        "rows": rows,
        "columns": columns,
        "row_totals": row_totals,
        "column_totals": column_totals,
        "cells": cells,
        "grand_total": grand_total,
        "grand_total_projects": grand_total_projects,
        "row_count": row_count,
        "column_count": column_count,
        "row_offset": row_offset,
        "column_offset": column_offset,
        "row_limit": row_limit,
        "column_limit": column_limit,
    }


def get_heatmap_kpis(
    db: Session,
    filter_years: Optional[List[int]],
    filter_country_ids: Optional[List[int]],
    filter_project_ids: Optional[List[int]],
    objective: str,
) -> Dict[str, Any]:
    objective_filtered_cte = _build_objective_filtered_commits_cte(
        filter_years=filter_years,
        filter_country_ids=filter_country_ids,
        filter_project_ids=filter_project_ids,
        objective=objective,
    )

    totals_query = select(
        func.coalesce(func.sum(objective_filtered_cte.c.sum_total), 0).label(
            "total_amount"
        ),
        func.coalesce(func.sum(objective_filtered_cte.c.sum_ada_ex), 0).label(
            "total_adaptation"
        ),
        func.coalesce(func.sum(objective_filtered_cte.c.sum_mit_ex), 0).label(
            "total_mitigation"
        ),
        func.coalesce(func.sum(objective_filtered_cte.c.sum_overlap), 0).label(
            "total_overlap"
        ),
        func.coalesce(
            func.count(objective_filtered_cte.c.project_id.distinct()), 0
        ).label("total_projects"),
        func.coalesce(
            func.count(objective_filtered_cte.c.country_id.distinct()), 0
        ).label("total_countries"),
    )

    totals = db.execute(totals_query).one()
    return {
        "total_projects": int(totals.total_projects or 0),
        "total_countries": int(totals.total_countries or 0),
        "total_amount": float(totals.total_amount or 0),
        "total_adaptation": float(totals.total_adaptation or 0),
        "total_mitigation": float(totals.total_mitigation or 0),
        "total_overlap": float(totals.total_overlap or 0),
    }


def get_heatmap_cell_projects(
    db: Session,
    year: int,
    country_id: int,
    objective: str,
    limit: int,
    offset: int,
) -> Dict[str, Any]:
    objective_filtered_cte = _build_objective_filtered_commits_cte(
        filter_years=[year],
        filter_country_ids=[country_id],
        filter_project_ids=None,
        objective=objective,
    )

    total_query = select(
        func.coalesce(
            func.count(objective_filtered_cte.c.project_id.distinct()), 0
        ).label("total")
    )
    total = int(db.execute(total_query).scalar_one_or_none() or 0)

    if total == 0:
        return {"total": 0, "has_more": False, "projects": []}

    projects_query = (
        select(
            objective_filtered_cte.c.project_id,
            objective_filtered_cte.c.project_name,
            objective_filtered_cte.c.sum_ada_ex.label("adaptation_exclusive"),
            objective_filtered_cte.c.sum_mit_ex.label("mitigation_exclusive"),
            objective_filtered_cte.c.sum_overlap.label("overlap"),
            objective_filtered_cte.c.sum_total.label("total_amount"),
        )
        .order_by(desc(objective_filtered_cte.c.sum_total))
        .limit(limit)
        .offset(offset)
    )
    projects_results = db.execute(projects_query).all()

    projects = []
    for row in projects_results:
        adaptation_exclusive = float(row.adaptation_exclusive or 0)
        mitigation_exclusive = float(row.mitigation_exclusive or 0)
        overlap = float(row.overlap or 0)
        objective_label = _classify_project_objective(
            adaptation_exclusive,
            mitigation_exclusive,
            overlap,
        )
        projects.append(
            {
                "id": int(row.project_id),
                "name": row.project_name,
                "objective": objective_label,
                "total_amount": float(row.total_amount or 0),
                "adaptation_exclusive": adaptation_exclusive,
                "mitigation_exclusive": mitigation_exclusive,
                "overlap": overlap,
            }
        )

    has_more = offset + len(projects) < total

    return {"total": total, "has_more": has_more, "projects": projects}


def get_commitment_projects(db: Session) -> list[dict]:
    """Retorna uma lista de projetos (id, name) que possuem pelo menos um compromisso."""

    # Esta query busca IDs e Nomes distintos de Projetos
    # que têm uma correspondência na tabela de Compromissos (Commitment).
    query = (
        select(Project.id, Project.name)
        .join(Commitment, Project.id == Commitment.project_id)
        .distinct()
        .order_by(Project.name)
    )

    results = db.execute(query).all()

    # Converte a lista de (Row) para uma lista de dicts
    return [{"id": row.id, "name": row.name} for row in results]

async def get_paginated_commitment_projects(
    db: Session,
    search: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Busca projetos (que têm compromissos) de forma paginada,
    com filtro de busca opcional.
    """

    # Query base: Projetos distintos que estão em Commitments
    base_query = (
        select(Project.id, Project.name)
        .join(Commitment, Project.id == Commitment.project_id)
        .distinct()
    )

    if search:
        # Adiciona filtro (case-insensitive)
        base_query = base_query.filter(Project.name.ilike(f"%{search}%"))

    # Clona a query base para fazer a contagem total (sem limit/offset)
    count_query = select(func.count()).select_from(base_query.alias())
    total = db.execute(count_query).scalar_one_or_none() or 0

    # Aplica ordenação, paginação e executa
    paginated_query = base_query.order_by(Project.name).limit(limit).offset(offset)
    results = db.execute(paginated_query).all()

    projects = [{"id": row.id, "name": row.name} for row in results]

    return {
        "projects": projects,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + len(projects)) < total
    }
