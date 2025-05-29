from typing import Optional

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload
from src.cfcgs_tracker.models import Fund, FundType, FundFocus
import pandas as pd

from src.cfcgs_tracker.schemas import FundSchema, FundStatusFilter, FundDataFilter, FundTypeSchema, FundFocusSchema, \
    FundTypeUpdateSchema, FundUpdateSchema, FundFocusUpdateSchema
from src.utils.parser import safe_float, safe_int


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
        fund_focus = db.query(FundFocus).filter_by(name=fund_focus_name).first()
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


def get_fund_status(db: Session,
                    filters: FundStatusFilter):
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


def get_funds_data(db: Session, filters: FundDataFilter, limit: int, offset: int):
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
        .options(
            joinedload(Fund.fund_type),
            joinedload(Fund.fund_focus)
        )
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


def update_fund_by_id(db: Session, fund_id: int, fund_data: FundUpdateSchema) -> Optional[FundSchema]:
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
        .options(
            joinedload(Fund.fund_type),
            joinedload(Fund.fund_focus)
        )
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
        fund_type = db.query(FundType).filter_by(name=fund_data.fund_type).first()
        if fund_type:
            fund.fund_type_id = fund_type.id

    if fund_data.fund_focus:
        fund_focus = db.query(FundFocus).filter_by(name=fund_data.fund_focus).first()
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
        raise HTTPException(status_code=400, detail="Cannot delete fund type in use by funds")

    db.delete(fund_type)
    db.commit()
    return True


def update_fund_type_by_id(db: Session, type_id: int, type_data: FundTypeUpdateSchema) -> Optional[FundTypeSchema]:
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

    return FundTypeSchema(
        id=fund_type.id,
        name=fund_type.name
    )


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
        raise HTTPException(status_code=400, detail="Cannot delete fund focus in use by funds")

    db.delete(fund_focus)
    db.commit()
    return True


def update_fund_focus_by_id(db: Session, focus_id: int, focus_data: FundFocusUpdateSchema) -> Optional[FundFocusSchema]:
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

    return FundFocusSchema(
        id=fund_focus.id,
        name=fund_focus.name
    )