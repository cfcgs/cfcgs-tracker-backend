import pytest

from src.cfcgs_tracker.adapters.llm.chatbot_agent import (
    UnsafeSQLQueryError,
    apply_limit_offset,
    has_explicit_limit,
    normalize_question_text,
    parse_router_response,
    sanitize_user_facing_answer,
    validate_safe_sql,
)


def test_parse_router_response_parses_sql_payload():
    response_type, content = parse_router_response(
        "[SQL] SELECT * FROM view_climate_finance_records_detailed"
    )

    assert response_type == "SQL"
    assert content == "SELECT * FROM view_climate_finance_records_detailed"


def test_has_explicit_limit_detects_limit_clause():
    assert has_explicit_limit("SELECT * FROM test LIMIT 10") is True
    assert has_explicit_limit("SELECT * FROM test") is False


def test_apply_limit_offset_wraps_query():
    paginated_query = apply_limit_offset(
        "SELECT * FROM view_provider_fund_profiles_detailed",
        limit=10,
        offset=20,
    )

    assert "LIMIT 10 OFFSET 20" in paginated_query
    assert "paginated_query" in paginated_query


def test_validate_safe_sql_accepts_allowed_views():
    sql = validate_safe_sql(
        """
        SELECT cfrd.project_title
        FROM view_climate_finance_records_detailed cfrd
        LEFT JOIN view_provider_fund_profiles_detailed vfpd
            ON vfpd.funding_provider_id = cfrd.funding_provider_id
        """
    )

    assert "view_climate_finance_records_detailed" in sql


def test_validate_safe_sql_accepts_cte_aliases():
    sql = validate_safe_sql(
        """
        WITH TopProject AS (
            SELECT
                cfrd.project_id,
                SUM(cfrd.total_amount_usd_millions) AS total_amount_usd_millions
            FROM view_climate_finance_records_detailed cfrd
            GROUP BY cfrd.project_id
        )
        SELECT tp.project_id
        FROM TopProject tp
        JOIN view_climate_finance_records_detailed cfrd
            ON cfrd.project_id = tp.project_id
        """
    )

    assert "WITH TopProject AS" in sql


def test_validate_safe_sql_rejects_non_select_query():
    with pytest.raises(UnsafeSQLQueryError):
        validate_safe_sql("DELETE FROM view_climate_finance_records_detailed")


def test_validate_safe_sql_rejects_disallowed_table():
    with pytest.raises(UnsafeSQLQueryError):
        validate_safe_sql("SELECT * FROM alembic_version")


def test_normalize_question_text_removes_accents():
    assert (
        normalize_question_text("Qual país mais recebeu?")
        == "qual pais mais recebeu?"
    )


def test_sanitize_user_facing_answer_removes_internal_terms():
    answer = sanitize_user_facing_answer(
        "A view view_provider_fund_profiles_detailed mostra dados dessa tabela."
    )

    assert "view_provider_fund_profiles_detailed" not in answer
    assert "tabela" not in answer.lower()
