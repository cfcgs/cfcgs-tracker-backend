from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.adapters.llm.prompts import (
    CHATBOT_SCHEMA_DESCRIPTION,
    FINAL_ANSWER_PROMPT_TEMPLATE,
    ROUTER_PROMPT_TEMPLATE,
    SQL_EXAMPLES,
)
from src.cfcgs_tracker.settings import Settings

FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|copy|grant|revoke)\b",
    re.IGNORECASE,
)
LIMIT_PATTERN = re.compile(r"\blimit\s+\d+\b", re.IGNORECASE)
TAGGED_RESPONSE_PATTERN = re.compile(
    r"^\s*\[(SQL|DIRECT|REFUSAL)\]\s*(.*)\s*$",
    re.IGNORECASE | re.DOTALL,
)
TAG_MARKER_PATTERN = re.compile(
    r"\[(SQL|DIRECT|REFUSAL)\]",
    re.IGNORECASE,
)
ALLOWED_SQL_OBJECTS = {
    "view_climate_finance_records_detailed",
    "view_provider_fund_profiles_detailed",
}
DEFAULT_SOURCE_LABELS = [
    {
        "name": "OECD",
        "url": "https://www.oecd.org/",
    },
    {
        "name": "Climate Funds Update",
        "url": "https://climatefundsupdate.org/",
    },
    {
        "name": "IDB",
        "url": "https://www.iadb.org/en",
    },
]
logger = logging.getLogger("uvicorn.error")
SUMMARY_MAX_CHARS = 900
ANSWER_SNIPPET_MAX_CHARS = 240
SQL_SNIPPET_MAX_CHARS = 320
MEMORY_CONTEXT_SNIPPET_MAX_CHARS = 180
INTERNAL_TERMS_PATTERN = re.compile(
    r"\b(view_[a-zA-Z0-9_]+|table|tables|tabela|tabelas|column|columns|coluna|colunas|schema|sql|database|banco de dados)\b",
    re.IGNORECASE,
)
COUNTRY_ALIASES = {
    "brasil": ["Brazil", "Brasil"],
    "bolivia": ["Bolivia", "Bolívia"],
    "africa do sul": ["South Africa", "África do Sul"],
    "africa subsaariana": [
        "Sub-Saharan Africa",
        "África Subsaariana",
    ],
}


class ChatbotGraphState(TypedDict, total=False):
    question: str
    page: int
    page_size: int
    confirm_pagination: bool
    disambiguation_choice: dict[str, str] | None
    answer: str
    needs_pagination_confirmation: bool
    pagination: dict[str, Any] | None
    sources: list[dict[str, str]] | None
    disambiguation: dict[str, Any] | None
    summary: str
    last_question: str
    last_answer: str
    last_sql: str | None
    pagination_request: dict[str, Any] | None
    last_used_at: float


class UnsafeSQLQueryError(Exception):
    """Raised when the generated SQL is unsafe or invalid."""


def extract_tagged_response(response_text: str) -> tuple[str, str]:
    match = TAGGED_RESPONSE_PATTERN.match(response_text.strip())
    if not match:
        raise ValueError("Resposta do modelo sem tag esperada.")
    response_type = match.group(1).upper()
    content = match.group(2).strip()
    nested_tags = TAG_MARKER_PATTERN.findall(content)
    if nested_tags:
        raise ValueError(
            "Faça apenas uma pergunta analítica por vez para eu responder "
            "com precisão."
        )
    return response_type, content


def normalize_sql(sql: str) -> str:
    return sql.strip().rstrip(";")


def has_explicit_limit(sql: str) -> bool:
    return bool(LIMIT_PATTERN.search(sql))


def apply_limit_offset(sql: str, *, limit: int, offset: int) -> str:
    normalized = normalize_sql(sql)
    return (
        "SELECT * FROM ("
        f"{normalized}"
        ") AS paginated_query "
        f"LIMIT {int(limit)} OFFSET {int(offset)}"
    )


def validate_safe_sql(sql: str) -> str:
    normalized = normalize_sql(sql)
    lowered = normalized.lower()
    lowered_without_wrapping = lowered.lstrip("(\n\r\t ")

    if not (
        lowered.startswith("select")
        or lowered.startswith("with")
        or lowered_without_wrapping.startswith("select")
        or lowered_without_wrapping.startswith("with")
    ):
        raise UnsafeSQLQueryError("A consulta deve ser apenas de leitura.")

    if FORBIDDEN_SQL_PATTERN.search(normalized):
        raise UnsafeSQLQueryError(
            "A consulta gerada contém operação proibida."
        )

    cte_names = {
        match.group(1).lower()
        for match in re.finditer(
            r"(?:with|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(",
            normalized,
            flags=re.IGNORECASE,
        )
    }

    referenced_objects = re.findall(
        r"\b(?:from|join)\s+([a-zA-Z0-9_\.\"]+)",
        normalized,
        flags=re.IGNORECASE,
    )
    for object_name in referenced_objects:
        cleaned_name = object_name.replace('"', "").split(".")[-1]
        if cleaned_name.lower() in cte_names:
            continue
        if cleaned_name.lower() not in ALLOWED_SQL_OBJECTS:
            raise UnsafeSQLQueryError(
                f"Objeto não permitido na consulta: {cleaned_name}"
            )

    return normalized


def truncate_text(value: str | None, *, max_chars: int) -> str:
    if not value:
        return ""
    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 1]}…"


def sanitize_user_facing_answer(value: str) -> str:
    sanitized = INTERNAL_TERMS_PATTERN.sub("", value)
    sanitized = re.sub(r"\s{2,}", " ", sanitized)
    sanitized = re.sub(r"\s+([,.;:])", r"\1", sanitized)
    return sanitized.strip()


def normalize_question_text(value: str) -> str:
    normalized = value.lower()
    normalized = normalized.replace("á", "a").replace("à", "a")
    normalized = normalized.replace("ã", "a").replace("â", "a")
    normalized = normalized.replace("é", "e").replace("ê", "e")
    normalized = normalized.replace("í", "i")
    normalized = normalized.replace("ó", "o").replace("ô", "o")
    normalized = normalized.replace("õ", "o")
    normalized = normalized.replace("ú", "u")
    normalized = normalized.replace("ç", "c")
    return " ".join(normalized.split())


class ClimateFinanceChatbotAgent:
    _checkpointer = MemorySaver()

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.settings = Settings()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=self.settings.GEMINI_API_KEY,
        )
        self.router_chain = (
            ChatPromptTemplate.from_template(ROUTER_PROMPT_TEMPLATE)
            | self.llm
            | StrOutputParser()
        )
        self.final_chain = (
            ChatPromptTemplate.from_template(FINAL_ANSWER_PROMPT_TEMPLATE)
            | self.llm
            | StrOutputParser()
        )
        self.graph = self._build_graph()

    async def ask(
        self,
        *,
        question: str,
        session_id: str = "default",
        page: int = 1,
        page_size: int = 10,
        confirm_pagination: bool = False,
        disambiguation_choice: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        result = await self.graph.ainvoke(
            {
                "question": question,
                "page": page,
                "page_size": page_size,
                "confirm_pagination": confirm_pagination,
                "disambiguation_choice": disambiguation_choice,
            },
            config={"configurable": {"thread_id": session_id}},
        )

        return self._build_response(
            answer=result["answer"],
            needs_pagination_confirmation=result.get(
                "needs_pagination_confirmation", False
            ),
            pagination=result.get("pagination"),
            sources=result.get("sources"),
        )

    def _build_graph(self):
        workflow = StateGraph(ChatbotGraphState)
        workflow.add_node("run_turn", self._run_turn)
        workflow.add_edge(START, "run_turn")
        workflow.add_edge("run_turn", END)
        return workflow.compile(checkpointer=self._checkpointer)

    async def _run_turn(
        self,
        state: ChatbotGraphState,
    ) -> ChatbotGraphState:
        question = state["question"]
        page = state.get("page", 1)
        page_size = state.get("page_size", 10)
        confirm_pagination = state.get("confirm_pagination", False)
        pagination_request = state.get("pagination_request")

        if confirm_pagination and pagination_request:
            return await self._handle_pagination_confirmation(
                state=state,
                page=page,
                page_size=page_size,
                pagination_request=pagination_request,
            )

        shortcut_sql = self._build_shortcut_sql(question)
        if shortcut_sql:
            logger.info(
                "[chatbot] using shortcut sql | question=%s | sql=%s",
                question,
                shortcut_sql,
            )
            return await self._answer_from_sql(
                state=state,
                question=question,
                sql=shortcut_sql,
                page=page,
                page_size=page_size,
            )

        memory_context = self._format_memory_context(state)
        raw_response = await self.router_chain.ainvoke(
            {
                "schema": CHATBOT_SCHEMA_DESCRIPTION,
                "chat_history": memory_context,
                "question": question,
                "examples": SQL_EXAMPLES,
            }
        )
        logger.info(
            "[chatbot] llm raw response | question=%s | response=%s",
            question,
            truncate_text(raw_response, max_chars=600),
        )
        response_type, content = extract_tagged_response(raw_response)

        if response_type == "REFUSAL":
            return self._build_state_update(
                state=state,
                question=question,
                answer=sanitize_user_facing_answer(content),
                sql=None,
                pagination=None,
                needs_pagination_confirmation=False,
                pagination_request=None,
                sources=None,
            )

        if response_type == "DIRECT":
            return self._build_state_update(
                state=state,
                question=question,
                answer=sanitize_user_facing_answer(content),
                sql=None,
                pagination=None,
                needs_pagination_confirmation=False,
                pagination_request=None,
                sources=None,
            )

        sql = validate_safe_sql(content)
        logger.info(
            "[chatbot] generated sql | question=%s | sql=%s", question, sql
        )
        return await self._answer_from_sql(
            state=state,
            question=question,
            sql=sql,
            page=page,
            page_size=page_size,
        )

    async def _handle_pagination_confirmation(
        self,
        *,
        state: ChatbotGraphState,
        page: int,
        page_size: int,
        pagination_request: dict[str, Any],
    ) -> ChatbotGraphState:
        effective_page_size = int(
            pagination_request.get("page_size") or page_size
        )
        sql = pagination_request["sql"]
        rows = await self._fetch_rows(
            sql,
            limit=effective_page_size,
            offset=max(page - 1, 0) * effective_page_size,
            paginate=True,
        )
        answer = f"Exibindo página {page} de resultados para a sua consulta."
        return self._build_state_update(
            state=state,
            question=pagination_request["question"],
            answer=sanitize_user_facing_answer(answer),
            sql=sql,
            pagination=self._build_pagination_payload(
                page=page,
                page_size=effective_page_size,
                total_rows=int(pagination_request["total_rows"]),
                rows=rows,
            ),
            needs_pagination_confirmation=False,
            pagination_request=pagination_request,
            sources=self._infer_sources(sql),
        )

    async def _count_rows(self, sql: str) -> int:
        count_sql = (
            f"SELECT COUNT(*) AS total_rows FROM ({sql}) AS count_subquery"
        )
        result = await self.session.execute(text(count_sql))
        total_rows = int(result.scalar() or 0)
        logger.info(
            "[chatbot] count rows | sql=%s | total_rows=%s",
            truncate_text(sql, max_chars=600),
            total_rows,
        )
        return total_rows

    async def _fetch_rows(
        self,
        sql: str,
        *,
        limit: int,
        offset: int,
        paginate: bool,
    ) -> list[dict[str, Any]]:
        executable_sql = (
            apply_limit_offset(sql, limit=limit, offset=offset)
            if paginate
            else sql
        )
        result = await self.session.execute(text(executable_sql))
        rows = [dict(row) for row in result.mappings().all()]
        logger.info(
            "[chatbot] fetched rows | sql=%s | row_count=%s | sample=%s",
            truncate_text(executable_sql, max_chars=600),
            len(rows),
            truncate_text(
                json.dumps(rows[:3], ensure_ascii=False, default=str),
                max_chars=800,
            ),
        )
        return rows

    async def _build_final_answer(
        self,
        *,
        question: str,
        query: str,
        rows: list[dict[str, Any]],
    ) -> str:
        if not rows:
            return sanitize_user_facing_answer(
                "Não encontrei resultados para sua consulta."
            )

        if len(rows) > 10:
            return sanitize_user_facing_answer(
                "Encontrei vários resultados para a sua consulta. "
                "Use a paginação para navegar pelos dados retornados."
            )

        answer = await self.final_chain.ainvoke(
            {
                "question": question,
                "query": query,
                "response": json.dumps(rows, ensure_ascii=False, default=str),
            }
        )
        return sanitize_user_facing_answer(answer)

    async def _answer_from_sql(
        self,
        *,
        state: ChatbotGraphState,
        question: str,
        sql: str,
        page: int,
        page_size: int,
    ) -> ChatbotGraphState:
        try:
            total_rows = await self._count_rows(sql)
            sources = self._infer_sources(sql)

            if total_rows > page_size and not has_explicit_limit(sql):
                answer = (
                    "Encontrei muitos resultados para essa consulta. "
                    f"Deseja que eu mostre os dados paginados em blocos de {page_size} linhas?"
                )
                return self._build_state_update(
                    state=state,
                    question=question,
                    answer=sanitize_user_facing_answer(answer),
                    sql=sql,
                    pagination=None,
                    needs_pagination_confirmation=True,
                    pagination_request={
                        "question": question,
                        "sql": sql,
                        "total_rows": total_rows,
                        "page_size": page_size,
                    },
                    sources=sources,
                )

            rows = await self._fetch_rows(
                sql,
                limit=page_size,
                offset=max(page - 1, 0) * page_size,
                paginate=not has_explicit_limit(sql),
            )
            answer = await self._build_final_answer(
                question=question,
                query=sql,
                rows=rows,
            )

            pagination = None
            if not has_explicit_limit(sql) and total_rows > page_size:
                pagination = self._build_pagination_payload(
                    page=page,
                    page_size=page_size,
                    total_rows=total_rows,
                    rows=rows,
                )

            return self._build_state_update(
                state=state,
                question=question,
                answer=sanitize_user_facing_answer(answer),
                sql=sql,
                pagination=pagination,
                needs_pagination_confirmation=False,
                pagination_request=None,
                sources=sources,
            )
        except (ProgrammingError, DBAPIError) as exc:
            logger.warning(
                "[chatbot] invalid generated sql | question=%s | sql=%s | detail=%s",
                question,
                truncate_text(sql, max_chars=600),
                str(exc),
            )
            raise ValueError(
                "Não consegui responder essa pergunta do jeito que ela foi "
                "formulada. Tente dividir em perguntas menores ou ser mais "
                "específico."
            ) from exc

    def _build_shortcut_sql(self, question: str) -> str | None:
        normalized = normalize_question_text(question)

        if (
            "setor" in normalized
            and "subsetor" in normalized
            and "mais recebeu" in normalized
        ):
            return """
SELECT
    cfrd.sector_name,
    cfrd.sub_sector_name,
    SUM(cfrd.total_amount_usd_millions) AS total_amount_usd_millions
FROM view_climate_finance_records_detailed cfrd
WHERE cfrd.sector_name IS NOT NULL
  AND cfrd.sub_sector_name IS NOT NULL
GROUP BY cfrd.sector_name, cfrd.sub_sector_name
ORDER BY total_amount_usd_millions DESC NULLS LAST
LIMIT 1
""".strip()

        if (
            ("pais" in normalized or "país" in question.lower())
            and "mais recebeu" in normalized
            and "financiamento" in normalized
        ):
            return """
SELECT
    cfrd.beneficiary_country_name,
    SUM(cfrd.total_amount_usd_millions) AS total_amount_usd_millions
FROM view_climate_finance_records_detailed cfrd
WHERE cfrd.beneficiary_country_name IS NOT NULL
  AND cfrd.beneficiary_country_name NOT ILIKE 'Global%'
  AND cfrd.beneficiary_country_name NOT ILIKE 'Multi-country%'
  AND cfrd.beneficiary_country_name NOT ILIKE 'Regional%'
GROUP BY cfrd.beneficiary_country_name
ORDER BY total_amount_usd_millions DESC NULLS LAST
LIMIT 1
""".strip()

        country_match = re.search(
            r"quanto\s+(?:o|a)?\s*(?P<country>.+?)\s+recebeu",
            normalized,
        )
        if country_match and "financiamento" in normalized:
            country_key = country_match.group("country").strip()
            aliases = COUNTRY_ALIASES.get(country_key)
            if aliases:
                clauses = [
                    f"cfrd.beneficiary_country_name ILIKE '{alias}'"
                    for alias in aliases
                ]
            else:
                original = country_match.group("country").strip().title()
                clauses = [f"cfrd.beneficiary_country_name ILIKE '{original}'"]
            where_clause = " OR ".join(clauses)
            return f"""
SELECT
    SUM(cfrd.total_amount_usd_millions) AS total_amount_usd_millions
FROM view_climate_finance_records_detailed cfrd
WHERE ({where_clause})
""".strip()

        return None

    def _infer_sources(self, sql: str) -> list[dict[str, str]] | None:
        lowered = sql.lower()
        if "view_climate_finance_records_detailed" in lowered:
            return DEFAULT_SOURCE_LABELS
        return None

    def _build_pagination_payload(
        self,
        *,
        page: int,
        page_size: int,
        total_rows: int,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "has_more": page * page_size < total_rows,
            "rows": rows,
        }

    def _build_response(
        self,
        *,
        answer: str,
        needs_pagination_confirmation: bool = False,
        pagination: dict[str, Any] | None = None,
        sources: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        return {
            "answer": answer,
            "needs_pagination_confirmation": (needs_pagination_confirmation),
            "pagination": pagination,
            "sources": sources,
            "disambiguation": None,
        }

    def _build_state_update(
        self,
        *,
        state: ChatbotGraphState,
        question: str,
        answer: str,
        sql: str | None,
        pagination: dict[str, Any] | None,
        needs_pagination_confirmation: bool,
        pagination_request: dict[str, Any] | None,
        sources: list[dict[str, str]] | None,
    ) -> ChatbotGraphState:
        summary = self._update_summary(
            state.get("summary", ""),
            question=question,
            answer=answer,
            sql=sql,
        )
        return {
            "answer": answer,
            "needs_pagination_confirmation": (needs_pagination_confirmation),
            "pagination": pagination,
            "sources": sources,
            "disambiguation": None,
            "summary": summary,
            "last_question": question,
            "last_answer": answer,
            "last_sql": sql,
            "pagination_request": pagination_request,
            "last_used_at": time.time(),
        }

    def _format_memory_context(self, state: ChatbotGraphState) -> str:
        parts: list[str] = []
        summary = state.get("summary")
        if summary:
            parts.append(
                "Resumo curto da conversa até agora:\n"
                f"{truncate_text(summary, max_chars=SUMMARY_MAX_CHARS)}"
            )

        last_question = state.get("last_question")
        if last_question:
            parts.append(
                "Última pergunta do usuário:\n"
                f"{truncate_text(last_question, max_chars=MEMORY_CONTEXT_SNIPPET_MAX_CHARS)}"
            )

        last_answer = state.get("last_answer")
        if last_answer:
            parts.append(
                "Última resposta do assistente:\n"
                f"{truncate_text(last_answer, max_chars=MEMORY_CONTEXT_SNIPPET_MAX_CHARS)}"
            )

        last_sql = state.get("last_sql")
        if last_sql:
            parts.append(
                "Última SQL executada:\n"
                f"{truncate_text(last_sql, max_chars=SQL_SNIPPET_MAX_CHARS)}"
            )

        if not parts:
            return "Nenhum contexto anterior salvo."
        return "\n\n".join(parts)

    def _update_summary(
        self,
        current_summary: str,
        *,
        question: str,
        answer: str,
        sql: str | None,
    ) -> str:
        snippets = [
            f"Pergunta: {truncate_text(question, max_chars=120)}",
            f"Resposta: {truncate_text(answer, max_chars=ANSWER_SNIPPET_MAX_CHARS)}",
        ]
        if sql:
            snippets.append(
                f"SQL: {truncate_text(sql, max_chars=SQL_SNIPPET_MAX_CHARS)}"
            )
        next_summary = "\n".join(
            part
            for part in [current_summary.strip(), " | ".join(snippets)]
            if part
        )
        return truncate_text(next_summary, max_chars=SUMMARY_MAX_CHARS)
