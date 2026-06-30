from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, TypedDict

from fuzzywuzzy import fuzz, process
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.adapters.llm.prompts import (
    FINAL_ANSWER_PROMPT_TEMPLATE,
    build_router_prompt,
    get_table_details,
)
from src.cfcgs_tracker.settings import Settings

FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|copy|grant|revoke)\b",
    re.IGNORECASE,
)
LIMIT_PATTERN = re.compile(r"\blimit\s+\d+\b", re.IGNORECASE)
TAGGED_RESPONSE_PATTERN = re.compile(
    r"^\s*\[(SQL|DIRECT|REFUSAL|ENTITY_RESOLUTION)\]\s*(.*)\s*$",
    re.IGNORECASE | re.DOTALL,
)
TAG_MARKER_PATTERN = re.compile(
    r"\[(SQL|DIRECT|REFUSAL|ENTITY_RESOLUTION)\]",
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


class ChatbotGraphState(TypedDict, total=False):
    question: str
    page: int
    page_size: int
    confirm_pagination: bool
    disambiguation_choice: dict[str, Any] | None
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


def parse_router_response(
    response_text: str,
) -> tuple[str, str | dict[str, str]]:
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
    if response_type == "ENTITY_RESOLUTION":
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Resposta do modelo inválida para resolução de entidade."
            ) from exc
        entity_type = str(payload.get("entity_type", "")).strip().lower()
        search_term = str(payload.get("search_term", "")).strip()
        if not entity_type or not search_term:
            raise ValueError(
                "Resposta do modelo incompleta para resolução de entidade."
            )
        return response_type, {
            "entity_type": entity_type,
            "search_term": search_term,
        }
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
    sql_without_string_literals = re.sub(
        r"'(?:''|[^'])*'",
        "''",
        normalized,
    )

    if not (
        lowered.startswith("select")
        or lowered.startswith("with")
        or lowered_without_wrapping.startswith("select")
        or lowered_without_wrapping.startswith("with")
    ):
        raise UnsafeSQLQueryError("A consulta deve ser apenas de leitura.")

    if FORBIDDEN_SQL_PATTERN.search(sql_without_string_literals):
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
            build_router_prompt() | self.llm | StrOutputParser()
        )
        self.final_chain = (
            ChatPromptTemplate.from_template(FINAL_ANSWER_PROMPT_TEMPLATE)
            | self.llm
            | StrOutputParser()
        )
        self.context_chain = RunnableLambda(
            self._extract_turn_context
        ) | RunnableLambda(self._enrich_turn_context)
        self.route_decision_chain = (
            RunnablePassthrough.assign(
                raw_router_response=(
                    RunnableLambda(self._build_router_prompt_payload)
                    | self.router_chain
                )
            )
            | RunnableLambda(self._attach_router_decision)
        )
        self.pagination_chain = RunnableLambda(
            lambda payload: payload,
            afunc=self._pagination_chain_step
        )
        self.entity_resolution_chain = RunnableLambda(
            lambda payload: payload,
            afunc=self._entity_resolution_chain_step
        )
        self.sql_answer_chain = RunnableLambda(
            lambda payload: payload,
            afunc=self._sql_answer_chain_step
        )
        self.state_update_chain = RunnableLambda(
            self._state_update_chain_step
        )
        self.dispatch_chain = RunnableLambda(
            lambda payload: payload,
            afunc=self._dispatch_chain_step,
        )
        self.graph = self._build_graph()

    async def ask(
        self,
        *,
        question: str,
        session_id: str,
        page: int = 1,
        page_size: int = 10,
        confirm_pagination: bool = False,
        disambiguation_choice: dict[str, Any] | None = None,
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
            session_id=session_id,
            answer=result["answer"],
            needs_pagination_confirmation=result.get(
                "needs_pagination_confirmation", False
            ),
            pagination=result.get("pagination"),
            sources=result.get("sources"),
            disambiguation=result.get("disambiguation"),
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
        turn_context = await self.context_chain.ainvoke(state)

        if (
            turn_context["confirm_pagination"]
            and turn_context["pagination_request"]
        ):
            return await self.pagination_chain.ainvoke(turn_context)

        routed_payload = await self.route_decision_chain.ainvoke(turn_context)
        return await self.dispatch_chain.ainvoke(routed_payload)

    def _extract_turn_context(
        self,
        state: ChatbotGraphState,
    ) -> dict[str, Any]:
        return {
            "state": state,
            "question": state["question"],
            "page": state.get("page", 1),
            "page_size": state.get("page_size", 10),
            "confirm_pagination": state.get("confirm_pagination", False),
            "pagination_request": state.get("pagination_request"),
            "disambiguation_choice": state.get("disambiguation_choice"),
        }

    def _enrich_turn_context(
        self,
        turn_context: dict[str, Any],
    ) -> dict[str, Any]:
        state = turn_context["state"]
        turn_context["memory_context"] = self._format_memory_context(state)
        turn_context["resolved_entities"] = self._format_resolved_entities(
            turn_context.get("disambiguation_choice")
        )
        turn_context.setdefault(
            "resolution_mode",
            "Nenhuma restrição adicional.",
        )
        return turn_context

    def _build_router_prompt_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema": get_table_details(),
            "chat_history": payload["memory_context"],
            "resolved_entities": payload["resolved_entities"],
            "resolution_mode": payload["resolution_mode"],
            "question": payload["question"],
        }

    def _attach_router_decision(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        logger.info(
            "[chatbot] llm raw response | question=%s | response=%s",
            payload["question"],
            truncate_text(payload["raw_router_response"], max_chars=600),
        )
        response_type, content = parse_router_response(
            payload["raw_router_response"]
        )
        payload["response_type"] = response_type
        payload["content"] = content
        return payload

    async def _dispatch_chain_step(
        self,
        payload: dict[str, Any],
    ) -> ChatbotGraphState:
        response_type = payload["response_type"]

        if response_type == "REFUSAL":
            return await self.state_update_chain.ainvoke(
                {
                    "state": payload["state"],
                    "question": payload["question"],
                    "answer": sanitize_user_facing_answer(
                        str(payload["content"])
                    ),
                    "sql": None,
                    "pagination": None,
                    "needs_pagination_confirmation": False,
                    "pagination_request": None,
                    "sources": None,
                }
            )

        if response_type == "DIRECT":
            return await self.state_update_chain.ainvoke(
                {
                    "state": payload["state"],
                    "question": payload["question"],
                    "answer": sanitize_user_facing_answer(
                        str(payload["content"])
                    ),
                    "sql": None,
                    "pagination": None,
                    "needs_pagination_confirmation": False,
                    "pagination_request": None,
                    "sources": None,
                }
            )

        if response_type == "ENTITY_RESOLUTION":
            return await self.entity_resolution_chain.ainvoke(payload)

        payload["sql"] = validate_safe_sql(str(payload["content"]))
        logger.info(
            "[chatbot] generated sql | question=%s | sql=%s",
            payload["question"],
            payload["sql"],
        )
        return await self.sql_answer_chain.ainvoke(payload)

    async def _entity_resolution_chain_step(
        self,
        payload: dict[str, Any],
    ) -> ChatbotGraphState:
        resolution = await self._resolve_entity_request(
            question=payload["question"],
            payload=payload["content"],
        )
        if resolution is None:
            raise ValueError(
                "Não consegui resolver a entidade para continuar a consulta."
            )

        auto_resolved_choice = resolution.get("resolved_choice")
        if not auto_resolved_choice:
            return await self.state_update_chain.ainvoke(
                {
                    "state": payload["state"],
                    "question": payload["question"],
                    "answer": resolution["message"],
                    "sql": None,
                    "pagination": None,
                    "needs_pagination_confirmation": False,
                    "pagination_request": None,
                    "sources": None,
                    "disambiguation": resolution["disambiguation"],
                }
            )

        rerouted_payload = {
            **payload,
            "resolved_entities": self._format_resolved_entities(
                auto_resolved_choice
            ),
            "resolution_mode": (
                "As entidades desta pergunta já foram resolvidas. "
                "Use os campos `resolved_id` e `resolved_name` "
                "fornecidos e não retorne [ENTITY_RESOLUTION] "
                "novamente."
            ),
        }
        rerouted_payload = await self.route_decision_chain.ainvoke(
            rerouted_payload
        )
        if rerouted_payload["response_type"] != "SQL":
            raise ValueError(
                "Não consegui gerar a consulta após resolver a entidade."
            )

        rerouted_payload["sql"] = validate_safe_sql(
            str(rerouted_payload["content"])
        )
        logger.info(
            "[chatbot] generated sql after entity resolution | question=%s | sql=%s",
            rerouted_payload["question"],
            rerouted_payload["sql"],
        )
        return await self.sql_answer_chain.ainvoke(rerouted_payload)

    async def _pagination_chain_step(
        self,
        payload: dict[str, Any],
    ) -> ChatbotGraphState:
        return await self._handle_pagination_confirmation(
            state=payload["state"],
            page=payload["page"],
            page_size=payload["page_size"],
            pagination_request=payload["pagination_request"],
        )

    async def _sql_answer_chain_step(
        self,
        payload: dict[str, Any],
    ) -> ChatbotGraphState:
        return await self._answer_from_sql(
            state=payload["state"],
            question=payload["question"],
            sql=payload["sql"],
            page=payload["page"],
            page_size=payload["page_size"],
        )

    def _state_update_chain_step(
        self,
        payload: dict[str, Any],
    ) -> ChatbotGraphState:
        return self._build_state_update(
            state=payload["state"],
            question=payload["question"],
            answer=payload["answer"],
            sql=payload["sql"],
            pagination=payload["pagination"],
            needs_pagination_confirmation=payload[
                "needs_pagination_confirmation"
            ],
            pagination_request=payload["pagination_request"],
            sources=payload["sources"],
            disambiguation=payload.get("disambiguation"),
        )

    def _format_resolved_entities(
        self,
        disambiguation_choice: dict[str, Any] | None,
    ) -> str:
        if not disambiguation_choice:
            return "Nenhuma entidade resolvida previamente."

        return json.dumps(
            [
                {
                    "entity_type": disambiguation_choice.get("kind"),
                    "resolved_name": disambiguation_choice.get("name"),
                    "resolved_id": disambiguation_choice.get("id"),
                }
            ],
            ensure_ascii=False,
        )

    async def _resolve_entity_request(
        self,
        *,
        question: str,
        payload: dict[str, str],
    ) -> dict[str, Any] | None:
        entity_type = payload["entity_type"]
        search_term = payload["search_term"]
        resolver_map = {
            "country": self._find_matching_countries,
            "sector": self._find_matching_sectors,
            "sub_sector": self._find_matching_sub_sectors,
            "project": self._find_matching_projects,
            "funding_provider": self._find_matching_funding_providers,
            "financial_instrument": self._find_matching_financial_instruments,
            "source": self._find_matching_sources,
        }
        resolver = resolver_map.get(entity_type)
        if not resolver:
            raise ValueError(
                "No momento esse tipo de resolução de entidade não é suportado."
            )
        options = await resolver(search_term)
        if not options:
            raise ValueError(
                "Não encontrei entidades compatíveis com o termo informado."
            )
        if len(options) == 1:
            return {
                "resolved_choice": options[0],
            }

        return {
            "message": (
                f'Encontrei mais de uma opção possível para "{search_term}". '
                "Qual delas você quer consultar?"
            ),
            "disambiguation": {
                "message": (
                    f'Encontrei mais de uma opção possível para "{search_term}". '
                    "Escolha uma das opções."
                ),
                "options": options,
                "mode": "select",
            },
        }

    async def _handle_pagination_confirmation(
        self,
        *,
        state: ChatbotGraphState,
        page: int,
        page_size: int,
        pagination_request: dict[str, Any], # {"question": ...,"sql": ...,"total_rows": ...,"page_size": ...}
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
            has_limit = has_explicit_limit(sql)

            if total_rows > page_size and not has_limit:
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
                paginate=not has_limit,
            )
            answer = await self._build_final_answer(
                question=question,
                query=sql,
                rows=rows,
            )

            return self._build_state_update(
                state=state,
                question=question,
                answer=sanitize_user_facing_answer(answer),
                sql=sql,
                pagination=None,
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

    async def _find_matching_countries(
        self,
        search_term: str,
    ) -> list[dict[str, Any]]:
        return await self._find_matching_entities(
            search_term=search_term,
            entities=await self._list_beneficiary_countries(),
            kind="country",
        )

    async def _find_matching_sectors(
        self,
        search_term: str,
    ) -> list[dict[str, Any]]:
        return await self._find_matching_entities(
            search_term=search_term,
            entities=await self._list_sectors(),
            kind="sector",
        )

    async def _find_matching_sub_sectors(
        self,
        search_term: str,
    ) -> list[dict[str, Any]]:
        return await self._find_matching_entities(
            search_term=search_term,
            entities=await self._list_sub_sectors(),
            kind="sub_sector",
        )

    async def _find_matching_projects(
        self,
        search_term: str,
    ) -> list[dict[str, Any]]:
        return await self._find_matching_entities(
            search_term=search_term,
            entities=await self._list_projects(),
            kind="project",
        )

    async def _find_matching_funding_providers(
        self,
        search_term: str,
    ) -> list[dict[str, Any]]:
        return await self._find_matching_entities(
            search_term=search_term,
            entities=await self._list_funding_providers(),
            kind="funding_provider",
        )

    async def _find_matching_financial_instruments(
        self,
        search_term: str,
    ) -> list[dict[str, Any]]:
        return await self._find_matching_entities(
            search_term=search_term,
            entities=await self._list_financial_instruments(),
            kind="financial_instrument",
        )

    async def _find_matching_sources(
        self,
        search_term: str,
    ) -> list[dict[str, Any]]:
        return await self._find_matching_entities(
            search_term=search_term,
            entities=await self._list_sources(),
            kind="source",
        )

    async def _find_matching_entities(
        self,
        *,
        search_term: str,
        entities: list[dict[str, Any]],
        kind: str,
    ) -> list[dict[str, Any]]:
        if not search_term:
            return []

        matched_names = self._rank_entity_candidates(
            search_term=search_term,
            entity_names=[entity["name"] for entity in entities],
        )
        by_name = {entity["name"]: entity for entity in entities}
        return [
            {
                "id": by_name[name].get("id"),
                "name": name,
                "kind": kind,
            }
            for name in matched_names
            if name in by_name
        ]

    async def _list_beneficiary_countries(self) -> list[dict[str, Any]]:
        result = await self.session.execute(
            text(
                """
SELECT DISTINCT
  cfrd.beneficiary_country_id AS entity_id,
  cfrd.beneficiary_country_name AS entity_name
FROM view_climate_finance_records_detailed cfrd
WHERE cfrd.beneficiary_country_name IS NOT NULL
  AND cfrd.beneficiary_country_name NOT ILIKE 'Global%'
  AND cfrd.beneficiary_country_name NOT ILIKE 'Multi-country%'
  AND cfrd.beneficiary_country_name NOT ILIKE 'Regional%'
ORDER BY cfrd.beneficiary_country_name
"""
            )
        )
        return [
            {"id": row.entity_id, "name": row.entity_name}
            for row in result
            if row.entity_name
        ]

    async def _list_sectors(self) -> list[dict[str, Any]]:
        result = await self.session.execute(
            text(
                """
SELECT DISTINCT
  cfrd.sector_id AS entity_id,
  cfrd.sector_name AS entity_name
FROM view_climate_finance_records_detailed cfrd
WHERE cfrd.sector_name IS NOT NULL
ORDER BY cfrd.sector_name
"""
            )
        )
        return [
            {"id": row.entity_id, "name": row.entity_name}
            for row in result
            if row.entity_name
        ]

    async def _list_sub_sectors(self) -> list[dict[str, Any]]:
        result = await self.session.execute(
            text(
                """
SELECT DISTINCT
  cfrd.sub_sector_id AS entity_id,
  cfrd.sub_sector_name AS entity_name
FROM view_climate_finance_records_detailed cfrd
WHERE cfrd.sub_sector_name IS NOT NULL
ORDER BY cfrd.sub_sector_name
"""
            )
        )
        return [
            {"id": row.entity_id, "name": row.entity_name}
            for row in result
            if row.entity_name
        ]

    async def _list_projects(self) -> list[dict[str, Any]]:
        result = await self.session.execute(
            text(
                """
SELECT DISTINCT
  cfrd.project_id AS entity_id,
  cfrd.project_title AS entity_name
FROM view_climate_finance_records_detailed cfrd
WHERE cfrd.project_title IS NOT NULL
ORDER BY cfrd.project_title
"""
            )
        )
        return [
            {"id": row.entity_id, "name": row.entity_name}
            for row in result
            if row.entity_name
        ]

    async def _list_funding_providers(self) -> list[dict[str, Any]]:
        result = await self.session.execute(
            text(
                """
SELECT DISTINCT
  cfrd.funding_provider_id AS entity_id,
  cfrd.funding_provider_name AS entity_name
FROM view_climate_finance_records_detailed cfrd
WHERE cfrd.funding_provider_name IS NOT NULL
ORDER BY cfrd.funding_provider_name
"""
            )
        )
        return [
            {"id": row.entity_id, "name": row.entity_name}
            for row in result
            if row.entity_name
        ]

    async def _list_financial_instruments(self) -> list[dict[str, Any]]:
        result = await self.session.execute(
            text(
                """
SELECT DISTINCT
  cfrd.financial_instrument_id AS entity_id,
  cfrd.financial_instrument_name AS entity_name
FROM view_climate_finance_records_detailed cfrd
WHERE cfrd.financial_instrument_name IS NOT NULL
ORDER BY cfrd.financial_instrument_name
"""
            )
        )
        return [
            {"id": row.entity_id, "name": row.entity_name}
            for row in result
            if row.entity_name
        ]

    async def _list_sources(self) -> list[dict[str, Any]]:
        result = await self.session.execute(
            text(
                """
SELECT DISTINCT
  cfrd.source_id AS entity_id,
  cfrd.source_name AS entity_name
FROM view_climate_finance_records_detailed cfrd
WHERE cfrd.source_name IS NOT NULL
ORDER BY cfrd.source_name
"""
            )
        )
        return [
            {"id": row.entity_id, "name": row.entity_name}
            for row in result
            if row.entity_name
        ]

    def _rank_entity_candidates(
        self,
        *,
        search_term: str,
        entity_names: list[str],
    ) -> list[str]:
        normalized_term = normalize_question_text(search_term)
        if not normalized_term:
            return []

        normalized_to_originals: dict[str, list[str]] = {}
        for entity_name in entity_names:
            normalized_name = normalize_question_text(entity_name)
            normalized_to_originals.setdefault(normalized_name, []).append(
                entity_name
            )

        exact_or_contains_matches = [
            entity_name
            for entity_name in entity_names
            if normalized_term in normalize_question_text(entity_name)
        ]
        if exact_or_contains_matches:
            return exact_or_contains_matches[:10]

        fuzzy_matches = process.extract(
            normalized_term,
            list(normalized_to_originals.keys()),
            scorer=fuzz.WRatio,
            limit=10,
        )
        ranked_matches: list[str] = []
        fallback_matches: list[str] = []
        for normalized_name, score in fuzzy_matches:
            if score >= 45:
                for entity_name in normalized_to_originals[normalized_name]:
                    if entity_name not in ranked_matches:
                        ranked_matches.append(entity_name)
                continue
            if score < 35:
                continue
            for entity_name in normalized_to_originals[normalized_name]:
                if entity_name not in fallback_matches:
                    fallback_matches.append(entity_name)

        if ranked_matches:
            return ranked_matches[:10]
        return fallback_matches[:5]

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
        session_id: str,
        answer: str,
        needs_pagination_confirmation: bool = False,
        pagination: dict[str, Any] | None = None,
        sources: list[dict[str, str]] | None = None,
        disambiguation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "answer": answer,
            "needs_pagination_confirmation": (needs_pagination_confirmation),
            "pagination": pagination,
            "sources": sources,
            "disambiguation": disambiguation,
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
        disambiguation: dict[str, Any] | None = None,
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
            "disambiguation": disambiguation,
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
