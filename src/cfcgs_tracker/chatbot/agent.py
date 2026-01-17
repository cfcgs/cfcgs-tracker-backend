# src/cfcgs_tracker/chatbot/agent.py

import operator
import re
import time
import unicodedata
from typing import TypedDict, Annotated, List, Optional, Dict, Any

from langchain_community.utilities import SQLDatabase
from langchain_community.cache import InMemoryCache
try:
    from langchain_core.globals import set_llm_cache
except ImportError:  # pragma: no cover - compat fallback
    from langchain.globals import set_llm_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy.orm import Session

from src.cfcgs_tracker.settings import Settings


# Cache de respostas para evitar chamadas repetidas ao modelo.
set_llm_cache(InMemoryCache())

# Memória compartilhada entre instâncias para manter contexto por sessão.
_SESSION_STORE: Dict[str, "ConversationState"] = {}

SOURCE_CFU = {
    "name": "Climate Funds Update (CFU)",
    "url": "https://climatefundsupdate.org/data-dashboard/",
}
SOURCE_OECD = {
    "name": "OECD (Organisation for Economic Co-operation and Development)",
    "url": "https://www.oecd.org/en.html",
}


# --- Definição do Estado do Grafo (mantido da versão LCEL) ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    sql_query: Optional[str]
    sql_result: Optional[str]
    needs_limit_confirmation: bool
    limit_suggestion_response: Optional[str]


class PaginationState(TypedDict, total=False):
    query: str
    standalone_question: str
    total_rows: Optional[int]
    page_size: int


class ConversationState(TypedDict, total=False):
    history: List[BaseMessage]
    pagination_request: Optional[PaginationState]
    last_rows: List[Dict[str, Any]]
    last_question: Optional[str]
    last_query: Optional[str]
    last_filters: Optional[Dict[str, Any]]
    last_used_at: float


class SQLExecutionError(Exception):
    """Erro amigável ao executar SQL."""


# --- Removido ENTITY_REFERENCE_CONFIG ---


# --- CUSTOM_TABLE_INFO_FULL (Schema completo como solicitado) ---
CUSTOM_TABLE_INFO_FULL = {
    "funds": "Tabela de fundos (pledge, deposit, approval). Colunas: id, fund_name, fund_type_id, fund_focus_id, ...",
    "fund_types": "Tipos de fundos. Colunas: id, name.",
    "fund_focuses": "Focos dos fundos. Colunas: id, name.",
    "projects": "Tabela de projetos (GRANDE ~64k). Colunas: id, name, fund_id, country_id (PODE SER NULO).",
    "regions": "Regiões geográficas. Colunas: id, name.",
    "countries": "Países. Colunas: id, name, region_id.",
    "funding_entities": "Entidades financiadoras (Provedores/Canais). Colunas: id, name.",
    "commitments": """Tabela principal de transações (MUITO GRANDE). Contém IDs e valores. 
        Colunas: id, year, amount_usd_thousand, adaptation_..., mitigation_..., overlap_..., 
        recipient_country_id (FK countries.id), provider_id (FK funding_entities.id), 
        channel_id (FK funding_entities.id), project_id (FK projects.id).
    """,
    "view_commitments_detailed": """
        VIEW AUXILIAR (alias vcd). Use para queries que precisam de múltiplos NOMES (project_name, country_name, region_name, provider_name, fund_name).
        Colunas: ... project_name, country_name (receptor), region_name, provider_name, year (INTEIRO), ...
    """
}

# --- [CORREÇÃO] Exemplos Few-Shot (Ajustado para o Contextualizador) ---
FEW_SHOT_EXAMPLES_FULL = """
**Exemplos:**

Q: Qual é a capital da França?
RESPOSTA: [REFUSAL] Desculpe, só posso responder perguntas sobre os dados de financiamento climático.

Q: O que é financiamento climático?
RESPOSTA: [DIRECT] Financiamento climático é o conjunto de recursos públicos e privados destinados a mitigar e adaptar-se às mudanças climáticas, apoiando projetos, políticas e ações concretas.

Q: Qual a diferença entre fundos bilaterais e multilaterais?
RESPOSTA: [DIRECT] Fundos bilaterais são recursos de um país para outro; fundos multilaterais são financiados por vários países e geridos por organizações internacionais.

Q: Algum projeto doou para o Brasil?
(CRÍTICO: Busca binária (SIM/NÃO). Use LIMIT 1 sem agregação.)
SQL: SELECT vcd.project_name, vcd.amount_usd_thousand FROM view_commitments_detailed vcd WHERE vcd.country_name ILIKE 'Brasil' AND vcd.amount_usd_thousand > 0 LIMIT 1

Q: Qual projeto mais doou para Angola em todos os anos?
SQL: SELECT vcd.project_name, vcd.country_name FROM view_commitments_detailed vcd WHERE vcd.country_name ILIKE 'Angola' GROUP BY vcd.project_name, vcd.country_name ORDER BY SUM(vcd.amount_usd_thousand) DESC LIMIT 1

Q: Quanto o projeto 'ENERGY SECTOR EFFICIENCY AND EXPANSION P' doou para Angola?
(CRÍTICO: Exemplo de agregação de valor para projeto e país específicos)
SQL: SELECT SUM(vcd.amount_usd_thousand) FROM view_commitments_detailed vcd WHERE vcd.project_name ILIKE 'ENERGY SECTOR EFFICIENCY AND EXPANSION P' AND vcd.country_name ILIKE 'Angola'

Q: Qual projeto mais doou para a Bolivia ao longo dos anos ?
SQL: SELECT vcd.project_name, vcd.country_name FROM view_commitments_detailed vcd WHERE (vcd.country_name ILIKE 'Bolivia' OR vcd.country_name ILIKE 'Bolívia') GROUP BY vcd.project_name, vcd.country_name ORDER BY SUM(vcd.amount_usd_thousand) DESC LIMIT 1

Q: Quanto o projeto 'LAGUNA COLORADA GEOTHERMAL POWER PLANT CONSTRUCTION PROJECT (SECOND STAGE)' doou para a Bolívia?
(CRÍTICO: Exemplo de agregação de valor para projeto e país específicos usando o OR para o país)
SQL: SELECT SUM(vcd.amount_usd_thousand) FROM view_commitments_detailed vcd WHERE vcd.project_name ILIKE 'LAGUNA COLORADA GEOTHERMAL POWER PLANT CONSTRUCTION PROJECT (SECOND STAGE)' AND (vcd.country_name ILIKE 'Bolivia' OR vcd.country_name ILIKE 'Bolívia')

Q: Qual projeto mais doou para a Africa do Sul ao longo dos anos ?
SQL: SELECT vcd.project_name, vcd.country_name FROM view_commitments_detailed vcd WHERE (vcd.country_name ILIKE 'South Africa' OR vcd.country_name ILIKE 'África do Sul') GROUP BY vcd.project_name, vcd.country_name ORDER BY SUM(vcd.amount_usd_thousand) DESC LIMIT 1

Q: Quanto o projeto 'POLICY REFORM LOAN TO SUPPORT THE JUST ENERGY TRANSITION II' doou para a África do Sul?
(CRÍTICO: Exemplo de agregação de valor para projeto e país específicos usando o OR para o país, simulando a saída do contextualizador)
SQL: SELECT SUM(vcd.amount_usd_thousand) FROM view_commitments_detailed vcd WHERE vcd.project_name ILIKE 'POLICY REFORM LOAN TO SUPPORT THE JUST ENERGY TRANSITION II' AND (vcd.country_name ILIKE 'South Africa' OR vcd.country_name ILIKE 'África do Sul')


Q: Qual ano teve a maior doação para mitigação?
(CRÍTICO: Ano é um inteiro. Não use EXTRACT(YEAR FROM vcd.year))
SQL: SELECT vcd.year, SUM(vcd.mitigation_amount_usd_thousand) AS total FROM view_commitments_detailed vcd GROUP BY vcd.year ORDER BY total DESC LIMIT 1

Q: Qual o projeto que mais financiou o Brasil em 2023?
(Usa ILIKE para flexibilidade)
SQL: SELECT vcd.project_name FROM view_commitments_detailed vcd WHERE vcd.country_name ILIKE 'Brasil' AND vcd.year = 2023 GROUP BY vcd.project_name ORDER BY SUM(vcd.amount_usd_thousand) DESC LIMIT 1

Q: qual ano o projeto 'ENERGY SECTOR EFFICIENCY AND EXPANSION P' mais doou para Angola? 
(Exemplo resolvido de acompanhamento com filtro de país, simulando a saída do contextualizador)
SQL: SELECT vcd.year FROM view_commitments_detailed vcd WHERE vcd.project_name ILIKE 'ENERGY SECTOR EFFICIENCY AND EXPANSION P' AND vcd.country_name ILIKE 'Angola' GROUP BY vcd.year ORDER BY SUM(vcd.amount_usd_thousand) DESC LIMIT 1

Q: qual projeto mais doou para a áfrica do sul em 2020?
(Usa OR para checar Inglês e Português)
SQL: SELECT vcd.project_name FROM view_commitments_detailed vcd WHERE (vcd.country_name ILIKE 'South Africa' OR vcd.country_name ILIKE 'África do Sul') AND vcd.year = 2020 GROUP BY vcd.project_name ORDER BY SUM(vcd.amount_usd_thousand) DESC LIMIT 1

Q: ranking dos 10 projetos que mais doaram para a áfrica do sul
(Usa OR e limita a 10)
SQL: SELECT vcd.project_name, SUM(vcd.amount_usd_thousand) AS total FROM view_commitments_detailed vcd WHERE (vcd.country_name ILIKE 'South Africa' OR vcd.country_name ILIKE 'África do Sul') GROUP BY vcd.project_name ORDER BY total DESC LIMIT 10

Q: Liste os 5 maiores financiamentos na África Subsaariana.
(Usa OR para checar Inglês e Português)
SQL: SELECT vcd.project_name, vcd.country_name, vcd.adaptation_amount_usd_thousand FROM view_commitments_detailed vcd WHERE (vcd.region_name ILIKE 'Sub-Saharan Africa' OR vcd.region_name ILIKE 'África Subsaariana') AND vcd.adaptation_amount_usd_thousand > 0 ORDER BY vcd.adaptation_amount_usd_thousand DESC LIMIT 5

Q: Quantos projetos doaram para o país África, regional?
(Preserva o nome literal com vírgula e busca por país)
SQL: SELECT COUNT(DISTINCT vcd.project_name) FROM view_commitments_detailed vcd WHERE (vcd.country_name ILIKE 'Africa, regional' OR vcd.country_name ILIKE 'África, regional')

Q: Ranking dos países que financiaram o Nepal? (Tem filtro, não é lento)
SQL: SELECT vcd.provider_name, SUM(vcd.amount_usd_thousand) AS total FROM view_commitments_detailed vcd WHERE vcd.country_name ILIKE 'Nepal' GROUP BY vcd.provider_name ORDER BY total DESC

Q: Quais projetos existem? (LENTO)
RESPOSTA: [NEEDS_LIMIT] Essa consulta pode retornar muitos projetos. Gostaria de ver os primeiros 10 resultados?
"""

# --- [PROMPT ATUALIZADO] (Com Regra de Manutenção de Filtro) ---
SQL_PROMPT_TEMPLATE = """Você é um assistente SQL de elite. Dada uma pergunta, histórico e schema, gere uma consulta SQL OU uma resposta de "sugestão" OU uma resposta de "recusa".

**Schema Completo (Tabelas Base + View Auxiliar):**
{schema}

**Histórico da Conversa (para contexto):**
{chat_history}

**Últimos Resultados Relevantes (dados concretos que você gerou recentemente):**
{recent_context}

**Pergunta do Usuário:**
{question}

{few_shot_examples}

**Regras de Geração:**
0.  **Análise Obrigatória:** Antes de propor SQL, releia todo o histórico e os resultados recentes para entender como a pergunta atual se conecta às respostas anteriores.
1.  **Contexto Inteligente:** Use o histórico APENAS para resolver referências ("esse projeto", "aquele país") e utilize a seção "Últimos Resultados" para descobrir nomes reais citados recentemente. Não invente filtros se a nova pergunta for independente.
1.5. **Manutenção de Filtro (CRÍTICO):** Se a nova pergunta for um acompanhamento (ex: 'quanto doou?', 'qual foi o ano?') e o filtro geográfico (`country_name`, `region_name`) ou temporal (`year`) da `Última Pergunta` for relevante, você deve **manter o filtro** na nova consulta SQL.
2.  **Lógica de Negócio (CRÍTICA):** Para encontrar o país **RECEPTOR**, **SEMPRE** use `commitments.recipient_country_id` (com JOIN) OU `vcd.country_name` (da view). **NÃO USE** `projects.country_id` para filtrar por país receptor.
3.  **Escolha (View vs. Tabelas):** Use a `view_commitments_detailed vcd` se a pergunta exigir múltiplos NOMES. Use as tabelas base para consultas simples. Se usar a view `vcd`, **NÃO FAÇA JOIN** com `projects` ou `countries`.
4.  **Filtros (Linguagem) - CRÍTICO:** Os dados de nomes (países, regiões) podem estar em Português ou Inglês. Para garantir que a consulta funcione, **SEMPRE** use o operador `ILIKE` (case-insensitive). **Para nomes de países/regiões comuns (como 'África do Sul', 'África Subsaariana'), gere uma cláusula `OR` para checar a versão em Inglês (prioridade) E a versão em Português.** Ex: `WHERE (vcd.country_name ILIKE 'South Africa' OR vcd.country_name ILIKE 'África do Sul')`. Para nomes que são iguais (ex: 'Brasil', 'Nepal'), use apenas `ILIKE 'Brasil'`. **Se o nome do local tiver vírgula ou sufixo 'regional', preserve o nome literal completo (incluindo a vírgula) e, na ausência de "região" explícito na pergunta, priorize `country_name`.**
5.  **Segurança:** NUNCA gere SQL que consulte `alembic_version`.
6.  **Resposta Direta (Conceitos Gerais):** Se a pergunta for conceitual ou de conhecimento geral sobre financiamento climático (ex: "o que é financiamento climático", diferenças entre fundos bilaterais e multilaterais, mecanismos, fontes), responda diretamente em linguagem natural usando o prefixo `[DIRECT]` (sem SQL).
7.  **Recusa (Tópico):** Se a pergunta não for sobre financiamento climático (nem conceitos gerais), responda APENAS com: `[REFUSAL] Desculpe, só posso responder perguntas sobre os dados de financiamento climático.`
8.  **Recusa (CSV):** Se pedirem CSV/planilha, responda APENAS com: `[REFUSAL] Desculpe, ainda não consigo gerar arquivos CSV.`
9.  **Sugestão de `LIMIT`:** Sugira `LIMIT` **APENAS** se for uma LISTAGEM ABERTA ou RANKING **E NÃO** tiver filtros `WHERE` fortes (como `year`, `country_name`, `project_name`). Se sugerir -> Responda SÓ com a sugestão + tag `[NEEDS_LIMIT]`.
10. **Geração de SQL:** Se nenhuma das regras 6, 7, 8 ou 9 se aplicar (é uma pergunta específica), gere a consulta SQL válida, começando com `[SQL]`.

**Sua Resposta (APENAS UMA DAS SEGUINTES):**
- `[SQL] SELECT ...`
- `[NEEDS_LIMIT] ...`
- `[DIRECT] ...`
- `[REFUSAL] ...`

SQL_ou_Resposta:"""

# --- PROMPT: Respostas gerais sobre financiamento climático ---
GENERAL_PROMPT_TEMPLATE = """Você é um especialista em financiamento climático.
Responda de forma clara e concisa, em português, em até 4 frases.
Se fizer sentido, destaque diferenças em uma frase curta.
Não cite banco de dados, SQL ou tabelas.

Pergunta: {question}

Resposta direta:"""

# --- PROMPT 2: Formulação da Resposta Final (Inalterado) ---
FINAL_PROMPT_TEMPLATE = """Você é um assistente prestativo. Dada uma pergunta, o histórico da conversa, a consulta SQL executada e o resultado do banco de dados, formule uma resposta final clara e concisa em linguagem natural.

**Histórico da Conversa (para contexto):**
{chat_history}

**Pergunta Original:**
{question}

**Consulta SQL Executada:**
{query}

**Resultado do SQL (lista de tuplas):**
{response}

**Paginação:**
{pagination_context}

**Regras da Resposta:**
1.  **Contexto:** Use o histórico e a pergunta para entender o que o usuário queria.
2.  **Resultado Vazio:** Se o "Resultado do SQL" for `[]` ou uma lista vazia, responda: "Não encontrei resultados para sua consulta."
3.  **Resultado com Dados:** Responda à pergunta do usuário de forma direta e em linguagem natural usando os dados do "Resultado do SQL".
4.  **LIMIT Padrão:** Se o SQL incluiu `LIMIT 10` (porque era um ranking/lista aberta que o usuário confirmou), mencione isso (ex: "Aqui estão os 10 principais resultados:", "O projeto que mais doou foi...").
5.  **Não Exponha SQL:** NUNCA mostre a "Consulta SQL Executada" ou o "Resultado do SQL" bruto na sua resposta.
6.  **Seja Conciso:** Apenas a resposta direta.

Resposta Final (em linguagem natural):"""

# --- [NOVO] PROMPT 3: Reescritor de Contexto ---
CONTEXTUALIZER_PROMPT_TEMPLATE = """Você recebe o histórico do chat, o resultado da última consulta (incluindo filtros e linhas de dados) e uma nova pergunta. Sua tarefa é reescrever a nova pergunta para que ela seja **totalmente independente** (sem pronomes ambíguos) e **mantenha os filtros essenciais do contexto anterior** se eles não forem substituídos pela nova pergunta.

**Regras de Geração (CRÍTICAS):**
1.  **Substituição de Pronome:** Se a pergunta atual contém pronomes como 'esse', 'este', ou 'aquela' (e suas variações), você DEVE substituí-los pelo valor literal da entidade (project_name, country_name) encontrado no `{recent_context}`.
2.  **Manutenção de Filtro:** Se a pergunta de acompanhamento (ex: 'quanto doou?') NÃO ESPECIFICAR o País/Região, você DEVE INCLUIR o filtro geográfico ou temporal relevante da `Última Pergunta` no corpo da nova pergunta.
3.  **Saída:** A pergunta reescrita deve estar totalmente resolvida em termos literais e pronta para gerar SQL. Exemplo de saída esperada: "Quanto o projeto 'LAGUNA COLORADA...' doou para a Bolívia?".

Histórico da Conversa:
{chat_history}

Últimos Resultados Relevantes (Inclui Filtros e Dados):
{recent_context}

Pergunta de Acompanhamento: {question}

Pergunta Independente (Resolvida em nomes literais, mantendo filtros de País/Ano):"""

PRONOUN_TERMS = (
    "esse",
    "essa",
    "este",
    "esta",
    "esses",
    "essas",
    "estes",
    "aquele",
    "aquela",
    "aqueles",
    "aquelas",
    "aquilo",
    "isso",
    "isto",
    "nesse",
    "nessa",
    "neste",
    "nesta",
    "desse",
    "dessa",
    "deste",
    "desta",
    "daquele",
    "daquela",
    "daqueles",
    "daquelas",
)

REFERENCE_NOUNS = (
    "projeto",
    "projetos",
    "project",
    "projects",
    "programa",
    "programas",
    "país",
    "pais",
    "países",
    "paises",
    "country",
    "countries",
    "região",
    "regioes",
    "regiões",
    "region",
    "regions",
    "fundo",
    "fundos",
    "fund",
    "funds",
    "provedor",
    "provedores",
    "provider",
    "providers",
    "entidade",
    "entidades",
)

_PRONOUN_PATTERN = re.compile(
    rf"\b({'|'.join(map(re.escape, PRONOUN_TERMS))})\b",
    re.IGNORECASE,
)
_PRONOUN_NOUN_PATTERN = re.compile(
    rf"\b({'|'.join(map(re.escape, PRONOUN_TERMS))})\b\s+"
    rf"(?P<noun>{'|'.join(map(re.escape, REFERENCE_NOUNS))})\b",
    re.IGNORECASE,
)
_GEO_COMMA_PATTERN = re.compile(
    r"(?:\bpara\b|\bpara\s+a\b|\bpara\s+o\b|\bem\b|\bna\b|\bno\b|\bnos\b|\bnas\b|\bà\b|\bao\b)\s+"
    r"(?P<name>[\wÀ-ÿ][\wÀ-ÿ\s.\-]*?,\s*[\wÀ-ÿ][\wÀ-ÿ\s.\-]*)",
    re.IGNORECASE,
)
_GEO_REGIONAL_PATTERN = re.compile(
    r"(?:\bpara\b|\bem\b|\bna\b|\bno\b|\bnos\b|\bnas\b|\bà\b|\bao\b)\s+"
    r"(?P<name>[\wÀ-ÿ][\wÀ-ÿ\s.\-]*?\bregional\b)",
    re.IGNORECASE,
)
_GEO_COUNTRY_PHRASE_PATTERN = re.compile(
    r"\bpa[ií]s\b\s+"
    r"(?P<name>[\wÀ-ÿ][\wÀ-ÿ\s.\-]*?(?:,\s*[\wÀ-ÿ][\wÀ-ÿ\s.\-]*|\bregional\b))",
    re.IGNORECASE,
)
_EXPLICIT_COUNTRY_PATTERN = re.compile(r"\bpa[ií]s\b\s*'([^']+)'", re.IGNORECASE)
_EXPLICIT_REGION_PATTERN = re.compile(r"\bregi[aã]o\b\s*'([^']+)'", re.IGNORECASE)
_COUNT_DISTINCT_PROJECT_PATTERN = re.compile(
    r"count\s*\(\s*distinct\s+vcd\.(?:project_name|project_id)\s*\)",
    re.IGNORECASE,
)

_NOUN_ENTITY_MAP = {
    "project_name": {"projeto", "projetos", "project", "projects", "programa", "programas"},
    "country_name": {"país", "pais", "países", "paises", "country", "countries"},
    "region_name": {"região", "regioes", "regiões", "region", "regions"},
    "fund_name": {"fundo", "fundos", "fund", "funds"},
    "provider_name": {"provedor", "provedores", "provider", "providers", "entidade", "entidades"},
}

GENERAL_TRIGGERS = (
    "o que é",
    "o que e",
    "o que são",
    "o que sao",
    "qual a diferença",
    "qual a diferenca",
    "diferença entre",
    "diferenca entre",
    "como funciona",
    "como funcionam",
    "explique",
    "defina",
    "conceito",
    "para que serve",
    "visão geral",
    "visao geral",
    "overview",
    "fale sobre",
)

GENERAL_TOPICS = (
    "financiamento climático",
    "financiamento climatico",
    "climate finance",
    "fundos bilaterais",
    "fundos multilaterais",
    "fundo bilateral",
    "fundo multilateral",
    "bilateral",
    "multilateral",
    "adaptação",
    "adaptacao",
    "mitigação",
    "mitigacao",
    "perdas e danos",
    "loss and damage",
    "transição justa",
    "transicao justa",
    "fundo verde para o clima",
    "green climate fund",
    "gcf",
)

GENERAL_PROJECT_ACTIONS = (
    "faz",
    "fazem",
    "são",
    "sao",
    "servem",
    "objetivo",
    "objetivos",
    "impacto",
    "atuam",
    "tratam",
    "significam",
    "representam",
)

DATA_QUERY_PATTERNS = (
    r"\b(19|20)\d{2}\b",
    r"\b(quanto|total|ranking|lista|listar|top|maior|menor|montante|valor)\b",
    r"\b(mais|menos)\s+(doou|doaram|financiou|financiaram|recebeu|receberam)\b",
    r"\b(doou|doaram|financiou|financiaram|recebeu|receberam)\b",
)


class ClimateDataAgent:
    """
    Agente LCEL manual para interagir com DB climático via Gemini.
    Usa uma etapa de reescrita de contexto para lidar com a memória.
    """

    def __init__(self, db_session: Session):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=Settings().GEMINI_API_KEY,
        )
        self.db_langchain = SQLDatabase(db_session.get_bind())
        self.db_session = db_session

        self.sessions = _SESSION_STORE

        self.contextualizer_prompt = ChatPromptTemplate.from_template(CONTEXTUALIZER_PROMPT_TEMPLATE)
        self.contextualizer_chain = self.contextualizer_prompt | self.llm | StrOutputParser()

        self.sql_prompt = ChatPromptTemplate.from_template(SQL_PROMPT_TEMPLATE).partial(
            few_shot_examples=FEW_SHOT_EXAMPLES_FULL,
            table_info=str(CUSTOM_TABLE_INFO_FULL),
        )
        self.sql_chain = self.sql_prompt | self.llm | StrOutputParser()

        self.general_prompt = ChatPromptTemplate.from_template(GENERAL_PROMPT_TEMPLATE)
        self.general_chain = self.general_prompt | self.llm | StrOutputParser()

        self.final_prompt = ChatPromptTemplate.from_template(FINAL_PROMPT_TEMPLATE)
        self.final_answer_chain = self.final_prompt | self.llm | StrOutputParser()

    def _get_state(self, session_id: str) -> ConversationState:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "pagination_request": None,
                "last_rows": [],
                "last_question": None,
                "last_query": None,
                "last_filters": None,
                "last_used_at": 0.0,
            }
        state = self.sessions[session_id]
        state.setdefault("history", [])
        state.setdefault("pagination_request", None)
        state.setdefault("last_rows", [])
        state.setdefault("last_question", None)
        state.setdefault("last_query", None)
        state.setdefault("last_filters", None)
        state.setdefault("last_used_at", 0.0)
        return state

    def _format_chat_history(self, chat_history: List[BaseMessage]) -> str:
        if not chat_history:
            return "Nenhum histórico."
        return "\n".join(
            [
                f"Humano: {msg.content}" if isinstance(msg, HumanMessage) else f"IA: {msg.content}"
                for msg in chat_history
            ]
        )

    def _format_recent_rows(self, rows: List[Dict[str, Any]]) -> str:
        snippets = []
        for row in rows[:3]:
            snippet = ", ".join(f"{key}: {value}" for key, value in row.items())
            snippets.append(f"- {snippet}")
        return "\n".join(snippets) if snippets else "Nenhum dado tabular recente."

    def _empty_recent_context(self) -> str:
        return (
            "Nenhuma pergunta anterior registrada.\n"
            "Filtros anteriores: nenhum.\n"
            "Entidades recentes: nenhuma.\n"
            "Nenhum dado tabular recente."
        )

    def _is_follow_up_question(self, question: str, session_id: str) -> bool:
        if _PRONOUN_PATTERN.search(question):
            return True
        entities = self._extract_recent_entities(session_id)
        if not entities:
            return False
        word_count = len(re.findall(r"\w+", question))
        return word_count <= 6

    def _is_general_finance_question(self, question: str) -> bool:
        normalized = question.strip().lower()
        if not normalized:
            return False

        has_topic = any(topic in normalized for topic in GENERAL_TOPICS)
        if not has_topic:
            has_climate = "clima" in normalized or "climate" in normalized
            has_finance = any(term in normalized for term in ("financiamento", "finance", "fundo", "fundos"))
            has_topic = has_climate and has_finance

        if not has_topic:
            return False

        has_trigger = any(trigger in normalized for trigger in GENERAL_TRIGGERS)
        is_short_topic = len(normalized.split()) <= 6
        if not has_trigger and not is_short_topic:
            return False

        if any(re.search(pattern, normalized) for pattern in DATA_QUERY_PATTERNS):
            return False

        return True

    def _answer_general_question(self, question: str) -> str:
        response = self.general_chain.invoke({"question": question})
        return response.strip()

    def _is_general_project_followup(self, question: str) -> bool:
        normalized = question.strip().lower()
        if not normalized:
            return False
        if any(re.search(pattern, normalized) for pattern in DATA_QUERY_PATTERNS):
            return False
        if "projeto" not in normalized and "projetos" not in normalized:
            return False
        if normalized.startswith("o que") and "projeto" in normalized:
            return True
        return any(term in normalized for term in GENERAL_PROJECT_ACTIONS)

    def _answer_general_project_followup(self) -> str:
        return (
            "Em geral, projetos de financiamento climático apoiam ações de mitigação e adaptação, "
            "como energia renovável, eficiência energética, agricultura resiliente, gestão de água "
            "e infraestrutura mais segura contra eventos extremos. Os objetivos e atividades variam "
            "por país, setor e ano. Se quiser detalhes de projetos específicos, posso listar exemplos "
            "ou filtrar por tema."
        )

    def _normalize_entity_value(self, value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if " / " in text:
            return text.split(" / ", 1)[0].strip()
        return text

    def _has_explicit_country(self, question: str) -> bool:
        if not question:
            return False
        return bool(
            _EXPLICIT_COUNTRY_PATTERN.search(question)
            or re.search(r"\bpa[ií]s\b", question, re.IGNORECASE)
            or re.search(r"\bcountry\b", question, re.IGNORECASE)
        )

    def _has_explicit_region(self, question: str) -> bool:
        if not question:
            return False
        return bool(
            _EXPLICIT_REGION_PATTERN.search(question)
            or re.search(r"\bregi[aã]o\b", question, re.IGNORECASE)
            or re.search(r"\bregion\b", question, re.IGNORECASE)
        )

    def _apply_geo_sql_override(self, question: str, query: str) -> str:
        if not question or not query:
            return query
        lowered = query.lower()
        has_region = "region_name" in lowered or "regions.name" in lowered
        has_country = "country_name" in lowered or "countries.name" in lowered
        if not has_region or has_country:
            return query
        if self._has_explicit_region(question):
            return query
        if not self._has_explicit_country(question) and not self._extract_geo_mentions(question):
            return query

        updated = query
        updated = re.sub(r"\bvcd\.region_name\b", "vcd.country_name", updated, flags=re.IGNORECASE)
        updated = re.sub(r"\bregion_name\b", "country_name", updated, flags=re.IGNORECASE)
        updated = re.sub(r"\bregions\.name\b", "countries.name", updated, flags=re.IGNORECASE)
        if updated != query:
            print("--- Ajuste automático: substituindo region_name por country_name (pergunta explícita de país) ---")
        return updated

    def _apply_heatmap_count_filter(self, query: str) -> str:
        if not query:
            return query
        lowered = query.lower()
        if "view_commitments_detailed" not in lowered:
            return query
        if not _COUNT_DISTINCT_PROJECT_PATTERN.search(query):
            return query
        if any(
            term in lowered
            for term in (
                "adaptation_amount_usd_thousand",
                "mitigation_amount_usd_thousand",
                "overlap_amount_usd_thousand",
            )
        ):
            return query

        objective_filter = (
            "(COALESCE(vcd.adaptation_amount_usd_thousand, 0)"
            " + COALESCE(vcd.mitigation_amount_usd_thousand, 0)"
            " + COALESCE(vcd.overlap_amount_usd_thousand, 0) > 0)"
        )

        clause_pattern = re.compile(r"\b(group\s+by|order\s+by|limit)\b", re.IGNORECASE)
        where_match = re.search(r"\bwhere\b", query, re.IGNORECASE)
        insert_match = clause_pattern.search(query)

        if where_match:
            if insert_match:
                return (
                    query[: insert_match.start()]
                    + f" AND {objective_filter} "
                    + query[insert_match.start():]
                )
            return f"{query} AND {objective_filter}"

        if insert_match:
            return (
                query[: insert_match.start()]
                + f" WHERE {objective_filter} "
                + query[insert_match.start():]
            )

        return f"{query} WHERE {objective_filter}"

    def _detect_sources_from_query(self, query: str) -> List[Dict[str, str]]:
        if not query:
            return []
        lowered = self._normalize_sql(query).lower()
        sources: List[Dict[str, str]] = []

        def add_source(source: Dict[str, str]) -> None:
            if source not in sources:
                sources.append(source)

        if (
            re.search(r"\bview_commitments_detailed\b", lowered)
            or re.search(r"\bcommitments\b", lowered)
        ):
            add_source(SOURCE_OECD)

        if (
            re.search(r"\bfunds\b", lowered)
            or re.search(r"\bfund_projects\b", lowered)
            or re.search(r"\bfund_types\b", lowered)
            or re.search(r"\bfund_focuses\b", lowered)
            or re.search(r"\bfund_status\b", lowered)
        ):
            add_source(SOURCE_CFU)

        return sources

    def _strip_accents(self, value: str) -> str:
        if not value:
            return value
        normalized = unicodedata.normalize("NFKD", value)
        return "".join(char for char in normalized if not unicodedata.combining(char))

    def _extract_geo_mentions(self, question: str) -> List[str]:
        if not question:
            return []
        mentions: List[str] = []
        for match in _GEO_COMMA_PATTERN.finditer(question):
            name = match.group("name").strip(" ?!.")
            if name:
                mentions.append(name)
        for match in _GEO_REGIONAL_PATTERN.finditer(question):
            name = match.group("name").strip(" ?!.")
            if name and name not in mentions:
                mentions.append(name)
        for match in _GEO_COUNTRY_PHRASE_PATTERN.finditer(question):
            name = match.group("name").strip(" ?!.")
            if name and name not in mentions:
                mentions.append(name)
        return mentions

    def _lookup_geo_candidates(self, term: str) -> List[Dict[str, str]]:
        if not term:
            return []
        cleaned = term.strip().strip(" ?!.")
        if not cleaned:
            return []
        variants = [cleaned]
        accentless = self._strip_accents(cleaned)
        if accentless and accentless.lower() != cleaned.lower():
            variants.append(accentless)

        def run_lookup(wildcard: bool) -> List[Dict[str, str]]:
            params = {}
            clauses = []
            for idx, variant in enumerate(variants):
                param_key = f"t{idx}"
                params[param_key] = f"%{variant}%" if wildcard else variant
                clauses.append(f"name ILIKE :{param_key}")
            where_clause = " OR ".join(clauses)
            sql = f"""
                SELECT 'region' AS kind, name FROM regions WHERE {where_clause}
                UNION ALL
                SELECT 'country' AS kind, name FROM countries WHERE {where_clause}
                LIMIT 10
            """
            try:
                result = self.db_session.execute(text(sql), params)
                return [{"kind": row[0], "name": row[1]} for row in result.fetchall()]
            except Exception as exc:
                print(f"--- Falha ao buscar nomes geográficos: {exc} ---")
                self.db_session.rollback()
                return []

        matches = run_lookup(False)
        if matches:
            return matches
        return run_lookup(True)

    def _apply_geo_disambiguation(self, question: str) -> tuple[str, Optional[str]]:
        mentions = self._extract_geo_mentions(question)
        if not mentions:
            return question, None

        updated_question = question
        for mention in mentions:
            matches = self._lookup_geo_candidates(mention)
            if not matches:
                continue
            explicit_region = re.search(
                rf"\bregi[aã]o\s+{re.escape(mention)}|\bregion\s+{re.escape(mention)}",
                updated_question,
                re.IGNORECASE,
            )
            explicit_country = re.search(
                rf"\bpa[ií]s\s+{re.escape(mention)}|\bcountry\s+{re.escape(mention)}",
                updated_question,
                re.IGNORECASE,
            )

            if explicit_region:
                matches = [item for item in matches if item["kind"] == "region"]
            elif explicit_country:
                matches = [item for item in matches if item["kind"] == "country"]
            else:
                country_matches = [item for item in matches if item["kind"] == "country"]
                if country_matches:
                    matches = country_matches
                else:
                    matches = [item for item in matches if item["kind"] == "region"]

            if not matches:
                continue

            if len(matches) == 1:
                canonical = matches[0]["name"]
                kind = matches[0]["kind"]
                quoted = f"'{canonical}'"
                if quoted in updated_question or f"\"{canonical}\"" in updated_question:
                    continue
                needs_prefix = False
                if kind == "region":
                    needs_prefix = not re.search(
                        rf"\bregi[aã]o\s+{re.escape(mention)}",
                        updated_question,
                        re.IGNORECASE,
                    )
                    replacement = f"região {quoted}" if needs_prefix else quoted
                else:
                    needs_prefix = not re.search(
                        rf"\bpa[ií]s\s+{re.escape(mention)}",
                        updated_question,
                        re.IGNORECASE,
                    )
                    replacement = f"país {quoted}" if needs_prefix else quoted

                updated_question = self._replace_case_insensitive(updated_question, mention, replacement)
                continue

            options = ", ".join(
                f"{item['name']} ({'região' if item['kind'] == 'region' else 'país'})"
                for item in matches
            )
            response = (
                f"Encontrei mais de um nome semelhante a '{mention}': {options}. "
                "Qual deles você quer consultar?"
            )
            return updated_question, response

        return updated_question, None

    def _extract_recent_entities(self, session_id: str) -> Dict[str, str]:
        state = self._get_state(session_id)
        entities: Dict[str, str] = {}
        rows = state.get("last_rows") or []
        for row in rows:
            for key in _NOUN_ENTITY_MAP.keys():
                if key in row and row[key]:
                    value = self._normalize_entity_value(row[key])
                    if value:
                        entities.setdefault(key, value)

        filters = state.get("last_filters") or {}
        for key in _NOUN_ENTITY_MAP.keys():
            if key in filters and key not in entities and filters[key]:
                value = self._normalize_entity_value(filters[key])
                if value:
                    entities[key] = value
        return entities

    def _summarize_recent_entities(self, session_id: str) -> str:
        entities = self._extract_recent_entities(session_id)
        if not entities:
            return "Entidades recentes: nenhuma."
        parts = []
        for key in ("project_name", "country_name", "region_name", "fund_name", "provider_name"):
            if key in entities:
                label = key.replace("_", " ")
                parts.append(f"{label}: {entities[key]}")
        return "Entidades recentes: " + "; ".join(parts)

    def _select_entity_for_question(
        self, question: str, entities: Dict[str, str]
    ) -> Optional[str]:
        lowered = question.lower()
        for entity_key, nouns in _NOUN_ENTITY_MAP.items():
            if any(noun in lowered for noun in nouns) and entities.get(entity_key):
                return entities[entity_key]
        for key in ("project_name", "country_name", "region_name", "fund_name", "provider_name"):
            if entities.get(key):
                return entities[key]
        return None

    def _replace_case_insensitive(self, text: str, needle: str, replacement: str) -> str:
        pattern = re.compile(re.escape(needle), re.IGNORECASE)
        return pattern.sub(replacement, text)

    def _normalize_session_id(self, session_id: Optional[str]) -> str:
        if not session_id:
            return "default"
        cleaned = str(session_id).strip()
        return cleaned or "default"

    def _resolve_session_id(self, session_id: str) -> str:
        if session_id in self.sessions:
            return session_id
        if self.sessions:
            most_recent_id, most_recent_state = max(
                self.sessions.items(),
                key=lambda item: item[1].get("last_used_at", 0.0),
            )
            if time.time() - most_recent_state.get("last_used_at", 0.0) <= 600:
                return most_recent_id
        return session_id

    def _needs_disambiguation(self, question: str, session_id: str) -> bool:
        if not _PRONOUN_PATTERN.search(question):
            return False
        entities = self._extract_recent_entities(session_id)
        return not entities

    def _extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        column_variants = {
            "project_name": ["project_name"],
            "country_name": ["country_name", "countries.name"],
            "region_name": ["region_name", "regions.name"],
            "provider_name": ["provider_name"],
            "fund_name": ["fund_name", "funds.fund_name"],
        }

        for field, variants in column_variants.items():
            matches = []
            for variant in variants:
                escaped = re.escape(variant)
                # Padrão para ILIKE 'Value' ou = 'Value'
                pattern = rf"(?:\b\w+\.)?{escaped}\s*(?:ILIKE|=)\s*'([^']+)'"
                # Inclui também o padrão com OR para países
                or_pattern = rf"(?:\b\w+\.)?{escaped}\s*(?:ILIKE|=)\s*'([^']+)'\s+OR\s+(?:\b\w+\.)?{escaped}\s*(?:ILIKE|=)\s*'([^']+)'"
                
                # Extrai primeiro os ORs (captura 2 valores)
                or_matches = re.findall(or_pattern, query, re.IGNORECASE)
                for match in or_matches:
                    matches.extend(match) # Adiciona ambos os valores capturados
                
                # Extrai os simples (captura 1 valor)
                simple_matches = re.findall(pattern, query, re.IGNORECASE)
                # Filtra os simples que já foram capturados pelo OR para evitar duplicação
                for simple_match in simple_matches:
                    is_duplicate = False
                    for or_match in or_matches:
                        if simple_match in or_match:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        matches.append(simple_match)

            if matches:
                # Remove duplicatas mantendo a ordem (Python 3.7+)
                unique = list(dict.fromkeys(matches)) 
                filters[field] = " / ".join(unique) if len(unique) > 1 else unique[0]

        year_equals = re.findall(r"(?:\b\w+\.)?year\s*=\s*(\d{4})", query, re.IGNORECASE)
        if year_equals:
            filters["year"] = ", ".join(dict.fromkeys(year_equals))
        year_between = re.search(
            r"(?:\b\w+\.)?year\s+BETWEEN\s+(\d{4})\s+AND\s+(\d{4})",
            query,
            re.IGNORECASE,
        )
        if year_between:
            filters["year_range"] = f"{year_between.group(1)}-{year_between.group(2)}"

        return filters

    def _describe_filters(self, filters: Optional[Dict[str, Any]]) -> str:
        if not filters:
            return "Filtros anteriores: nenhum."
        parts = []
        for key, value in filters.items():
            label = key.replace("_", " ")
            parts.append(f"{label}: {value}")
        return "Filtros anteriores: " + "; ".join(parts)

    def _resolve_manual_references(self, question: str, session_id: str) -> str:
        if not _PRONOUN_PATTERN.search(question):
            return question

        entities = self._extract_recent_entities(session_id)
        if not entities:
            return question

        def resolve_for_noun(noun: str) -> Optional[str]:
            normalized = noun.lower()
            for entity_key, nouns in _NOUN_ENTITY_MAP.items():
                if normalized in nouns and entities.get(entity_key):
                    return entities[entity_key]
            return None

        def build_replacement(value: str) -> str:
            return value if "'" in value else f"'{value}'"

        def replace_pronoun_noun(match: re.Match) -> str:
            noun = match.group("noun")
            replacement = resolve_for_noun(noun) or self._select_entity_for_question(
                question, entities
            )
            if not replacement:
                return match.group(0)
            return f"{noun} {build_replacement(replacement)}"

        updated = _PRONOUN_NOUN_PATTERN.sub(replace_pronoun_noun, question)

        if _PRONOUN_PATTERN.search(updated):
            fallback = self._select_entity_for_question(updated, entities)
            if fallback:
                updated = _PRONOUN_PATTERN.sub(build_replacement(fallback), updated)

        return updated


    def _build_recent_context(self, session_id: str) -> str:
        state = self._get_state(session_id)
        rows = state.get("last_rows") or []
        question = state.get("last_question")
        filters = state.get("last_filters")
        header = f"Última pergunta: {question}" if question else "Nenhuma pergunta anterior registrada."
        filters_text = self._describe_filters(filters)
        entities_text = self._summarize_recent_entities(session_id)
        rows_text = self._format_recent_rows(rows)
        return f"{header}\n{filters_text}\n{entities_text}\n{rows_text}"

    def _update_context_rows(
        self,
        session_id: str,
        *,
        question: str,
        query: str,
        columns: List[str],
        rows: List[tuple],
    ):
        if not rows:
            self._clear_context_rows(session_id)
            return
        serialized = self._serialize_rows(columns, rows[:3])
        filters = self._extract_filters_from_query(query)
        state = self._get_state(session_id)
        state["last_question"] = question
        state["last_query"] = query
        state["last_rows"] = serialized
        state["last_filters"] = filters

    def _clear_context_rows(self, session_id: str):
        state = self._get_state(session_id)
        state["last_question"] = None
        state["last_query"] = None
        state["last_rows"] = []
        state["last_filters"] = None

    def _is_confirmation_message(self, question: str) -> bool:
        normalized = question.strip().lower()
        normalized = normalized.replace("?", "").replace("!", "").strip()
        if len(normalized) > 120:
            return False

        simple_answers = {
            "sim",
            "sim claro",
            "sim pode",
            "pode",
            "pode sim",
            "pode ser",
            "claro",
            "claro pode",
            "ok",
            "ok pode",
            "prosseguir",
            "continuar",
        }
        if normalized in simple_answers:
            return True

        contains_pagination = "pagin" in normalized
        contains_continue = "continu" in normalized or "prosseg" in normalized
        contains_show = "mostrar" in normalized or "mostra" in normalized
        contains_result = "resultado" in normalized or "resultados" in normalized

        if contains_pagination and (contains_show or contains_result):
            return True
        if contains_continue:
            return True
        if contains_show and contains_result and len(normalized.split()) <= 8:
            return True
        return False

    def _normalize_sql(self, query: str) -> str:
        """Remove espaços extras e o ; final para facilitar os wraps."""
        return query.strip().rstrip(";")

    def _wrap_query_with_limit(self, query: str, limit: int, offset: int) -> str:
        base_query = self._normalize_sql(query)
        wrapped = f"SELECT * FROM ({base_query}) AS paginated_subquery LIMIT {limit}"
        if offset > 0:
            wrapped = f"{wrapped} OFFSET {offset}"
        return wrapped

    def _serialize_rows(self, columns: List[str], rows: List[tuple]) -> List[Dict[str, object]]:
        return [
            {column: value for column, value in zip(columns, row)}
            for row in rows
        ]

    def _count_rows(self, query: str) -> Optional[int]:
        count_sql = (
            f"SELECT COUNT(*) AS total_rows FROM ({self._normalize_sql(query)}) AS count_subquery"
        )
        try:
            result = self.db_session.execute(text(count_sql))
            total = result.scalar()
            return int(total) if total is not None else 0
        except Exception as exc:
            # [MELHORIA DE LOG]
            print(f"--- Falha ao obter COUNT da consulta: {exc} ---")
            self.db_session.rollback()
            return None

    def _build_pagination_prompt(self, total_rows: Optional[int], page_size: int) -> str:
        if total_rows:
            return (
                f"A consulta retornará aproximadamente {total_rows} registros. "
                f"Posso exibir em páginas de {page_size} itens. Deseja continuar?"
            )
        return (
            f"A consulta pode retornar muitos registros. Posso exibir em páginas de {page_size} itens. "
            f"Deseja continuar?"
        )

    def _format_pagination_context(
        self, page: int, page_size: int, total_rows: Optional[int], enabled: bool
    ) -> str:
        if not enabled:
            return "Sem paginação ativa."
        total = str(total_rows) if total_rows is not None else "desconhecido"
        return (
            f"Resultados paginados: página {page}, {page_size} itens por página, "
            f"total estimado {total}."
        )

    def _build_pagination_payload(
        self,
        columns: List[str],
        rows: List[tuple],
        page: int,
        page_size: int,
        total_rows: Optional[int],
    ):
        computed_total = total_rows if total_rows is not None else (len(rows) + (page - 1) * page_size)
        has_more = page * page_size < computed_total
        return {
            "page": page,
            "page_size": page_size,
            "total_rows": computed_total,
            "has_more": has_more,
            "rows": self._serialize_rows(columns, rows),
        }

    def _execute_paginated_flow(
        self,
        *,
        session_id: str,
        standalone_question: str,
        chat_history: List[BaseMessage],
        query: str,
        page: int,
        page_size: int,
        total_rows: Optional[int],
    ):
        offset = (page - 1) * page_size
        paginated_query = self._wrap_query_with_limit(query, page_size, offset)
        columns, rows = self.run_query(paginated_query)
        if not rows:
            final_output = "Não encontrei resultados para sua consulta."
        else:
            sql_result = str(rows)
            pagination_context = self._format_pagination_context(
                page, page_size, total_rows, True
            )
            formatted_history = self._format_chat_history(chat_history)
            final_output = self.final_answer_chain.invoke({
                "question": standalone_question,
                "chat_history": formatted_history,
                "query": paginated_query,
                "response": sql_result,
                "pagination_context": pagination_context,
            })
        pagination_payload = self._build_pagination_payload(
            columns, rows, page, page_size, total_rows
        )
        self._update_context_rows(
            session_id,
            question=standalone_question,
            query=paginated_query,
            columns=columns,
            rows=rows,
        )
        return final_output, pagination_payload

    def run_query(self, query: str):
        """Executa a consulta SQL e retorna colunas e linhas."""
        cleaned_query = self._normalize_sql(query)
        try:
            result = self.db_session.execute(text(cleaned_query))
            rows = [tuple(row) for row in result.fetchall()]
            columns = list(result.keys())
            return columns, rows
        except ProgrammingError as e:
            print(f"--- ERRO AO EXECUTAR SQL: {e} ---")
            self.db_session.rollback()
            # [CORREÇÃO] Retorna APENAS uma mensagem segura, escondendo detalhes do DB.
            safe_error_message = "Desculpe, a consulta gerada não é válida. Por favor, tente reformular sua pergunta. (Código SQL incorreto)"
            raise SQLExecutionError(safe_error_message) from e
        except Exception as e:
            print(f"--- ERRO INESPERADO AO EXECUTAR QUERY: {e} ---")
            self.db_session.rollback()
            raise SQLExecutionError("Desculpe, ocorreu um erro inesperado ao consultar o banco.") from e

    def run(
        self,
        question: str,
        session_id: str = "default",
        page: int = 1,
        page_size: int = 10,
        confirm_pagination: bool = False,
    ) -> Dict[str, object]:
        """
        Executa a pergunta do usuário orquestrando as cadeias LCEL.
        """
        session_id = self._resolve_session_id(self._normalize_session_id(session_id))
        print(f"--- Iniciando Run com Pergunta Original: {question} ---")
        print(f"--- Session ID usado: {session_id} ---")
        state = self._get_state(session_id)
        state["last_used_at"] = time.time()
        chat_history = state.get("history", [])
        formatted_history = self._format_chat_history(chat_history)
        final_output = "Desculpe, ocorreu um erro de processamento."
        pagination_payload = None
        needs_confirmation = False
        response_sources: Optional[List[Dict[str, str]]] = None

        page = max(1, page)
        page_size = max(1, min(page_size, 50))

        pending = state.get("pagination_request")
        is_confirmation_flow = bool(
            pending
            and (
                confirm_pagination
                or self._is_confirmation_message(question)
                or page > 1
            )
        )

        try:
            if is_confirmation_flow and pending:
                pending["page_size"] = page_size
                final_output, pagination_payload = self._execute_paginated_flow(
                    session_id=session_id,
                    standalone_question=pending["standalone_question"],
                    chat_history=chat_history,
                    query=pending["query"],
                    page=page,
                    page_size=page_size,
                    total_rows=pending.get("total_rows"),
                )
                response_sources = self._detect_sources_from_query(pending["query"])
                if pagination_payload and not pagination_payload["has_more"]:
                    state["pagination_request"] = None
            else:
                state["pagination_request"] = None

                is_follow_up = self._is_follow_up_question(question, session_id)
                recent_context_payload = (
                    self._build_recent_context(session_id)
                    if is_follow_up
                    else self._empty_recent_context()
                )
                sql_history = formatted_history if is_follow_up else "Nenhum histórico."

                # --- NOVO LOG DE ESTADO PARA DEBUG ---
                print("--- ESTADO DO CONTEXTO INICIAL (Contextualizer Input) ---")
                print(f"Histórico de Conversa:\n{formatted_history}")
                print(f"Contexto Recente (Payload):\n{recent_context_payload}")
                print(f"Follow-up detectado: {is_follow_up}")
                print("---------------------------------------------------------")
                # ---------------------------------------------------------
                
                # Resolve pronomes antes e depois do contextualizer para reforçar o contexto.
                use_contextualizer = is_follow_up and bool(state.get("last_rows") or chat_history)
                resolved_question = self._resolve_manual_references(question, session_id)

                if self._needs_disambiguation(question, session_id):
                    final_output = (
                        "Para continuar, indique o nome do projeto, país ou fundo a que você se refere."
                    )
                elif self._is_general_project_followup(resolved_question):
                    final_output = self._answer_general_project_followup()
                elif self._is_general_finance_question(resolved_question):
                    print("--- Resposta geral sem SQL (financiamento climático) ---")
                    final_output = self._answer_general_question(resolved_question)
                else:
                    if use_contextualizer:
                        print("--- Gerando pergunta independente (contextualizando via LLM)... ---")
                        # Passamos a pergunta original. O contextualizer LLM deve resolver "esse projeto".
                        standalone_question = self.contextualizer_chain.invoke({
                            "question": resolved_question,
                            "chat_history": formatted_history,
                            "recent_context": recent_context_payload,
                        })
                    else:
                        standalone_question = resolved_question

                    standalone_question = self._resolve_manual_references(
                        standalone_question, session_id
                    )
                    standalone_question, disambiguation_response = self._apply_geo_disambiguation(
                        standalone_question
                    )
                    if disambiguation_response:
                        final_output = disambiguation_response
                    else:
                        print(f"--- Pergunta Independente Gerada: {standalone_question} ---")

                        sql_or_response = self.sql_chain.invoke({
                            "question": standalone_question,
                            "chat_history": sql_history,
                            "schema": self.db_langchain.get_table_info(),
                            "recent_context": recent_context_payload,
                        })

                        print(f"--- Saída da SQL Chain: {sql_or_response} ---")

                        if sql_or_response.startswith("[REFUSAL]"):
                            if self._is_general_finance_question(standalone_question):
                                final_output = self._answer_general_question(standalone_question)
                            else:
                                final_output = sql_or_response.replace("[REFUSAL]", "").strip()

                        elif sql_or_response.startswith("[DIRECT]"):
                            final_output = sql_or_response.replace("[DIRECT]", "").strip()

                        elif sql_or_response.startswith("[NEEDS_LIMIT]"):
                            final_output = sql_or_response.replace("[NEEDS_LIMIT]", "").strip()

                        elif sql_or_response.startswith("[SQL]"):
                            query = sql_or_response.replace("[SQL]", "").strip()
                            query = self._apply_geo_sql_override(standalone_question, query)
                            query = self._apply_heatmap_count_filter(query)
                            total_rows = self._count_rows(query)
                            contains_limit = "limit" in query.lower()
                            should_prompt = (
                                not confirm_pagination
                                and not contains_limit
                                and (total_rows is None or total_rows > page_size)
                            )

                            if should_prompt:
                                final_output = self._build_pagination_prompt(total_rows, page_size)
                                needs_confirmation = True
                                state["pagination_request"] = {
                                    "query": query,
                                    "standalone_question": standalone_question,
                                    "total_rows": total_rows,
                                    "page_size": page_size,
                                }
                            elif confirm_pagination:
                                state["pagination_request"] = {
                                    "query": query,
                                    "standalone_question": standalone_question,
                                    "total_rows": total_rows,
                                    "page_size": page_size,
                                }
                                final_output, pagination_payload = self._execute_paginated_flow(
                                    session_id=session_id,
                                    standalone_question=standalone_question,
                                    chat_history=chat_history,
                                    query=query,
                                    page=page,
                                    page_size=page_size,
                                    total_rows=total_rows,
                                )
                                response_sources = self._detect_sources_from_query(query)
                            else:
                                columns, rows = self.run_query(query)
                                response_sources = self._detect_sources_from_query(query)
                                if not rows:
                                    if "limit 1" in query.lower():
                                        final_output = (
                                            "Não encontrei resultados para sua consulta. "
                                            "(Não, não há registros correspondentes)."
                                        )
                                    else:
                                        final_output = "Não encontrei resultados para sua consulta."
                                else:
                                    sql_result = str(rows)
                                    final_output = self.final_answer_chain.invoke({
                                        "question": standalone_question,
                                        "chat_history": formatted_history,
                                        "query": query,
                                        "response": sql_result,
                                        "pagination_context": self._format_pagination_context(
                                            1, page_size, total_rows, False
                                        ),
                                    })
                                    self._update_context_rows(
                                        session_id,
                                        question=standalone_question,
                                        query=query,
                                        columns=columns,
                                        rows=rows,
                                    )
                        else:
                            print(f"--- ERRO DE FORMATAÇÃO DO LLM: A saída não continha prefixo. ---")
                            final_output = (
                                sql_or_response
                                if len(sql_or_response) > 50
                                else "Desculpe, tive um problema ao interpretar sua pergunta. Tente reformular."
                            )

        except SQLExecutionError as sql_error:
            final_output = str(sql_error)
        except Exception as e:
            print(f"--- Erro GERAL ao executar o agente: {e} ---")
            error_text = str(e).lower()
            if (
                "resourceexhausted" in error_text
                or "quota" in error_text
                or "429" in error_text
                or "rate_limit" in error_text
            ):
                final_output = (
                    "Desculpe, o limite de uso da API do modelo foi atingido. "
                    "Tente novamente em alguns minutos."
                )
            elif "413" in error_text:
                final_output = "Desculpe, a consulta gerou muitos dados e excedeu o limite de processamento. Tente ser mais específico."
            else:
                final_output = "Desculpe, ocorreu um erro inesperado ao processar sua pergunta."

        # ETAPA FINAL: Atualiza o Histórico de Mensagens no Estado
        MAX_HISTORY_TURNS = 10 
        updated_history = chat_history + [HumanMessage(content=question), AIMessage(content=final_output)]
        state["history"] = updated_history[-MAX_HISTORY_TURNS:]

        print(f"--- Resposta Final para o Usuário: {final_output} ---")
        return {
            "answer": final_output,
            "needs_pagination_confirmation": needs_confirmation,
            "pagination": pagination_payload,
            "sources": response_sources or None,
        }
