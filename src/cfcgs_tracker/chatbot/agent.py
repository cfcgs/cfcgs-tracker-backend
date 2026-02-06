# src/cfcgs_tracker/chatbot/agent.py

import json
import operator
import re
import time
import unicodedata
from difflib import SequenceMatcher
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
    disambiguation_request: Optional[Dict[str, Any]]
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

Q: Quanto o fundo 'CLEAN TECHNOLOGY FUND' prometeu?
(CRÍTICO: Pergunta sobre fundo. Use a tabela funds e a coluna pledge.)
SQL: SELECT funds.pledge FROM funds WHERE funds.fund_name ILIKE 'CLEAN TECHNOLOGY FUND'

Q: Algum projeto doou para o Brasil?
(CRÍTICO: Busca binária (SIM/NÃO). Use LIMIT 1 sem agregação.)
SQL: SELECT vcd.project_name, vcd.amount_usd_thousand FROM view_commitments_detailed vcd WHERE vcd.country_name ILIKE 'Brasil' AND vcd.amount_usd_thousand > 0 LIMIT 1

Q: Qual projeto mais doou para Angola em todos os anos?
SQL: SELECT vcd.project_name, SUM(vcd.amount_usd_thousand) AS total_amount FROM view_commitments_detailed vcd WHERE vcd.country_name ILIKE 'Angola' GROUP BY vcd.project_name ORDER BY total_amount DESC LIMIT 1

Q: Quanto o projeto 'ENERGY SECTOR EFFICIENCY AND EXPANSION P' doou para Angola?
(CRÍTICO: Exemplo de agregação de valor para projeto e país específicos)
SQL: SELECT SUM(vcd.amount_usd_thousand) FROM view_commitments_detailed vcd WHERE vcd.project_name ILIKE 'ENERGY SECTOR EFFICIENCY AND EXPANSION P' AND vcd.country_name ILIKE 'Angola'

Q: Qual projeto mais doou para a Bolivia ao longo dos anos ?
SQL: SELECT vcd.project_name, SUM(vcd.amount_usd_thousand) AS total_amount FROM view_commitments_detailed vcd WHERE (vcd.country_name ILIKE 'Bolivia' OR vcd.country_name ILIKE 'Bolívia') GROUP BY vcd.project_name ORDER BY total_amount DESC LIMIT 1

Q: Quanto o projeto 'LAGUNA COLORADA GEOTHERMAL POWER PLANT CONSTRUCTION PROJECT (SECOND STAGE)' doou para a Bolívia?
(CRÍTICO: Exemplo de agregação de valor para projeto e país específicos usando o OR para o país)
SQL: SELECT SUM(vcd.amount_usd_thousand) FROM view_commitments_detailed vcd WHERE vcd.project_name ILIKE 'LAGUNA COLORADA GEOTHERMAL POWER PLANT CONSTRUCTION PROJECT (SECOND STAGE)' AND (vcd.country_name ILIKE 'Bolivia' OR vcd.country_name ILIKE 'Bolívia')

Q: Qual projeto mais doou para a Africa do Sul ao longo dos anos ?
SQL: SELECT vcd.project_name, SUM(vcd.amount_usd_thousand) AS total_amount FROM view_commitments_detailed vcd WHERE (vcd.country_name ILIKE 'South Africa' OR vcd.country_name ILIKE 'África do Sul') GROUP BY vcd.project_name ORDER BY total_amount DESC LIMIT 1

Q: Quanto o projeto 'POLICY REFORM LOAN TO SUPPORT THE JUST ENERGY TRANSITION II' doou para a África do Sul?
(CRÍTICO: Exemplo de agregação de valor para projeto e país específicos usando o OR para o país, simulando a saída do contextualizador)
SQL: SELECT SUM(vcd.amount_usd_thousand) FROM view_commitments_detailed vcd WHERE vcd.project_name ILIKE 'POLICY REFORM LOAN TO SUPPORT THE JUST ENERGY TRANSITION II' AND (vcd.country_name ILIKE 'South Africa' OR vcd.country_name ILIKE 'África do Sul')


Q: Qual ano teve a maior doação para mitigação?
(CRÍTICO: Ano é um inteiro. Não use EXTRACT(YEAR FROM vcd.year))
SQL: SELECT vcd.year, SUM(vcd.mitigation_amount_usd_thousand) AS total FROM view_commitments_detailed vcd GROUP BY vcd.year ORDER BY total DESC LIMIT 1

Q: Algum país não recebeu nada para mitigação?
(CRÍTICO: Considere apenas países com total doado > 0)
SQL: SELECT vcd.country_name FROM view_commitments_detailed vcd GROUP BY vcd.country_name HAVING SUM(vcd.amount_usd_thousand) > 0 AND SUM(vcd.mitigation_amount_usd_thousand) = 0 LIMIT 1

Q: Qual o projeto que mais financiou o Brasil em 2023?
(Usa ILIKE para flexibilidade)
SQL: SELECT vcd.project_name, SUM(vcd.amount_usd_thousand) AS total_amount FROM view_commitments_detailed vcd WHERE vcd.country_name ILIKE 'Brasil' AND vcd.year = 2023 GROUP BY vcd.project_name ORDER BY total_amount DESC LIMIT 1

Q: qual ano o projeto 'ENERGY SECTOR EFFICIENCY AND EXPANSION P' mais doou para Angola? 
(Exemplo resolvido de acompanhamento com filtro de país, simulando a saída do contextualizador)
SQL: SELECT vcd.year FROM view_commitments_detailed vcd WHERE vcd.project_name ILIKE 'ENERGY SECTOR EFFICIENCY AND EXPANSION P' AND vcd.country_name ILIKE 'Angola' GROUP BY vcd.year ORDER BY SUM(vcd.amount_usd_thousand) DESC LIMIT 1

Q: qual projeto mais doou para a áfrica do sul em 2020?
(Usa OR para checar Inglês e Português)
SQL: SELECT vcd.project_name, SUM(vcd.amount_usd_thousand) AS total_amount FROM view_commitments_detailed vcd WHERE (vcd.country_name ILIKE 'South Africa' OR vcd.country_name ILIKE 'África do Sul') AND vcd.year = 2020 GROUP BY vcd.project_name ORDER BY total_amount DESC LIMIT 1

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
2.5. **Cobertura de Países (CRÍTICO):** Sempre considere **apenas países que receberam algum valor**. Para isso, use `vcd.amount_usd_thousand > 0` ou `commitments.amount_usd_thousand > 0` como filtro/base. Em perguntas do tipo "algum país não recebeu X", trabalhe sobre o conjunto com `SUM(amount_usd_thousand) > 0` e depois aplique a condição do objetivo (ex: mitigação = 0). **Não use a tabela `countries` isoladamente** para listas/checagens de países.
3.  **Escolha (View vs. Tabelas):** Use a `view_commitments_detailed vcd` se a pergunta exigir múltiplos NOMES. Use as tabelas base para consultas simples. Se usar a view `vcd`, **NÃO FAÇA JOIN** com `projects` ou `countries`.
3.5. **"Ambos/Overlap/Sobreposição" (CRÍTICO):** Sempre interprete "ambos/sobreposição/overlap" como **a parte compartilhada** (overlap), nunca como o total de adaptação ou mitigação. Para somas ou rankings de "ambos", use `overlap_amount_usd_thousand`.
     **Se a pergunta disser "apenas/somente/exclusivamente"**, então filtre explicitamente pelas partes exclusivas:
     - `adapt_ex = GREATEST(COALESCE(adaptation,0) - COALESCE(overlap,0), 0)`
     - `mit_ex = GREATEST(COALESCE(mitigation,0) - COALESCE(overlap,0), 0)`
     e aplique `SUM(adapt_ex)=0`, `SUM(mit_ex)=0` e `SUM(overlap)>0`.
3.6. **Fundos (CRÍTICO):** Se a pergunta for sobre um **fundo** (pledge, approval, deposit, disbursement, projetos aprovados), use a tabela `funds` ou `vcd.fund_name` (se precisar cruzar com compromissos). **Não** use `projects.name` para filtrar fundos.
4.  **Filtros (Linguagem) - CRÍTICO:** Os dados de nomes (países, regiões) podem estar em Português ou Inglês. Para garantir que a consulta funcione, **SEMPRE** use o operador `ILIKE` (case-insensitive). **Para nomes de países/regiões comuns (como 'África do Sul', 'África Subsaariana'), gere uma cláusula `OR` para checar a versão em Inglês (prioridade) E a versão em Português.** Ex: `WHERE (vcd.country_name ILIKE 'South Africa' OR vcd.country_name ILIKE 'África do Sul')`. Para nomes que são iguais (ex: 'Brasil', 'Nepal'), use apenas `ILIKE 'Brasil'`. **Se o nome do local tiver vírgula ou sufixo 'regional', preserve o nome literal completo (incluindo a vírgula) e, na ausência de "região" explícito na pergunta, priorize `country_name`.**
5.  **Segurança:** NUNCA gere SQL que consulte `alembic_version`.
6.  **Resposta Direta (Conceitos Gerais):** Se a pergunta for conceitual ou de conhecimento geral sobre financiamento climático (ex: "o que é financiamento climático", diferenças entre fundos bilaterais e multilaterais, mecanismos, fontes), responda diretamente em linguagem natural usando o prefixo `[DIRECT]` (sem SQL).
7.  **Recusa (Tópico):** Se a pergunta não for sobre financiamento climático (nem conceitos gerais), responda APENAS com: `[REFUSAL] Desculpe, só posso responder perguntas sobre os dados de financiamento climático.`
8.  **Recusa (CSV):** Se pedirem CSV/planilha, responda APENAS com: `[REFUSAL] Desculpe, ainda não consigo gerar arquivos CSV.`
9.  **Sugestão de `LIMIT`:** Sugira `LIMIT` **APENAS** se for uma LISTAGEM ABERTA ou RANKING **E NÃO** tiver filtros `WHERE` fortes (como `year`, `country_name`, `project_name`). Se sugerir -> Responda SÓ com a sugestão + tag `[NEEDS_LIMIT]`.
10. **Ranking com soma (CRÍTICO):** Se a consulta usar `ORDER BY SUM(...)` com `GROUP BY`, inclua a `SUM(...)` no `SELECT` com alias (ex: `total_amount`) para permitir respostas mesmo quando o nome é nulo.
11. **Geração de SQL:** Se nenhuma das regras 6, 7, 8 ou 9 se aplicar (é uma pergunta específica), gere a consulta SQL válida, começando com `[SQL]`.

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
5.  **Nome Nulo:** Se o resultado incluir `project_name` nulo/None/null e houver o valor total doado, responda que o projeto que mais doou está com nome nulo e informe o valor aproximado doado.
6.  **Unidade Monetária (CRÍTICO):**
    - Se a consulta envolver **fundos** (`funds.pledge`, `funds.deposit`, `funds.approval`, `funds.disbursement`), informe os valores em **USD** e deixe claro a unidade. Para valores monetários de fundos, use **milhões de USD (USD mi)**.
    - Se a consulta envolver `funds.projects_approved`, trate como **contagem de projetos**, sem unidade monetária.
    - Se a consulta envolver colunas `*_usd_thousand` ou `amount_usd_thousand`, informe em **milhares de USD**.
    - Se a consulta somar `amount_usd_thousand` **sem** filtro de objetivo climático, deixe claro que o total inclui registros **sem classificação de objetivo**.
7.  **Não Exponha SQL:** NUNCA mostre a "Consulta SQL Executada" ou o "Resultado do SQL" bruto na sua resposta.
8.  **Seja Conciso:** Apenas a resposta direta.

Resposta Final (em linguagem natural):"""

# --- PROMPT 3: Roteador de intenção ---
INTENT_ROUTER_PROMPT_TEMPLATE = """Você é um roteador de intenção para um assistente de dados climáticos.
Decida a ação correta com base na pergunta e no histórico.

Retorne APENAS um JSON com:
- intent: uma das opções
  ["confirm_pagination","decline_pagination","greeting","general_finance","general_projects","confirm_context","ask_clarify","query"]
- is_follow_up: true/false (se a pergunta depende da resposta anterior)
- response: texto curto em português (somente para intent greeting/ask_clarify)
- country_mention: string com o país citado pelo usuário (ou "" se não houver)
- project_mention: string com o projeto citado pelo usuário (ou "" se não houver)
- fund_mention: string com o fundo citado pelo usuário (ou "" se não houver)
- objective_only: uma das opções ["mitigation","adaptation","both",""] (usar apenas se o usuário pedir "somente/apenas/só")
- year_start: número (ou null)
- year_end: número (ou null)

Regras:
1) Se existe uma confirmação de paginação pendente e o usuário respondeu "sim/ok/pode", use intent "confirm_pagination".
2) Se existe confirmação pendente e o usuário recusou ("não"), use "decline_pagination".
3) "greeting" se for saudação ou small talk.
4) "general_finance" para perguntas conceituais sobre financiamento climático, sem necessidade de SQL.
5) "general_projects" para pedidos genéricos sobre projetos (sem filtros claros).
6) "confirm_context" quando o usuário pergunta se a resposta anterior era sobre um país/entidade específica.
7) "ask_clarify" quando falta informação para consultar o banco.
8) Caso contrário, "query".
9) Sempre que houver país mencionado, preencha country_mention com o nome literal como o usuário escreveu.
10) Para intervalos de ano ("de 2000 a 2023"), use year_start e year_end.
11) Se a pergunta depender de um país do contexto recente (follow-up), use esse país em country_mention.
12) country_mention deve conter apenas o nome do país (sem prefixos como "país" ou "região").
13) Sempre que houver projeto mencionado, preencha project_mention com o nome literal como o usuário escreveu.
14) Se a pergunta depender de um projeto do contexto recente (follow-up), use esse projeto em project_mention.
15) project_mention deve conter apenas o nome do projeto (sem prefixos como "projeto").
16) Sempre que houver fundo mencionado (ou termos como pledge/aprovação/deposito/desembolso/tipos de fundo), preencha fund_mention com o nome literal como o usuário escreveu.
17) Se a pergunta depender de um fundo do contexto recente (follow-up), use esse fundo em fund_mention.
18) fund_mention deve conter apenas o nome do fundo (sem prefixos como "fundo").

Não gere SQL e não inclua explicações fora do JSON.

Confirmação pendente: {pending_pagination}
Histórico: {chat_history}
Contexto recente: {recent_context}
Pergunta: {question}
JSON:"""

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

SQL_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:sql)?\s*(?P<sql>.+?)```",
    re.IGNORECASE | re.DOTALL,
)

ARTICLE_PREFIXES = {
    "a",
    "o",
    "os",
    "as",
    "ao",
    "aos",
    "à",
    "às",
    "na",
    "no",
    "nas",
    "nos",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "the",
}


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

        self.intent_prompt = ChatPromptTemplate.from_template(INTENT_ROUTER_PROMPT_TEMPLATE)
        self.intent_chain = self.intent_prompt | self.llm | StrOutputParser()

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
                "confirmed_country": None,
                "confirmed_project": None,
                "confirmed_fund": None,
                "last_used_at": 0.0,
            }
        state = self.sessions[session_id]
        state.setdefault("history", [])
        state.setdefault("pagination_request", None)
        state.setdefault("last_rows", [])
        state.setdefault("last_question", None)
        state.setdefault("last_query", None)
        state.setdefault("last_filters", None)
        state.setdefault("confirmed_country", None)
        state.setdefault("confirmed_project", None)
        state.setdefault("confirmed_fund", None)
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

    def _answer_general_question(self, question: str) -> str:
        response = self.general_chain.invoke({"question": question})
        return response.strip()

    def _answer_greeting(self) -> str:
        return (
            "Olá! Posso ajudar com dados de financiamento climático. "
            "Você quer informações sobre projetos, países, anos, fundos ou valores?"
        )

    def _answer_project_data_overview(self) -> str:
        return (
            "Posso te ajudar com dados sobre projetos de financiamento climático, por exemplo:\n"
            "- Valores de financiamento (adaptação, mitigação ou ambos).\n"
            "- Projetos (nomes, fundos associados e países beneficiários).\n"
            "- Países (quem recebeu financiamento e de quem).\n"
            "- Provedores (entidades doadoras).\n"
            "- Anos (evolução por período).\n"
            "- Tipos e focos de fundos.\n"
            "Diga exatamente o que você deseja consultar."
        )

    def _answer_confirmation_without_context(self) -> str:
        return (
            "Para prosseguir, diga o que você deseja consultar. "
            "Exemplos: \"projetos no Brasil em 2020\" ou \"top 10 provedores\"."
        )

    def _answer_context_confirmation(
        self, country_mention: str, session_id: str
    ) -> Optional[str]:
        asked = (country_mention or "").strip()

        if not asked:
            return None

        state = self._get_state(session_id)
        last_filters = state.get("last_filters") or {}
        last_country = last_filters.get("country_name")
        if not last_country:
            return (
                f"A última resposta não estava vinculada a um país específico. "
                f"Você quer que eu consulte {asked}?"
            )

        last_country_primary = last_country.split(" / ", 1)[0].strip()
        if self._normalize_geo_key(last_country_primary) == self._normalize_geo_key(asked):
            return f"Sim, os dados anteriores são referentes a {last_country_primary}."

        return (
            f"Os dados anteriores são referentes a {last_country_primary}. "
            f"Você quer que eu consulte {asked}?"
        )

    def _parse_intent_router_output(self, text: str) -> Dict[str, object]:
        allowed = {
            "confirm_pagination",
            "decline_pagination",
            "greeting",
            "general_finance",
            "general_projects",
            "confirm_context",
            "ask_clarify",
            "query",
        }
        allowed_objectives = {"mitigation", "adaptation", "both", ""}
        if not text:
            return {
                "intent": "query",
                "is_follow_up": False,
                "response": "",
                "country_mention": "",
                "project_mention": "",
                "fund_mention": "",
                "objective_only": "",
                "year_start": None,
                "year_end": None,
            }
        candidate = text.strip()
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            candidate = match.group(0)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return {
                "intent": "query",
                "is_follow_up": False,
                "response": "",
                "country_mention": "",
                "project_mention": "",
                "fund_mention": "",
                "objective_only": "",
                "year_start": None,
                "year_end": None,
            }
        intent = data.get("intent")
        if intent not in allowed:
            intent = "query"
        is_follow_up = bool(data.get("is_follow_up"))
        response = data.get("response") or ""
        country_mention = (data.get("country_mention") or "").strip()
        project_mention = (data.get("project_mention") or "").strip()
        fund_mention = (data.get("fund_mention") or "").strip()
        objective_only = data.get("objective_only") or ""
        if objective_only not in allowed_objectives:
            objective_only = ""

        def parse_year(value: object) -> Optional[int]:
            if value is None or value == "":
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        year_start = parse_year(data.get("year_start"))
        year_end = parse_year(data.get("year_end"))

        return {
            "intent": intent,
            "is_follow_up": is_follow_up,
            "response": response,
            "country_mention": country_mention,
            "project_mention": project_mention,
            "fund_mention": fund_mention,
            "objective_only": objective_only,
            "year_start": year_start,
            "year_end": year_end,
        }

    def _route_intent(
        self,
        *,
        question: str,
        session_id: str,
        pending_pagination: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        recent_context = self._build_recent_context(session_id)
        chat_history = self._format_chat_history(self._get_state(session_id).get("history", []))
        if pending_pagination:
            pending_context = (
                f"Pergunta pendente: {pending_pagination.get('standalone_question')}."
            )
        else:
            pending_context = "Nenhuma."
        raw = self.intent_chain.invoke({
            "question": question,
            "chat_history": chat_history,
            "recent_context": recent_context,
            "pending_pagination": pending_context,
        })
        return self._parse_intent_router_output(raw)

    def _build_objective_only_query(
        self,
        *,
        objective: str,
        country: Optional[str],
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> Optional[str]:
        if not objective or not country:
            return None

        safe_name = country.replace("'", "''")
        where_sql = f"vcd.country_name ILIKE '{safe_name}'"

        if objective == "mitigation":
            amount_expr = (
                "GREATEST(COALESCE(vcd.mitigation_amount_usd_thousand, 0) - "
                "COALESCE(vcd.overlap_amount_usd_thousand, 0), 0)"
            )
        elif objective == "adaptation":
            amount_expr = (
                "GREATEST(COALESCE(vcd.adaptation_amount_usd_thousand, 0) - "
                "COALESCE(vcd.overlap_amount_usd_thousand, 0), 0)"
            )
        else:
            amount_expr = "COALESCE(vcd.overlap_amount_usd_thousand, 0)"

        filters = f"({where_sql})"
        if year_start and year_end:
            if year_start == year_end:
                filters += f" AND vcd.year = {year_start}"
            else:
                filters += f" AND vcd.year BETWEEN {min(year_start, year_end)} AND {max(year_start, year_end)}"
        elif year_start:
            filters += f" AND vcd.year = {year_start}"

        return (
            "SELECT SUM("
            + amount_expr
            + ") AS total_amount "
            "FROM view_commitments_detailed vcd "
            f"WHERE {filters}"
        )

    def _build_limited_question(self, question: str, limit: int) -> str:
        return f"{question}. Liste apenas os primeiros {limit} resultados."

    def _generate_sql_for_limit(
        self,
        *,
        question: str,
        chat_history: str,
        recent_context: str,
        page_size: int,
    ) -> Optional[str]:
        limited_question = self._build_limited_question(question, page_size)
        sql_or_response = self.sql_chain.invoke({
            "question": limited_question,
            "chat_history": chat_history,
            "schema": self.db_langchain.get_table_info(),
            "recent_context": recent_context,
        })
        if not sql_or_response.startswith("[SQL]"):
            return None
        return sql_or_response.replace("[SQL]", "").strip()

    def _normalize_entity_value(self, value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if " / " in text:
            return text.split(" / ", 1)[0].strip()
        return text

    def _sanitize_country_mention(self, mention: str) -> str:
        if not mention:
            return ""
        cleaned = mention.strip().strip(" ?!.")
        if not cleaned:
            return ""
        parts = cleaned.split()
        if parts and parts[0].lower() in ARTICLE_PREFIXES:
            parts = parts[1:]
        return " ".join(parts).strip()

    def _sanitize_project_mention(self, mention: str) -> str:
        if not mention:
            return ""
        cleaned = mention.strip().strip(" ?!.")
        if not cleaned:
            return ""
        parts = cleaned.split()
        if parts and parts[0].lower() in ARTICLE_PREFIXES:
            parts = parts[1:]
        if parts and parts[0].lower() in {"projeto", "projetos", "project", "projects"}:
            parts = parts[1:]
        return " ".join(parts).strip()

    def _sanitize_fund_mention(self, mention: str) -> str:
        if not mention:
            return ""
        cleaned = mention.strip().strip(" ?!.")
        if not cleaned:
            return ""
        parts = cleaned.split()
        if parts and parts[0].lower() in ARTICLE_PREFIXES:
            parts = parts[1:]
        if parts and parts[0].lower() in {"fundo", "fundos", "fund", "funds"}:
            parts = parts[1:]
        return " ".join(parts).strip()

    def _apply_geo_sql_override(self, query: str) -> str:
        if not query:
            return query
        lowered = query.lower()
        if "region_name" not in lowered and "regions.name" not in lowered:
            return query
        updated = re.sub(r"\bvcd\.region_name\b", "vcd.country_name", query, flags=re.IGNORECASE)
        updated = re.sub(r"\bregion_name\b", "country_name", updated, flags=re.IGNORECASE)
        updated = re.sub(r"\bregions\.name\b", "countries.name", updated, flags=re.IGNORECASE)
        if updated != query:
            print("--- Ajuste automático: substituindo region_name por country_name ---")
        return updated

    def _apply_confirmed_country_override(self, query: str, country: Optional[str]) -> str:
        if not query or not country:
            return query
        safe_country = country.replace("'", "''")
        updated = query

        string_literal = r"'(?:''|[^'])*'"
        or_pattern = re.compile(
            rf"(?:vcd\.)?country_name\s+ILIKE\s+{string_literal}\s+OR\s+(?:vcd\.)?country_name\s+ILIKE\s+{string_literal}",
            re.IGNORECASE,
        )
        updated = or_pattern.sub(f"vcd.country_name ILIKE '{safe_country}'", updated)

        single_pattern = re.compile(
            rf"(?:vcd\.)?country_name\s+(?:ILIKE|=)\s+{string_literal}",
            re.IGNORECASE,
        )
        updated = single_pattern.sub(f"vcd.country_name ILIKE '{safe_country}'", updated)

        countries_pattern = re.compile(
            rf"countries\.name\s+(?:ILIKE|=)\s+{string_literal}",
            re.IGNORECASE,
        )
        updated = countries_pattern.sub(f"countries.name ILIKE '{safe_country}'", updated)

        return updated

    def _apply_confirmed_project_override(self, query: str, project: Optional[str]) -> str:
        if not query or not project:
            return query
        safe_project = project.replace("'", "''")
        updated = query
        string_literal = r"'(?:''|[^'])*'"

        or_pattern = re.compile(
            rf"(?:vcd\.)?project_name\s+ILIKE\s+{string_literal}\s+OR\s+(?:vcd\.)?project_name\s+ILIKE\s+{string_literal}",
            re.IGNORECASE,
        )
        updated = or_pattern.sub(f"vcd.project_name ILIKE '{safe_project}'", updated)

        single_pattern = re.compile(
            rf"(?:vcd\.)?project_name\s+(?:ILIKE|=)\s+{string_literal}",
            re.IGNORECASE,
        )
        updated = single_pattern.sub(f"vcd.project_name ILIKE '{safe_project}'", updated)

        projects_pattern = re.compile(
            rf"projects\.name\s+(?:ILIKE|=)\s+{string_literal}",
            re.IGNORECASE,
        )
        updated = projects_pattern.sub(f"projects.name ILIKE '{safe_project}'", updated)

        return updated

    def _apply_confirmed_fund_override(self, query: str, fund: Optional[str]) -> str:
        if not query or not fund:
            return query
        safe_fund = fund.replace("'", "''")
        updated = query
        string_literal = r"'(?:''|[^'])*'"

        def detect_fund_alias(sql: str) -> str:
            reserved = {
                "where",
                "join",
                "inner",
                "left",
                "right",
                "full",
                "group",
                "order",
                "limit",
                "offset",
                "on",
                "union",
                "having",
                "select",
                "from",
            }
            alias_match = re.search(
                r"\bFROM\s+funds(?:\s+AS)?\s+(\w+)\b",
                sql,
                re.IGNORECASE,
            )
            if not alias_match:
                alias_match = re.search(
                    r"\bJOIN\s+funds(?:\s+AS)?\s+(\w+)\b",
                    sql,
                    re.IGNORECASE,
                )
            if alias_match:
                candidate = alias_match.group(1)
                if candidate.lower() not in reserved:
                    return candidate
            if re.search(r"\bfunds\b", sql, re.IGNORECASE):
                return "funds"
            if re.search(r"\bview_commitments_detailed\b", sql, re.IGNORECASE):
                return "vcd"
            return "funds"

        fund_alias = detect_fund_alias(updated)

        prefix_pattern = r"(?:\b\w+\.)*fund_name"
        condition_pattern = re.compile(
            rf"{prefix_pattern}\s+(?:ILIKE|=)\s+{string_literal}",
            re.IGNORECASE,
        )
        updated = condition_pattern.sub(
            f"{fund_alias}.fund_name ILIKE '{safe_fund}'", updated
        )

        projects_pattern = re.compile(
            rf"\bprojects\.name\s+(?:ILIKE|=)\s+{string_literal}",
            re.IGNORECASE,
        )
        updated = projects_pattern.sub(
            f"{fund_alias}.fund_name ILIKE '{safe_fund}'", updated
        )

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

    def _normalize_geo_key(self, value: str) -> str:
        if not value:
            return ""
        text = self._strip_accents(value).lower()
        text = re.sub(r"[^\w\s,/-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _lookup_geo_candidates(self, term: str) -> List[Dict[str, str]]:
        if not term:
            return []
        cleaned = self._sanitize_country_mention(term)
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
                clauses.append(f"vcd.country_name ILIKE :{param_key}")
            where_clause = " OR ".join(clauses)
            sql = f"""
                SELECT DISTINCT 'country' AS kind, vcd.country_name AS name
                FROM view_commitments_detailed vcd
                WHERE vcd.country_name IS NOT NULL
                  AND vcd.country_name <> ''
                  AND vcd.amount_usd_thousand > 0
                  AND ({where_clause})
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
        matches = run_lookup(True)
        if matches:
            return matches
        return self._fuzzy_lookup_geo_candidates(cleaned)

    def _fuzzy_lookup_geo_candidates(self, term: str) -> List[Dict[str, str]]:
        normalized_term = self._normalize_geo_key(term)
        if not normalized_term:
            return []
        sql = """
            SELECT DISTINCT 'country' AS kind, vcd.country_name AS name
            FROM view_commitments_detailed vcd
            WHERE vcd.country_name IS NOT NULL
              AND vcd.country_name <> ''
              AND vcd.amount_usd_thousand > 0
        """
        try:
            result = self.db_session.execute(text(sql))
            candidates = []
            for kind, name in result.fetchall():
                normalized_name = self._normalize_geo_key(name)
                if not normalized_name:
                    continue
                score = SequenceMatcher(None, normalized_term, normalized_name).ratio()
                if score >= 0.6:
                    candidates.append({"kind": kind, "name": name, "score": score})
            candidates.sort(key=lambda item: item["score"], reverse=True)
            return [{"kind": item["kind"], "name": item["name"]} for item in candidates[:5]]
        except Exception as exc:
            print(f"--- Falha ao buscar nomes geográficos (fuzzy): {exc} ---")
            self.db_session.rollback()
            return []

    def _lookup_project_candidates(self, term: str) -> List[Dict[str, str]]:
        if not term:
            return []
        cleaned = self._sanitize_project_mention(term)
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
                param_key = f"p{idx}"
                params[param_key] = f"%{variant}%" if wildcard else variant
                clauses.append(f"vcd.project_name ILIKE :{param_key}")
            where_clause = " OR ".join(clauses)
            sql = f"""
                SELECT DISTINCT 'project' AS kind, vcd.project_name AS name
                FROM view_commitments_detailed vcd
                WHERE vcd.project_name IS NOT NULL
                  AND vcd.project_name <> ''
                  AND vcd.amount_usd_thousand > 0
                  AND ({where_clause})
                LIMIT 10
            """
            try:
                result = self.db_session.execute(text(sql), params)
                return [{"kind": row[0], "name": row[1]} for row in result.fetchall()]
            except Exception as exc:
                print(f"--- Falha ao buscar nomes de projetos: {exc} ---")
                self.db_session.rollback()
                return []

        matches = run_lookup(False)
        if matches:
            return matches
        matches = run_lookup(True)
        if matches:
            return matches
        return self._fuzzy_lookup_project_candidates(cleaned)

    def _lookup_fund_candidates(self, term: str) -> List[Dict[str, str]]:
        if not term:
            return []
        cleaned = self._sanitize_fund_mention(term)
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
                param_key = f"f{idx}"
                params[param_key] = f"%{variant}%" if wildcard else variant
                clauses.append(f"funds.fund_name ILIKE :{param_key}")
            where_clause = " OR ".join(clauses)
            sql = f"""
                SELECT DISTINCT 'fund' AS kind, funds.fund_name AS name
                FROM funds
                WHERE funds.fund_name IS NOT NULL
                  AND funds.fund_name <> ''
                  AND ({where_clause})
                LIMIT 10
            """
            try:
                result = self.db_session.execute(text(sql), params)
                return [{"kind": row[0], "name": row[1]} for row in result.fetchall()]
            except Exception as exc:
                print(f"--- Falha ao buscar nomes de fundos: {exc} ---")
                self.db_session.rollback()
                return []

        matches = run_lookup(False)
        if matches:
            return matches
        matches = run_lookup(True)
        if matches:
            return matches
        return self._fuzzy_lookup_fund_candidates(cleaned)

    def _fuzzy_lookup_fund_candidates(self, term: str) -> List[Dict[str, str]]:
        normalized_term = self._normalize_geo_key(term)
        if not normalized_term:
            return []
        sql = """
            SELECT DISTINCT 'fund' AS kind, funds.fund_name AS name
            FROM funds
            WHERE funds.fund_name IS NOT NULL
              AND funds.fund_name <> ''
        """
        try:
            result = self.db_session.execute(text(sql))
            candidates = []
            for kind, name in result.fetchall():
                normalized_name = self._normalize_geo_key(name)
                if not normalized_name:
                    continue
                score = SequenceMatcher(None, normalized_term, normalized_name).ratio()
                if score >= 0.6:
                    candidates.append({"kind": kind, "name": name, "score": score})
            candidates.sort(key=lambda item: item["score"], reverse=True)
            return [{"kind": item["kind"], "name": item["name"]} for item in candidates[:5]]
        except Exception as exc:
            print(f"--- Falha ao buscar nomes de fundos (fuzzy): {exc} ---")
            self.db_session.rollback()
            return []

    def _fuzzy_lookup_project_candidates(self, term: str) -> List[Dict[str, str]]:
        normalized_term = self._normalize_geo_key(term)
        if not normalized_term:
            return []
        sql = """
            SELECT DISTINCT 'project' AS kind, vcd.project_name AS name
            FROM view_commitments_detailed vcd
            WHERE vcd.project_name IS NOT NULL
              AND vcd.project_name <> ''
              AND vcd.amount_usd_thousand > 0
        """
        try:
            result = self.db_session.execute(text(sql))
            candidates = []
            for kind, name in result.fetchall():
                normalized_name = self._normalize_geo_key(name)
                if not normalized_name:
                    continue
                score = SequenceMatcher(None, normalized_term, normalized_name).ratio()
                if score >= 0.6:
                    candidates.append({"kind": kind, "name": name, "score": score})
            candidates.sort(key=lambda item: item["score"], reverse=True)
            return [{"kind": item["kind"], "name": item["name"]} for item in candidates[:5]]
        except Exception as exc:
            print(f"--- Falha ao buscar nomes de projetos (fuzzy): {exc} ---")
            self.db_session.rollback()
            return []

    def _build_geo_disambiguation_payload(
        self,
        mention: str,
        matches: List[Dict[str, str]],
        mode: str,
    ) -> Dict[str, object]:
        options = [{"name": item["name"], "kind": "country"} for item in matches]
        options_text = ", ".join(item["name"] for item in matches)
        if mode == "confirm":
            message = (
                f"Encontrei um nome de país semelhante a '{mention}': {options_text}. "
                "Você quer usar essa opção?"
            )
        else:
            message = (
                f"Encontrei opções de nomes de países semelhantes a '{mention}': {options_text}. "
                "Selecione a opção correta."
            )
        return {
            "message": message,
            "options": options,
            "mode": mode,
        }

    def _build_project_disambiguation_payload(
        self,
        mention: str,
        matches: List[Dict[str, str]],
        mode: str,
    ) -> Dict[str, object]:
        options = [{"name": item["name"], "kind": "project"} for item in matches]
        options_text = ", ".join(item["name"] for item in matches)
        if mode == "confirm":
            message = (
                f"Encontrei um nome de projeto semelhante a '{mention}': {options_text}. "
                "Você quer usar essa opção?"
            )
        else:
            message = (
                f"Encontrei opções de nomes de projetos semelhantes a '{mention}': {options_text}. "
                "Selecione a opção correta."
            )
        return {
            "message": message,
            "options": options,
            "mode": mode,
        }

    def _build_fund_disambiguation_payload(
        self,
        mention: str,
        matches: List[Dict[str, str]],
        mode: str,
    ) -> Dict[str, object]:
        options = [{"name": item["name"], "kind": "fund"} for item in matches]
        options_text = ", ".join(item["name"] for item in matches)
        if mode == "confirm":
            message = (
                f"Encontrei um nome de fundo semelhante a '{mention}': {options_text}. "
                "Você quer usar essa opção?"
            )
        else:
            message = (
                f"Encontrei opções de nomes de fundos semelhantes a '{mention}': {options_text}. "
                "Selecione a opção correta."
            )
        return {
            "message": message,
            "options": options,
            "mode": mode,
        }

    def _apply_geo_choice(
        self,
        question: str,
        mention: str,
        choice: Dict[str, str],
    ) -> str:
        if not question or not mention or not choice:
            return question
        canonical = choice.get("name")
        if not canonical:
            return question
        escaped = canonical.replace("'", "''")
        quoted = f"'{escaped}'"
        replacement = f"país {quoted}"
        article_pattern = re.compile(
            rf"\b(?:a|o|os|as|ao|aos|à|às)\s+{re.escape(mention)}",
            re.IGNORECASE,
        )
        if article_pattern.search(question):
            return article_pattern.sub(replacement, question, count=1)
        return self._replace_case_insensitive(question, mention, replacement)

    def _apply_project_choice(
        self,
        question: str,
        mention: str,
        choice: Dict[str, str],
    ) -> str:
        if not question or not mention or not choice:
            return question
        canonical = choice.get("name")
        if not canonical:
            return question
        escaped = canonical.replace("'", "''")
        quoted = f"'{escaped}'"
        replacement = f"projeto {quoted}"
        article_pattern = re.compile(
            rf"\b(?:o|os|a|as|ao|aos|à|às)\s+{re.escape(mention)}",
            re.IGNORECASE,
        )
        if article_pattern.search(question):
            return article_pattern.sub(replacement, question, count=1)
        return self._replace_case_insensitive(question, mention, replacement)

    def _apply_fund_choice(
        self,
        question: str,
        mention: str,
        choice: Dict[str, str],
    ) -> str:
        if not question or not mention or not choice:
            return question
        canonical = choice.get("name")
        if not canonical:
            return question
        escaped = canonical.replace("'", "''")
        quoted = f"'{escaped}'"
        replacement = f"fundo {quoted}"
        article_pattern = re.compile(
            rf"\b(?:o|os|a|as|ao|aos|à|às)\s+{re.escape(mention)}",
            re.IGNORECASE,
        )
        if article_pattern.search(question):
            return article_pattern.sub(replacement, question, count=1)
        return self._replace_case_insensitive(question, mention, replacement)

    def _apply_geo_disambiguation(
        self,
        question: str,
        mention: Optional[str],
    ) -> tuple[str, Optional[Dict[str, object]]]:
        mention = self._sanitize_country_mention(mention or "")
        if not mention:
            return question, None

        def find_candidates(raw_mention: str) -> tuple[str, List[Dict[str, str]]]:
            matches = self._lookup_geo_candidates(raw_mention)
            if matches:
                return raw_mention, matches
            parts = raw_mention.split()
            while len(parts) > 1:
                parts = parts[:-1]
                candidate = " ".join(parts).strip()
                if not candidate:
                    break
                matches = self._lookup_geo_candidates(candidate)
                if matches:
                    return candidate, matches
            return raw_mention, []

        effective_mention, matches = find_candidates(mention)
        if not matches:
            return question, None

        mode = "confirm" if len(matches) == 1 else "select"
        payload = self._build_geo_disambiguation_payload(
            mention=effective_mention,
            matches=matches,
            mode=mode,
        )
        payload["mention"] = effective_mention
        return question, payload

    def _apply_project_disambiguation(
        self,
        question: str,
        mention: Optional[str],
    ) -> tuple[str, Optional[Dict[str, object]]]:
        mention = self._sanitize_project_mention(mention or "")
        if not mention:
            return question, None

        def find_candidates(raw_mention: str) -> tuple[str, List[Dict[str, str]]]:
            matches = self._lookup_project_candidates(raw_mention)
            if matches:
                return raw_mention, matches
            parts = raw_mention.split()
            while len(parts) > 1:
                parts = parts[:-1]
                candidate = " ".join(parts).strip()
                if not candidate:
                    break
                matches = self._lookup_project_candidates(candidate)
                if matches:
                    return candidate, matches
            return raw_mention, []

        effective_mention, matches = find_candidates(mention)
        if not matches:
            return question, None

        mode = "confirm" if len(matches) == 1 else "select"
        payload = self._build_project_disambiguation_payload(
            mention=effective_mention,
            matches=matches,
            mode=mode,
        )
        payload["mention"] = effective_mention
        return question, payload

    def _apply_fund_disambiguation(
        self,
        question: str,
        mention: Optional[str],
    ) -> tuple[str, Optional[Dict[str, object]]]:
        mention = self._sanitize_fund_mention(mention or "")
        if not mention:
            return question, None

        def find_candidates(raw_mention: str) -> tuple[str, List[Dict[str, str]]]:
            matches = self._lookup_fund_candidates(raw_mention)
            if matches:
                return raw_mention, matches
            parts = raw_mention.split()
            while len(parts) > 1:
                parts = parts[:-1]
                candidate = " ".join(parts).strip()
                if not candidate:
                    break
                matches = self._lookup_fund_candidates(candidate)
                if matches:
                    return candidate, matches
            return raw_mention, []

        effective_mention, matches = find_candidates(mention)
        if not matches:
            return question, None

        mode = "confirm" if len(matches) == 1 else "select"
        payload = self._build_fund_disambiguation_payload(
            mention=effective_mention,
            matches=matches,
            mode=mode,
        )
        payload["mention"] = effective_mention
        return question, payload

    def _infer_disambiguation_choice(
        self,
        question: str,
        pending: Optional[Dict[str, object]],
    ) -> Optional[Dict[str, str]]:
        if not question or not pending:
            return None
        pending_type = pending.get("type")
        if pending_type not in {"geo", "project", "fund"}:
            return None
        options = pending.get("options") or []
        if not options:
            return None
        normalized_question = self._normalize_geo_key(question)
        if not normalized_question:
            return None
        for option in options:
            name = (option or {}).get("name")
            if not name:
                continue
            normalized_name = self._normalize_geo_key(name)
            if not normalized_name:
                continue
            if normalized_question == normalized_name or normalized_name in normalized_question:
                return {"name": name, "kind": option.get("kind") or pending_type}
        return None

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
        if session_id == "default" and self.sessions:
            most_recent_id, most_recent_state = max(
                self.sessions.items(),
                key=lambda item: item[1].get("last_used_at", 0.0),
            )
            if time.time() - most_recent_state.get("last_used_at", 0.0) <= 600:
                return most_recent_id
        return session_id

    def _extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        column_variants = {
            "project_name": ["project_name"],
            "country_name": ["country_name", "countries.name"],
            "region_name": ["region_name", "regions.name"],
            "provider_name": ["provider_name"],
            "fund_name": ["fund_name", "funds.fund_name"],
        }
        string_literal = r"'(?:''|[^'])*'"

        def normalize_literal(value: str) -> str:
            text = value.strip()
            if text.startswith("'") and text.endswith("'") and len(text) >= 2:
                text = text[1:-1]
            return text.replace("''", "'")

        for field, variants in column_variants.items():
            matches = []
            for variant in variants:
                escaped = re.escape(variant)
                # Padrão para ILIKE 'Value' ou = 'Value'
                pattern = rf"(?:\b\w+\.)?{escaped}\s*(?:ILIKE|=)\s*({string_literal})"
                # Inclui também o padrão com OR para países
                or_pattern = rf"(?:\b\w+\.)?{escaped}\s*(?:ILIKE|=)\s*({string_literal})\s+OR\s+(?:\b\w+\.)?{escaped}\s*(?:ILIKE|=)\s*({string_literal})"
                
                # Extrai primeiro os ORs (captura 2 valores)
                or_matches = re.findall(or_pattern, query, re.IGNORECASE)
                for match in or_matches:
                    matches.extend(normalize_literal(item) for item in match) # Adiciona ambos os valores capturados
                
                # Extrai os simples (captura 1 valor)
                simple_matches = re.findall(pattern, query, re.IGNORECASE)
                # Filtra os simples que já foram capturados pelo OR para evitar duplicação
                for simple_match in simple_matches:
                    simple_value = normalize_literal(simple_match)
                    is_duplicate = False
                    for or_match in or_matches:
                        if simple_value in [normalize_literal(item) for item in or_match]:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        matches.append(simple_value)

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
        return question


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
        country_name = filters.get("country_name")
        if country_name:
            state["confirmed_country"] = country_name.split(" / ", 1)[0].strip()
        project_name = filters.get("project_name")
        if project_name:
            state["confirmed_project"] = project_name.split(" / ", 1)[0].strip()
        fund_name = filters.get("fund_name")
        if fund_name:
            state["confirmed_fund"] = fund_name.split(" / ", 1)[0].strip()

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

    def _extract_sql_candidate(self, text: str) -> Optional[str]:
        if not text:
            return None
        match = SQL_CODE_BLOCK_PATTERN.search(text)
        if match:
            return match.group("sql").strip()
        cleaned = text.strip()
        cleaned = re.sub(r"^sql\s*:?", "", cleaned, flags=re.IGNORECASE).strip()
        lowered = cleaned.lower()
        keyword_match = re.search(r"\b(select|with)\b", lowered)
        if not keyword_match:
            return None
        candidate = cleaned[keyword_match.start():].strip()
        if "from" not in candidate.lower():
            return None
        return candidate

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
        disambiguation_choice: Optional[Dict[str, str]] = None,
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
        response_disambiguation: Optional[Dict[str, object]] = None

        page = max(1, page)
        page_size = max(1, min(page_size, 50))

        confirmed_country_override: Optional[str] = None
        confirmed_project_override: Optional[str] = None
        confirmed_fund_override: Optional[str] = None
        skip_geo_disambiguation = False
        skip_project_disambiguation = False
        skip_fund_disambiguation = False
        pending_disambiguation = state.get("disambiguation_request")
        if pending_disambiguation and not disambiguation_choice:
            inferred_choice = self._infer_disambiguation_choice(question, pending_disambiguation)
            if inferred_choice:
                disambiguation_choice = inferred_choice
            else:
                pending_question = pending_disambiguation.get("question")
                if pending_question and pending_question != question:
                    state["disambiguation_request"] = None
                    pending_disambiguation = None
        if disambiguation_choice and pending_disambiguation:
            if pending_disambiguation.get("type") == "geo":
                question = self._apply_geo_choice(
                    pending_disambiguation.get("question", question),
                    pending_disambiguation.get("mention", ""),
                    disambiguation_choice,
                )
                confirmed_country_override = disambiguation_choice.get("name") or None
                state["confirmed_country"] = confirmed_country_override
                skip_geo_disambiguation = True
            elif pending_disambiguation.get("type") == "project":
                question = self._apply_project_choice(
                    pending_disambiguation.get("question", question),
                    pending_disambiguation.get("mention", ""),
                    disambiguation_choice,
                )
                confirmed_project_override = disambiguation_choice.get("name") or None
                state["confirmed_project"] = confirmed_project_override
                skip_project_disambiguation = True
            elif pending_disambiguation.get("type") == "fund":
                question = self._apply_fund_choice(
                    pending_disambiguation.get("question", question),
                    pending_disambiguation.get("mention", ""),
                    disambiguation_choice,
                )
                confirmed_fund_override = disambiguation_choice.get("name") or None
                state["confirmed_fund"] = confirmed_fund_override
                skip_fund_disambiguation = True
            state["disambiguation_request"] = None

        pending = state.get("pagination_request")
        if pending and not confirm_pagination and page == 1:
            routing = self._route_intent(
                question=question,
                session_id=session_id,
                pending_pagination=pending,
            )
            if routing["intent"] == "confirm_pagination":
                confirm_pagination = True
            elif routing["intent"] == "decline_pagination":
                state["pagination_request"] = None
                final_output = routing.get("response") or "Tudo bem. Se quiser, faça uma nova consulta."
                updated_history = chat_history + [HumanMessage(content=question), AIMessage(content=final_output)]
                state["history"] = updated_history[-10:]
                print(f"--- Resposta Final para o Usuário: {final_output} ---")
                return {
                    "answer": final_output,
                    "needs_pagination_confirmation": False,
                    "pagination": None,
                    "sources": None,
                    "disambiguation": None,
                }
            else:
                state["pagination_request"] = None

        is_confirmation_flow = bool(
            pending
            and (
                confirm_pagination
                or page > 1
            )
        )

        try:
            if is_confirmation_flow and pending:
                pending["page_size"] = page_size
                if not pending.get("query"):
                    limited_query = self._generate_sql_for_limit(
                        question=pending["standalone_question"],
                        chat_history=formatted_history,
                        recent_context=self._build_recent_context(session_id),
                        page_size=page_size,
                    )
                    if not limited_query:
                        final_output = self._answer_confirmation_without_context()
                        updated_history = chat_history + [HumanMessage(content=question), AIMessage(content=final_output)]
                        state["history"] = updated_history[-10:]
                        print(f"--- Resposta Final para o Usuário: {final_output} ---")
                        return {
                            "answer": final_output,
                            "needs_pagination_confirmation": False,
                            "pagination": None,
                            "sources": None,
                            "disambiguation": None,
                        }
                    limited_query = self._apply_geo_sql_override(limited_query)
                    limited_query = self._apply_confirmed_country_override(
                        limited_query, confirmed_country_override
                    )
                    limited_query = self._apply_confirmed_project_override(
                        limited_query, confirmed_project_override
                    )
                    limited_query = self._apply_confirmed_fund_override(
                        limited_query, confirmed_fund_override
                    )
                    limited_query = self._apply_heatmap_count_filter(limited_query)
                    columns, rows = self.run_query(limited_query)
                    response_sources = self._detect_sources_from_query(limited_query)
                    if not rows:
                        final_output = "Não encontrei resultados para sua consulta."
                        pagination_payload = None
                    else:
                        sql_result = str(rows)
                        final_output = self.final_answer_chain.invoke({
                            "question": pending["standalone_question"],
                            "chat_history": formatted_history,
                            "query": limited_query,
                            "response": sql_result,
                            "pagination_context": self._format_pagination_context(
                                1, page_size, None, False
                            ),
                        })
                        pagination_payload = self._build_pagination_payload(
                            columns, rows, 1, page_size, None
                        )
                        self._update_context_rows(
                            session_id,
                            question=pending["standalone_question"],
                            query=limited_query,
                            columns=columns,
                            rows=rows,
                        )
                    state["pagination_request"] = None
                else:
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

                routing = self._route_intent(
                    question=question,
                    session_id=session_id,
                )
                is_follow_up = bool(routing.get("is_follow_up"))
                routing_country = self._sanitize_country_mention(
                    routing.get("country_mention") or ""
                )
                routing_project = self._sanitize_project_mention(
                    routing.get("project_mention") or ""
                )
                routing_fund = self._sanitize_fund_mention(
                    routing.get("fund_mention") or ""
                )
                last_confirmed = state.get("confirmed_country")
                if not last_confirmed:
                    last_filters = state.get("last_filters") or {}
                    previous = last_filters.get("country_name")
                    if previous:
                        last_confirmed = previous.split(" / ", 1)[0].strip()
                        state["confirmed_country"] = last_confirmed
                if routing_country and last_confirmed:
                    if self._normalize_geo_key(routing_country) == self._normalize_geo_key(last_confirmed):
                        confirmed_country_override = last_confirmed
                        skip_geo_disambiguation = True
                last_project = state.get("confirmed_project")
                if not last_project:
                    last_filters = state.get("last_filters") or {}
                    previous_project = last_filters.get("project_name")
                    if previous_project:
                        last_project = previous_project.split(" / ", 1)[0].strip()
                        state["confirmed_project"] = last_project
                if routing_project and last_project:
                    if self._normalize_geo_key(routing_project) == self._normalize_geo_key(last_project):
                        confirmed_project_override = last_project
                        skip_project_disambiguation = True
                last_fund = state.get("confirmed_fund")
                if not last_fund:
                    last_filters = state.get("last_filters") or {}
                    previous_fund = last_filters.get("fund_name")
                    if previous_fund:
                        last_fund = previous_fund.split(" / ", 1)[0].strip()
                        state["confirmed_fund"] = last_fund
                if routing_fund and last_fund:
                    if self._normalize_geo_key(routing_fund) == self._normalize_geo_key(last_fund):
                        confirmed_fund_override = last_fund
                        skip_fund_disambiguation = True
                routing_objective = routing.get("objective_only") or ""
                routing_year_start = routing.get("year_start")
                routing_year_end = routing.get("year_end")
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
                print(f"Follow-up detectado (router): {is_follow_up}")
                print(f"Intent (router): {routing.get('intent')}")
                print("---------------------------------------------------------")
                # ---------------------------------------------------------

                resolved_question = self._resolve_manual_references(question, session_id)

                intent = routing.get("intent")
                if intent == "greeting":
                    final_output = routing.get("response") or self._answer_greeting()
                elif intent == "general_finance":
                    print("--- Resposta geral sem SQL (financiamento climático) ---")
                    final_output = self._answer_general_question(resolved_question)
                elif intent == "general_projects":
                    final_output = self._answer_project_data_overview()
                elif intent == "confirm_context":
                    final_output = self._answer_context_confirmation(
                        routing_country, session_id
                    ) or routing.get("response") or self._answer_confirmation_without_context()
                elif intent == "ask_clarify":
                    final_output = routing.get("response") or self._answer_confirmation_without_context()
                else:
                    # Resolve pronomes antes e depois do contextualizer para reforçar o contexto.
                    use_contextualizer = is_follow_up and bool(state.get("last_rows") or chat_history)
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
                    disambiguation_response = None
                    if not skip_fund_disambiguation:
                        standalone_question, disambiguation_response = self._apply_fund_disambiguation(
                            standalone_question,
                            mention=routing_fund,
                        )
                        if disambiguation_response:
                            response_disambiguation = disambiguation_response
                            response_disambiguation["type"] = "fund"
                            response_disambiguation["question"] = standalone_question
                            response_disambiguation["mention"] = disambiguation_response.get("mention")
                            final_output = disambiguation_response["message"]
                            state["disambiguation_request"] = {
                                "type": "fund",
                                "question": standalone_question,
                                "mention": disambiguation_response.get("mention", ""),
                            }
                            updated_history = chat_history + [HumanMessage(content=question), AIMessage(content=final_output)]
                            state["history"] = updated_history[-10:]
                            print(f"--- Resposta Final para o Usuário: {final_output} ---")
                            return {
                                "answer": final_output,
                                "needs_pagination_confirmation": False,
                                "pagination": None,
                                "sources": None,
                                "disambiguation": response_disambiguation,
                            }
                    if not skip_project_disambiguation:
                        standalone_question, disambiguation_response = self._apply_project_disambiguation(
                            standalone_question,
                            mention=routing_project,
                        )
                        if disambiguation_response:
                            response_disambiguation = disambiguation_response
                            response_disambiguation["type"] = "project"
                            response_disambiguation["question"] = standalone_question
                            response_disambiguation["mention"] = disambiguation_response.get("mention")
                            final_output = disambiguation_response["message"]
                            state["disambiguation_request"] = {
                                "type": "project",
                                "question": standalone_question,
                                "mention": disambiguation_response.get("mention", ""),
                            }
                            updated_history = chat_history + [HumanMessage(content=question), AIMessage(content=final_output)]
                            state["history"] = updated_history[-10:]
                            print(f"--- Resposta Final para o Usuário: {final_output} ---")
                            return {
                                "answer": final_output,
                                "needs_pagination_confirmation": False,
                                "pagination": None,
                                "sources": None,
                                "disambiguation": response_disambiguation,
                            }

                    if not skip_geo_disambiguation:
                        standalone_question, disambiguation_response = self._apply_geo_disambiguation(
                            standalone_question,
                            mention=routing_country,
                        )
                        if disambiguation_response:
                            response_disambiguation = disambiguation_response
                            response_disambiguation["type"] = "geo"
                            response_disambiguation["question"] = standalone_question
                            response_disambiguation["mention"] = disambiguation_response.get("mention")
                            final_output = disambiguation_response["message"]
                            state["disambiguation_request"] = {
                                "type": "geo",
                                "question": standalone_question,
                                "mention": disambiguation_response.get("mention", ""),
                            }
                            updated_history = chat_history + [HumanMessage(content=question), AIMessage(content=final_output)]
                            state["history"] = updated_history[-10:]
                            print(f"--- Resposta Final para o Usuário: {final_output} ---")
                            return {
                                "answer": final_output,
                                "needs_pagination_confirmation": False,
                                "pagination": None,
                                "sources": None,
                                "disambiguation": response_disambiguation,
                            }
                    if response_disambiguation:
                        final_output = response_disambiguation["message"]
                        state["disambiguation_request"] = {
                            "type": response_disambiguation.get("type"),
                            "question": standalone_question,
                        }
                        updated_history = chat_history + [HumanMessage(content=question), AIMessage(content=final_output)]
                        state["history"] = updated_history[-10:]
                        print(f"--- Resposta Final para o Usuário: {final_output} ---")
                        return {
                            "answer": final_output,
                            "needs_pagination_confirmation": False,
                            "pagination": None,
                            "sources": None,
                            "disambiguation": response_disambiguation,
                        }
                    else:
                        print(f"--- Pergunta Independente Gerada: {standalone_question} ---")

                        objective_only_query = self._build_objective_only_query(
                            objective=routing_objective,
                            country=confirmed_country_override or routing_country,
                            year_start=routing_year_start,
                            year_end=routing_year_end,
                        )
                        if objective_only_query:
                            sql_or_response = f"[SQL] {objective_only_query}"
                        else:
                            sql_or_response = self.sql_chain.invoke({
                                "question": standalone_question,
                                "chat_history": sql_history,
                                "schema": self.db_langchain.get_table_info(),
                                "recent_context": recent_context_payload,
                            })

                        print(f"--- Saída da SQL Chain: {sql_or_response} ---")

                        if sql_or_response.startswith("[REFUSAL]"):
                            final_output = sql_or_response.replace("[REFUSAL]", "").strip()

                        elif sql_or_response.startswith("[DIRECT]"):
                            final_output = sql_or_response.replace("[DIRECT]", "").strip()

                        elif sql_or_response.startswith("[NEEDS_LIMIT]"):
                            final_output = sql_or_response.replace("[NEEDS_LIMIT]", "").strip()
                            needs_confirmation = True
                            state["pagination_request"] = {
                                "query": None,
                                "standalone_question": standalone_question,
                                "total_rows": None,
                                "page_size": page_size,
                            }

                        else:
                            query = None
                            if sql_or_response.startswith("[SQL]"):
                                query = sql_or_response.replace("[SQL]", "").strip()
                            else:
                                query = self._extract_sql_candidate(sql_or_response)

                            if query:
                                query = self._apply_geo_sql_override(query)
                                query = self._apply_confirmed_country_override(
                                    query, confirmed_country_override
                                )
                                query = self._apply_confirmed_project_override(
                                    query, confirmed_project_override
                                )
                                query = self._apply_confirmed_fund_override(
                                    query, confirmed_fund_override
                                )
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
                                final_output = "Desculpe, tive um problema ao interpretar sua pergunta. Tente reformular."

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
            "disambiguation": response_disambiguation or None,
        }
