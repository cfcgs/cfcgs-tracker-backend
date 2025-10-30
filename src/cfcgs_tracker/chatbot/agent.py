# src/cfcgs_tracker/chatbot/agent.py

import operator
from typing import TypedDict, Annotated, List, Optional
import os

from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
# [REMOVIDO] SQLDatabaseToolkit não será usado diretamente para pegar ferramentas
# from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent  # Usaremos este diretamente
from sqlalchemy.exc import ProgrammingError
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate  # Adicionar PromptTemplate etc.
from sqlalchemy.orm import Session

from src.cfcgs_tracker.settings import Settings

# --- Informações das Tabelas SIMPLIFICADAS ---
CUSTOM_TABLE_INFO_SIMPLIFIED = {
    "view_commitments_detailed": """
        VIEW PRINCIPAL (use alias vcd). Contém dados JUNTOS. 
        Colunas: commitment_id, year, amount_usd_thousand, adaptation_..., mitigation_..., overlap_..., 
        project_id, project_name, country_id, country_name (receptor), region_name, 
        provider_name, channel_name, fund_name, fund_type_name, fund_focus_name. 
        **REGRA:** USE vcd SEMPRE para nomes. Ex: SELECT vcd.project_name ... FROM view_commitments_detailed vcd WHERE vcd.country_name = 'Brazil' ...
    """,
    # Outras tabelas omitidas para brevidade
}

# --- Exemplos Few-Shot ---
FEW_SHOT_EXAMPLES = """
**Exemplos:**
Q: Qual o projeto que mais financiou o Brasil em 2023?
SQL: SELECT vcd.project_name FROM view_commitments_detailed vcd WHERE vcd.country_name = 'Brasil' AND vcd.year = 2023 GROUP BY vcd.project_name ORDER BY SUM(vcd.amount_usd_thousand) DESC LIMIT 1

Q: qual ano esse projeto (referindo-se ao projeto da resposta anterior) mais doou para o Brasil?
SQL: SELECT vcd.year FROM view_commitments_detailed vcd WHERE vcd.project_name = 'NOME_DO_PROJETO_DA_RESPOSTA_ANTERIOR' AND vcd.country_name = 'Brasil' GROUP BY vcd.year ORDER BY SUM(vcd.amount_usd_thousand) DESC LIMIT 1

Q: Liste os 5 maiores financiamentos para Adaptação na África Subsaariana.
SQL: SELECT vcd.project_name, vcd.country_name, vcd.adaptation_amount_usd_thousand FROM view_commitments_detailed vcd WHERE vcd.region_name = 'Sub-Saharan Africa' AND vcd.adaptation_amount_usd_thousand > 0 ORDER BY vcd.adaptation_amount_usd_thousand DESC LIMIT 5

Q: Ranking dos países que financiaram o Nepal? (LENTO)
SQL: SELECT vcd.provider_name FROM view_commitments_detailed vcd WHERE vcd.country_name = 'Nepal' GROUP BY vcd.provider_name ORDER BY SUM(vcd.amount_usd_thousand) DESC LIMIT 10 -- Adicionado LIMIT 10 por padrão para queries de ranking abertas.

Q: Quais projetos existem? (LENTO)
SQL: SELECT DISTINCT vcd.project_name FROM view_commitments_detailed vcd LIMIT 10 -- Adicionado LIMIT 10 por padrão para queries de listagem abertas.
"""

# --- [PROMPT v5 - Simplificado para create_sql_agent direto, Foco no Contexto e LIMIT Padrão] ---
# Usando um formato mais simples que create_sql_agent entende melhor
SYSTEM_PREFIX = """Você é um agente SQL para dados de financiamento climático. Gere SQL usando a view `view_commitments_detailed` (alias `vcd`).

**Regras:**
1.  **Use a VIEW `vcd` SEMPRE** para nomes (`vcd.project_name`, `vcd.country_name`, etc.). NÃO FAÇA JOIN extra.
2.  **Use nomes em INGLÊS** no WHERE (ex: `vcd.country_name = 'Brazil'`, `vcd.region_name = 'Sub-Saharan Africa'`).
3.  **Contexto:** Se a pergunta usar "esse projeto", "dele", etc., use o nome correto do histórico da conversa (última `AIMessage`).
4.  **LIMIT Padrão:** Para perguntas ABERTAS de RANKING (`ORDER BY`) ou LISTAGEM GERAL (`SELECT DISTINCT ... FROM view_...` sem `WHERE` forte), **ADICIONE `LIMIT 10`** à consulta SQL por padrão para evitar lentidão. Informe ao usuário que está mostrando os 10 primeiros. NÃO pergunte antes.
5.  **CSV/Exportar:** Responda APENAS "Desculpe, ainda não consigo gerar arquivos CSV.". NÃO gere SQL.
6.  **Erros SQL:** Se uma query falhar, responda APENAS "Desculpe, tive um problema ao executar a consulta.". NÃO mostre o erro ou o SQL.
7.  **Resultado Vazio:** Se a query retornar vazio, responda APENAS "Não encontrei resultados para sua consulta.".
8.  **Resposta Final:** APENAS a info (linguagem natural), OU a recusa CSV, OU o erro genérico, OU "Não encontrei resultados". NUNCA inclua SQL.

**Schema Info:**
{table_info}

**Exemplos:**
{few_shot_examples}

Use as seguintes ferramentas: {tools}
Histórico da Conversa:
{chat_history}

Pergunta do Usuário: {input}
Pensamento e SQL:{agent_scratchpad}
"""

# Criando o prompt a partir do template string
prompt = PromptTemplate.from_template(SYSTEM_PREFIX).partial(
    table_info=str(CUSTOM_TABLE_INFO_SIMPLIFIED),
    few_shot_examples=FEW_SHOT_EXAMPLES
)


# prompt = ChatPromptTemplate.from_messages( # Usar PromptTemplate simples pode ser melhor
#     [
#         SystemMessagePromptTemplate(prompt=simple_prompt),
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )


class ClimateDataAgent:
    """
    Agente LangChain SIMPLIFICADO para interagir com DB climático via Groq.
    Usa create_sql_agent diretamente, sem LangGraph e sem query checker.
    Aplica LIMIT 10 por padrão em queries abertas de ranking/listagem.
    """

    def __init__(self, db_session: Session):
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, api_key=Settings().GROQ_API_KEY)
        self.db_langchain = SQLDatabase(db_session.get_bind())

        # O toolkit é criado implicitamente pelo create_sql_agent
        # toolkit = SQLDatabaseToolkit(db=self.db_langchain, llm=self.llm)
        # tools = toolkit.get_tools() # Não pegamos as ferramentas separadamente

        # Cria o Agente SQL direto
        # Passamos o prompt customizado aqui
        # handle_parsing_errors: Tentar True primeiro, se der loop, tentar False ou custom handler.
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            db=self.db_langchain,
            agent_type="openai-tools",
            verbose=True,
            prompt=prompt,  # Usando o PromptTemplate simples
            # Não vamos usar o checker, confiamos no prompt e no tratamento de erro pós-execução
            # agent_executor_kwargs={"handle_parsing_errors": True} # Pode causar loop se a correção falhar
            handle_parsing_errors=self._handle_execution_error  # Usar handler mais simples
        )

        # [NOVO] Memória para guardar o histórico da conversa
        self.memory = {}  # Dicionário simples para guardar histórico por 'session_id' (usaremos 'default' por enquanto)

    # Handler de erro simplificado para create_sql_agent
    def _handle_execution_error(self, error: Exception) -> str:
        """Retorna mensagem genérica em caso de erro na execução do SQL."""
        print(f"--- Erro durante execução SQL: {error} ---")
        # Verifica Rate Limit primeiro
        if "rate_limit_exceeded" in str(error) or "413" in str(error):
            return "Desculpe, a consulta gerou muitos dados e excedeu o limite de processamento. Tente ser mais específico."
        # Erro SQL genérico (pois o prompt instruiu a não mostrar)
        return "Desculpe, tive um problema ao executar a consulta."

    def run(self, question: str, session_id: str = "default") -> str:
        """
        Executa uma pergunta contra o agente, usando memória simples.
        """
        print(f"--- Iniciando Run com Pergunta: {question} ---")

        # Recupera histórico da memória (se existir)
        chat_history = self.memory.get(session_id, [])

        try:
            # Invoca o agente com input e histórico
            response = self.agent_executor.invoke({
                "input": question,
                "chat_history": chat_history
            })

            output = response.get("output", "Não consegui processar a resposta.")

            # [NOVO] Verifica explicitamente se a resposta é VAZIA ou contém SQL (falha do LLM)
            sql_found_in_output = "SELECT" in output and ("FROM" in output or "from" in output)
            is_empty_result_message = output.strip() == ""  # A ferramenta SQL pode retornar '' se não achar nada

            if sql_found_in_output:
                print("--- ERRO DETECTADO: Agente retornou SQL! ---")
                final_output = "Desculpe, tive um problema ao formatar a resposta."
            elif is_empty_result_message:
                print("--- Resultado Vazio Detectado ---")
                final_output = "Não encontrei resultados para sua consulta."
            else:
                final_output = output

            # Atualiza histórico na memória (guarda pergunta e resposta)
            # Limita o histórico para evitar contexto muito grande (ex: últimos 6 turnos = 3 perguntas/respostas)
            MAX_HISTORY_TURNS = 6
            updated_history = chat_history + [HumanMessage(content=question), AIMessage(content=final_output)]
            self.memory[session_id] = updated_history[-MAX_HISTORY_TURNS:]  # Mantém apenas os últimos N turnos

            print(f"--- Resposta Final para o Usuário: {final_output} ---")
            return final_output

        except Exception as e:
            # Captura erros gerais
            print(f"--- Erro GERAL ao executar o agente: {e} ---")
            # Usa o handler para tentar dar uma mensagem mais específica
            return self._handle_execution_error(e)

# --- Definição do Estado do Grafo (REMOVIDA - Não usamos mais LangGraph) ---