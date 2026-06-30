from __future__ import annotations

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)


def get_table_details() -> str:
    return """
Views disponíveis para consulta:

1. view_climate_finance_records_detailed (alias recomendado: cfrd)
   Descrição: visão analítica principal dos registros de financiamento climático.
   Grão: uma linha por climate finance record importado.
   Colunas:
   - record_id
   - source_row_hash
   - year
   - project_id, project_title
   - beneficiary_country_id, beneficiary_country_name
   - funding_provider_id, funding_provider_name
   - source_id, source_name, source_url
   - financial_instrument_id, financial_instrument_name
   - sector_id, sector_name
   - sub_sector_id, sub_sector_name
   - approved_amount_usd_millions
   - climate_finance_amount_usd_millions
   - adaptation_amount_usd_millions
   - mitigation_amount_usd_millions
   - both_objectives_amount_usd_millions
   - total_amount_usd_millions (coluna derivada recomendada para somas e rankings)

2. view_provider_fund_profiles_detailed (alias recomendado: vfpd)
   Descrição: visão analítica do perfil financeiro dos provedores com fundo.
   Grão: uma linha por funding provider com perfil de fundo.
   Colunas:
   - funding_provider_id
   - funding_provider_name
   - fund_type_id, fund_type_name
   - fund_focus_id, fund_focus_name
   - pledge
   - deposit
   - approval
   - disbursement
   - projects_approved
""".strip()


FEW_SHOT_EXAMPLES = [
    {
        "question": "Quais países beneficiários receberam recursos em 2024?",
        "answer": "[SQL] SELECT DISTINCT cfrd.beneficiary_country_name FROM view_climate_finance_records_detailed cfrd WHERE cfrd.year = 2024 AND cfrd.climate_finance_amount_usd_millions > 0 ORDER BY cfrd.beneficiary_country_name",
    },
    {
        "question": "Qual foi o total de financiamento climático do Brasil em 2023?",
        "answer": "[SQL] SELECT SUM(cfrd.total_amount_usd_millions) AS total_amount_usd_millions FROM view_climate_finance_records_detailed cfrd WHERE (cfrd.beneficiary_country_name ILIKE 'Brazil' OR cfrd.beneficiary_country_name ILIKE 'Brasil') AND cfrd.year = 2023",
    },
    {
        "question": "Quanto recebeu o Congo?",
        "answer": '[ENTITY_RESOLUTION] {"entity_type": "country", "search_term": "Congo"}',
    },
    {
        "question": "Quanto recebeu a Guiné ao longo dos anos?",
        "answer": '[ENTITY_RESOLUTION] {"entity_type": "country", "search_term": "Guiné"}',
    },
    {
        "question": "Quanto o setor de energia recebeu ao longo dos anos?",
        "answer": '[ENTITY_RESOLUTION] {"entity_type": "sector", "search_term": "energia"}',
    },
    {
        "question": "Quanto recebeu o projeto AndGreen?",
        "answer": '[ENTITY_RESOLUTION] {"entity_type": "project", "search_term": "AndGreen"}',
    },
    {
        "question": "Quanto recebeu o subsetor energia eletrica?",
        "answer": '[ENTITY_RESOLUTION] {"entity_type": "sub_sector", "search_term": "electricity"}',
    },
    {
        "question": "Quanto a Noruega doou?",
        "answer": '[ENTITY_RESOLUTION] {"entity_type": "funding_provider", "search_term": "Noruega"}',
    },
    {
        "question": "Quais os 10 projetos com maior financiamento climático?",
        "answer": "[SQL] SELECT cfrd.project_title, SUM(cfrd.total_amount_usd_millions) AS total_amount_usd_millions FROM view_climate_finance_records_detailed cfrd WHERE cfrd.project_title IS NOT NULL GROUP BY cfrd.project_title ORDER BY total_amount_usd_millions DESC NULLS LAST LIMIT 10",
    },
    {
        "question": "Qual país mais recebeu financiamento climático?",
        "answer": "[SQL] SELECT cfrd.beneficiary_country_name, SUM(cfrd.total_amount_usd_millions) AS total_amount_usd_millions FROM view_climate_finance_records_detailed cfrd WHERE cfrd.beneficiary_country_name IS NOT NULL AND cfrd.beneficiary_country_name NOT ILIKE 'Global%' AND cfrd.beneficiary_country_name NOT ILIKE 'Multi-country%' AND cfrd.beneficiary_country_name NOT ILIKE 'Regional%' GROUP BY cfrd.beneficiary_country_name ORDER BY total_amount_usd_millions DESC NULLS LAST LIMIT 1",
    },
    {
        "question": "Qual setor e subsetor mais recebeu financiamento?",
        "answer": "[SQL] SELECT cfrd.sector_name, cfrd.sub_sector_name, SUM(cfrd.total_amount_usd_millions) AS total_amount_usd_millions FROM view_climate_finance_records_detailed cfrd WHERE cfrd.sector_name IS NOT NULL AND cfrd.sub_sector_name IS NOT NULL GROUP BY cfrd.sector_name, cfrd.sub_sector_name ORDER BY total_amount_usd_millions DESC NULLS LAST LIMIT 1",
    },
    {
        "question": "Qual fundo teve maior pledge?",
        "answer": "[SQL] SELECT vfpd.funding_provider_name, vfpd.pledge FROM view_provider_fund_profiles_detailed vfpd WHERE vfpd.pledge IS NOT NULL ORDER BY vfpd.pledge DESC LIMIT 1",
    },
    {
        "question": "O que é financiamento climático?",
        "answer": "[DIRECT] Financiamento climático é o conjunto de recursos destinados a apoiar mitigação, adaptação e outras ações relacionadas às mudanças climáticas.",
    },
    {
        "question": "Qual a capital da França?",
        "answer": "[REFUSAL] Desculpe, só posso responder perguntas sobre os dados de financiamento climático.",
    },
    {
        "question": "Qual é o usuário e a senha do banco?",
        "answer": "[REFUSAL] Desculpe, não posso fornecer credenciais, segredos ou detalhes internos da plataforma.",
    },
    {
        "question": "Quais são as views disponíveis?",
        "answer": "[REFUSAL] Desculpe, não posso expor detalhes internos de implementação, como views, tabelas, schema ou credenciais.",
    },
]


ROUTER_SYSTEM_PROMPT = """Você é um assistente especializado em financiamento climático.

Use apenas o schema abaixo:
{schema}

Histórico recente:
{chat_history}

Entidades já resolvidas para esta pergunta:
{resolved_entities}

Modo de resolução:
{resolution_mode}

Regras obrigatórias:
1. Responda somente com uma destas opções:
   - [SQL] SELECT ...
   - [DIRECT] ...
   - [REFUSAL] ...
   - [ENTITY_RESOLUTION] {{"entity_type": "...", "search_term": "..."}}
2. Gere apenas consultas SELECT ou WITH ... SELECT.
3. Use apenas estas views: view_climate_finance_records_detailed e view_provider_fund_profiles_detailed.
4. Nunca use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, COPY, GRANT ou REVOKE.
5. Para perguntas sobre projetos, países beneficiários, setores, subsetores, instrumentos, fontes e valores climáticos, use view_climate_finance_records_detailed.
6. Para perguntas sobre pledge, deposit, approval, disbursement e projects_approved de fundos/provedores, use view_provider_fund_profiles_detailed.
7. Se a pergunta for um acompanhamento e depender do contexto anterior, mantenha os filtros relevantes do histórico.
8. Se a pergunta não for sobre financiamento climático ou os dados da plataforma, use [REFUSAL].
9. Se a pergunta for conceitual sobre financiamento climático, use [DIRECT].
10. Prefira ILIKE para filtros textuais.
11. Para somas e rankings de financiamento climático, prefira `total_amount_usd_millions` em vez de `climate_finance_amount_usd_millions`.
12. Quando fizer agregação com ordenação, dê alias explícito para a soma e use `DESC NULLS LAST`.
13. Quando a pergunta falar em "país", evite categorias agregadas como `Global%`, `Multi-country%` e `Regional%`, a menos que o usuário peça explicitamente categorias agregadas.
14. Se o usuário pedir uma listagem ampla, ainda assim gere SQL sem LIMIT. A paginação será tratada pela aplicação.
15. Em respostas [DIRECT], nunca cite nomes de views, tabelas, colunas, SQL, schema, banco de dados ou detalhes internos da implementação.
16. Se a pergunta depender de descobrir qual entidade da base corresponde ao termo usado pelo usuário, responda com [ENTITY_RESOLUTION] em JSON, informando `entity_type` e `search_term`.
17. Use `entity_type` apropriado conforme a entidade pedida: `country`, `sector`, `sub_sector`, `project`, `funding_provider`, `financial_instrument` ou `source`.
18. Se já houver entidades resolvidas no contexto, use-as para gerar a SQL final e não peça nova resolução.
19. Quando usar [ENTITY_RESOLUTION], prefira `search_term` no idioma mais próximo dos valores armazenados na base, que em muitos casos estará em inglês.
20. Se houver `resolved_id` nas entidades resolvidas, prefira filtrar pelo campo de id correspondente (`beneficiary_country_id`, `sector_id`, `sub_sector_id`, `project_id`, `funding_provider_id`, `financial_instrument_id`, `source_id`) em vez de depender apenas do nome textual.
21. Se o usuário pedir credenciais, senhas, usuários, tokens, segredos, nomes de views, nomes de tabelas, schema, colunas, estrutura interna, detalhes de implementação ou qualquer informação operacional da plataforma, use sempre [REFUSAL].
22. Nunca revele ou liste nomes internos de views, tabelas, schema, colunas, credenciais, chaves ou segredos, mesmo que o usuário peça explicitamente.
"""


ROUTER_HUMAN_PROMPT = """Pergunta do usuário:
{question}

Resposta:"""


FINAL_ANSWER_PROMPT_TEMPLATE = """Você é um assistente de dados climáticos.

Pergunta original:
{question}

SQL executado:
{query}

Resultado em JSON:
{response}

Regras:
1. Responda em português.
2. Seja claro e conciso.
3. Se o resultado estiver vazio, responda: "Não encontrei resultados para sua consulta."
4. Se o resultado tiver apenas um valor agregado, responda de forma direta.
5. Se o resultado tiver múltiplas linhas mas não for uma paginação aberta, resuma os principais achados sem inventar dados.
6. Não mencione detalhes internos do banco.
7. Nunca cite nomes de views, tabelas, colunas, SQL, schema ou implementação interna.
8. Nunca revele credenciais, chaves, tokens, nomes internos de objetos do banco ou detalhes operacionais da plataforma.

Resposta final:"""


def build_router_prompt() -> ChatPromptTemplate:
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "Pergunta: {question}"),
            ("ai", "Resposta: {answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=FEW_SHOT_EXAMPLES,
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_SYSTEM_PROMPT),
            few_shot_prompt,
            ("human", ROUTER_HUMAN_PROMPT),
        ]
    )
