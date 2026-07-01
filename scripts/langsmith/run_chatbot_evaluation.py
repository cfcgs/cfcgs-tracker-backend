# ruff: noqa: E402
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langsmith import Client
from langsmith.evaluation import aevaluate
from langsmith.utils import LangSmithAuthError

from evals.langsmith.chatbot_evaluators import CHATBOT_EVALUATORS
from src.cfcgs_tracker.adapters.orm import session_factory
from src.cfcgs_tracker.entrypoints.api.schemas.chatbot import (
    DisambiguationOption,
)
from src.cfcgs_tracker.service_layer.services.chatbot import ChatbotService
from src.cfcgs_tracker.settings import Settings
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

DEFAULT_DATASET_NAME = "cfcgs-chatbot-eval-dataset"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LangSmith evaluation against the real chatbot.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="LangSmith dataset name.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="cfcgs-chatbot-eval",
        help="Experiment prefix shown in LangSmith.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="Number of examples evaluated in parallel.",
    )
    return parser


def classify_behavior(response: dict[str, Any] | None) -> str:
    if not response:
        return "unknown"
    if response.get("disambiguation"):
        return "entity_resolution"
    if response.get("needs_pagination_confirmation"):
        return "pagination_confirmation"

    answer = str(response.get("answer") or "").lower()
    if "não posso" in answer or "nao posso" in answer:
        return "refusal"
    if response.get("sources"):
        return "sql_answer"
    return "direct"


def normalize_text(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def find_selected_option(
    response: dict[str, Any],
    selected_entity_name: str,
) -> dict[str, Any] | None:
    disambiguation = response.get("disambiguation") or {}
    options = disambiguation.get("options") or []
    normalized_target = normalize_text(selected_entity_name)
    for option in options:
        if normalize_text(str(option.get("name") or "")) == normalized_target:
            return option
    return None


async def run_chatbot_target(inputs: dict[str, Any]) -> dict[str, Any]:
    question = str(inputs["question"])
    page_size = int(inputs.get("page_size", 10))
    selected_entity_name = inputs.get("selected_entity_name")
    confirm_pagination_after_prompt = bool(
        inputs.get("confirm_pagination_after_prompt", False)
    )
    session_id = str(uuid4())

    async with session_factory() as session:
        service = ChatbotService(session)
        initial_response = await service.ask(
            question=question,
            session_id=session_id,
            page=1,
            page_size=page_size,
        )

        final_response = initial_response
        selected_option = None
        confirmed_pagination = False

        if selected_entity_name and initial_response.get("disambiguation"):
            selected_option = find_selected_option(
                initial_response,
                str(selected_entity_name),
            )
            if selected_option:
                final_response = await service.ask(
                    question=question,
                    session_id=session_id,
                    page=1,
                    page_size=page_size,
                    disambiguation_choice=DisambiguationOption(
                        **selected_option
                    ).model_dump(),
                )

        if (
            confirm_pagination_after_prompt
            and final_response.get("needs_pagination_confirmation")
        ):
            confirmed_pagination = True
            final_response = await service.ask(
                question=question,
                session_id=session_id,
                page=1,
                page_size=page_size,
                confirm_pagination=True,
            )

    return {
        "question": question,
        "session_id": session_id,
        "selected_entity_name": selected_entity_name,
        "selected_option": selected_option,
        "confirmed_pagination": confirmed_pagination,
        "initial_behavior": classify_behavior(initial_response),
        "final_behavior": classify_behavior(final_response),
        "initial_response": initial_response,
        "final_response": final_response,
        "answer": final_response.get("answer"),
        "needs_pagination_confirmation": final_response.get(
            "needs_pagination_confirmation",
            False,
        ),
        "pagination": final_response.get("pagination"),
        "sources": final_response.get("sources"),
        "disambiguation": final_response.get("disambiguation"),
    }


async def check_database_connection() -> None:
    try:
        async with session_factory() as session:
            await session.execute(text("SELECT 1"))
    except OperationalError as exc:
        raise ValueError(
            "Não foi possível conectar ao PostgreSQL usando o DATABASE_URL atual. "
            "Verifique se o banco está em execução e se a porta 5432 está acessível."
        ) from exc


async def amain() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings()

    if not settings.LANGSMITH_API_KEY:
        raise ValueError(
            "LANGSMITH_API_KEY is required in the environment or .env file."
        )

    client = Client(
        api_key=settings.LANGSMITH_API_KEY,
        api_url="https://api.smith.langchain.com",
    )

    try:
        client.read_dataset(dataset_name=args.dataset_name)
    except LangSmithAuthError as exc:
        raise ValueError(
            "Falha de autenticacao no LangSmith. "
            "Verifique se LANGSMITH_API_KEY esta correta e ativa."
        ) from exc

    await check_database_connection()

    experiment_prefix = (
        f"{args.experiment_prefix}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    )

    results = await aevaluate(
        run_chatbot_target,
        data=args.dataset_name,
        evaluators=CHATBOT_EVALUATORS,
        experiment_prefix=experiment_prefix,
        description=(
            "Avaliacao do chatbot real do CFCGS Tracker com LangSmith."
        ),
        metadata={
            "project": "cfcgs-tracker",
            "component": "chatbot",
            "evaluation_type": "behavioral_contract",
        },
        max_concurrency=args.max_concurrency,
        client=client,
        blocking=True,
        upload_results=True,
    )

    await results.wait()
    print(f"Experiment: {results.experiment_name}")
    print(f"URL: {results.url}")
    print(f"Comparison URL: {await results.get_comparison_url()}")

    dataframe = results.to_pandas()
    print("\nEvaluation preview:")
    preview_columns = [
        column
        for column in [
            "inputs.question",
            "outputs.answer",
            "feedback.behavior_matches.score",
            "feedback.contains_expected_terms.score",
            "feedback.avoids_forbidden_terms.score",
            "feedback.entity_resolution_matches.score",
            "feedback.selected_entity_respected.score",
            "feedback.numeric_answer_when_expected.score",
        ]
        if column in dataframe.columns
    ]
    if preview_columns:
        print(dataframe[preview_columns].head(20).to_string(index=False))
    else:
        print(dataframe.head(20).to_string(index=False))


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
