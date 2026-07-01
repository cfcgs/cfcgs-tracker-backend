# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langsmith import Client
from langsmith.utils import LangSmithAuthError

from src.cfcgs_tracker.settings import Settings


DEFAULT_DATASET_NAME = "cfcgs-chatbot-eval-dataset"
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "evals"
    / "langsmith"
    / "chatbot_dataset.json"
)


def load_examples(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open(encoding="utf-8") as file:
        examples = json.load(file)

    if not isinstance(examples, list):
        raise ValueError("Dataset file must contain a JSON array.")

    normalized_examples: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        if not isinstance(example, dict):
            raise ValueError(
                f"Example at position {index} must be an object."
            )
        inputs = example.get("inputs")
        outputs = example.get("outputs")
        metadata = example.get("metadata", {})

        if not isinstance(inputs, dict) or not inputs:
            raise ValueError(
                f"Example at position {index} must contain non-empty inputs."
            )
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError(
                f"Example at position {index} must contain non-empty outputs."
            )
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Example at position {index} must contain object metadata."
            )

        normalized_examples.append(
            {
                "inputs": inputs,
                "outputs": outputs,
                "metadata": metadata,
            }
        )
    return normalized_examples


def get_existing_dataset(client: Client, dataset_name: str):
    return next(
        client.list_datasets(dataset_name=dataset_name, limit=1),
        None,
    )


def create_or_replace_dataset(
    *,
    client: Client,
    dataset_name: str,
    examples: list[dict[str, Any]],
    replace: bool,
) -> None:
    existing_dataset = get_existing_dataset(client, dataset_name)

    if existing_dataset and replace:
        client.delete_dataset(dataset_id=existing_dataset.id)
        existing_dataset = None
        print(f"Deleted existing dataset: {dataset_name}")

    if not existing_dataset:
        client.create_dataset(
            dataset_name=dataset_name,
            description=(
                "Dataset de avaliacao do chatbot CFCGS para o TCC, "
                "com casos de comportamento, desambiguacao, seguranca, "
                "paginacao e consultas analiticas."
            ),
            metadata={
                "project": "cfcgs-tracker",
                "use_case": "chatbot_evaluation",
                "language": "pt-BR",
            },
        )
        print(f"Created dataset: {dataset_name}")
    else:
        print(f"Using existing dataset: {dataset_name}")

    client.create_examples(
        dataset_name=dataset_name,
        examples=examples,
    )
    print(
        f"Uploaded {len(examples)} examples to dataset '{dataset_name}'."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or replace the LangSmith chatbot dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="LangSmith dataset name.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the local JSON dataset file.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete an existing dataset with the same name before upload.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings()

    dataset_path = args.dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not settings.LANGSMITH_API_KEY:
        raise ValueError(
            "LANGSMITH_API_KEY is required in the environment or .env file."
        )

    examples = load_examples(dataset_path)
    dataset_name = args.dataset_name or DEFAULT_DATASET_NAME
    client = Client(
        api_key=settings.LANGSMITH_API_KEY,
        api_url="https://api.smith.langchain.com",
    )
    try:
        create_or_replace_dataset(
            client=client,
            dataset_name=dataset_name,
            examples=examples,
            replace=args.replace,
        )
    except LangSmithAuthError as exc:
        raise ValueError(
            "Falha de autenticacao no LangSmith. "
            "Verifique se LANGSMITH_API_KEY esta correta e ativa."
        ) from exc


if __name__ == "__main__":
    main()
