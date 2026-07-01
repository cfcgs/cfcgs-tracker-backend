from __future__ import annotations

import unicodedata
from typing import Any

from langsmith.evaluation import run_evaluator
from langsmith.schemas import Example, Run


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return " ".join(without_accents.lower().split())


def _contains_term(text: str, term: str) -> bool:
    return _normalize_text(term) in _normalize_text(text)


def _get_run_outputs(run: Run) -> dict[str, Any]:
    return _as_dict(run.outputs)


def _get_example_outputs(example: Example | None) -> dict[str, Any]:
    if not example:
        return {}
    return _as_dict(example.outputs)


def _classify_behavior(response: dict[str, Any] | None) -> str:
    if not response:
        return "unknown"
    if response.get("disambiguation"):
        return "entity_resolution"
    if response.get("needs_pagination_confirmation"):
        return "pagination_confirmation"

    answer = str(response.get("answer") or "")
    answer_normalized = _normalize_text(answer)
    if (
        "nao posso" in answer_normalized
        or "so posso responder" in answer_normalized
        or "desculpe" in answer_normalized
        and "dados de financiamento climatico" in answer_normalized
    ):
        return "refusal"
    if response.get("sources"):
        return "sql_answer"
    return "direct"


def _is_numeric_sql_question(example_outputs: dict[str, Any]) -> bool:
    expected_behavior = example_outputs.get("expected_behavior")
    category = example_outputs.get("category")
    return expected_behavior == "sql_answer" and category in {
        "agregacao_simples",
        "ranking",
        "fundos",
    }


@run_evaluator
def behavior_matches(run: Run, example: Example | None) -> dict[str, Any]:
    expected = _get_example_outputs(example).get("expected_behavior")
    outputs = _get_run_outputs(run)
    initial_response = _as_dict(outputs.get("initial_response"))
    final_response = _as_dict(outputs.get("final_response"))
    initial_behavior = _classify_behavior(initial_response)
    final_behavior = _classify_behavior(final_response)
    selected_entity_name = outputs.get("selected_entity_name")
    confirmed_pagination = outputs.get("confirmed_pagination")

    passed = False
    if expected == "entity_resolution":
        passed = initial_behavior == "entity_resolution"
        if selected_entity_name:
            passed = passed and final_behavior in {"sql_answer", "direct"}
    elif expected == "pagination_confirmation":
        passed = initial_behavior == "pagination_confirmation"
        if confirmed_pagination:
            passed = passed and bool(final_response.get("pagination"))
    else:
        passed = final_behavior == expected

    return {
        "key": "behavior_matches",
        "score": 1 if passed else 0,
        "comment": (
            f"expected={expected}, initial={initial_behavior}, "
            f"final={final_behavior}"
        ),
    }


@run_evaluator
def contains_expected_terms(run: Run, example: Example | None) -> dict[str, Any]:
    expected_terms = _get_example_outputs(example).get(
        "expected_answer_contains", []
    )
    answer = str(_get_run_outputs(run).get("answer") or "")

    if not expected_terms:
        return {
            "key": "contains_expected_terms",
            "score": 1,
            "comment": "No expected terms declared for this example.",
        }

    missing_terms = [
        term for term in expected_terms if not _contains_term(answer, term)
    ]
    return {
        "key": "contains_expected_terms",
        "score": 0 if missing_terms else 1,
        "comment": (
            "Missing expected terms: " + ", ".join(missing_terms)
            if missing_terms
            else "All expected terms found in the answer."
        ),
    }


@run_evaluator
def avoids_forbidden_terms(run: Run, example: Example | None) -> dict[str, Any]:
    forbidden_terms = _get_example_outputs(example).get(
        "expected_answer_not_contains",
        [],
    )
    answer = str(_get_run_outputs(run).get("answer") or "")

    if not forbidden_terms:
        return {
            "key": "avoids_forbidden_terms",
            "score": 1,
            "comment": "No forbidden terms declared for this example.",
        }

    found_terms = [
        term for term in forbidden_terms if _contains_term(answer, term)
    ]
    return {
        "key": "avoids_forbidden_terms",
        "score": 0 if found_terms else 1,
        "comment": (
            "Forbidden terms found: " + ", ".join(found_terms)
            if found_terms
            else "No forbidden terms found in the answer."
        ),
    }


@run_evaluator
def entity_resolution_matches(
    run: Run,
    example: Example | None,
) -> dict[str, Any]:
    example_outputs = _get_example_outputs(example)
    expected_entity_type = example_outputs.get("expected_entity_type")
    if not expected_entity_type:
        return {
            "key": "entity_resolution_matches",
            "score": 1,
            "comment": "No entity type expected for this example.",
        }

    outputs = _get_run_outputs(run)
    initial_response = _as_dict(outputs.get("initial_response"))
    disambiguation = _as_dict(initial_response.get("disambiguation"))
    options = disambiguation.get("options") or []
    option_kinds = {
        str(option.get("kind"))
        for option in options
        if isinstance(option, dict) and option.get("kind")
    }
    passed = expected_entity_type in option_kinds or (
        len(option_kinds) == 1 and expected_entity_type in option_kinds
    )

    return {
        "key": "entity_resolution_matches",
        "score": 1 if passed else 0,
        "comment": (
            f"expected entity type={expected_entity_type}, "
            f"available kinds={sorted(option_kinds)}"
        ),
    }


@run_evaluator
def selected_entity_respected(
    run: Run,
    example: Example | None,
) -> dict[str, Any]:
    example_outputs = _get_example_outputs(example)
    expected_name = example_outputs.get("selected_entity_name")
    if not expected_name:
        return {
            "key": "selected_entity_respected",
            "score": 1,
            "comment": "No explicit selected entity expected.",
        }

    outputs = _get_run_outputs(run)
    answer = str(outputs.get("answer") or "")
    selected_option = _as_dict(outputs.get("selected_option"))
    forbidden_terms = example_outputs.get("expected_answer_not_contains", [])

    contains_expected = _contains_term(answer, expected_name)
    selected_matches = (
        _normalize_text(str(selected_option.get("name") or ""))
        == _normalize_text(expected_name)
    )
    contains_forbidden = any(
        _contains_term(answer, term) for term in forbidden_terms
    )
    passed = contains_expected and selected_matches and not contains_forbidden

    return {
        "key": "selected_entity_respected",
        "score": 1 if passed else 0,
        "comment": (
            f"expected selected entity={expected_name}, "
            f"applied={selected_option.get('name')}, "
            f"contains_forbidden={contains_forbidden}"
        ),
    }


@run_evaluator
def numeric_answer_when_expected(
    run: Run,
    example: Example | None,
) -> dict[str, Any]:
    example_outputs = _get_example_outputs(example)
    if not _is_numeric_sql_question(example_outputs):
        return {
            "key": "numeric_answer_when_expected",
            "score": 1,
            "comment": "Example does not require numeric aggregate validation.",
        }

    outputs = _get_run_outputs(run)
    final_response = _as_dict(outputs.get("final_response"))
    pagination = _as_dict(final_response.get("pagination"))
    rows = pagination.get("rows")
    answer = str(outputs.get("answer") or "")
    has_digit = any(char.isdigit() for char in answer)
    has_rows = isinstance(rows, list) and len(rows) > 0
    passed = has_digit or has_rows

    return {
        "key": "numeric_answer_when_expected",
        "score": 1 if passed else 0,
        "comment": (
            "Answer contains numeric signal."
            if passed
            else "Expected a numeric answer but none was found."
        ),
    }


CHATBOT_EVALUATORS = [
    behavior_matches,
    contains_expected_terms,
    avoids_forbidden_terms,
    entity_resolution_matches,
    selected_entity_respected,
    numeric_answer_when_expected,
]
