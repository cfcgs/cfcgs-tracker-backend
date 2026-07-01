from http import HTTPStatus

import pytest
from fastapi import HTTPException

from src.cfcgs_tracker.adapters.llm.chatbot_agent import UnsafeSQLQueryError
from src.cfcgs_tracker.entrypoints.api.routers.chatbot import ask_chatbot
from src.cfcgs_tracker.entrypoints.api.schemas.chatbot import ChatQuery


class FakeChatbotService:
    def __init__(self, *, response=None, error: Exception | None = None) -> None:
        self.response = response or {
            "session_id": "session-1",
            "answer": "ok",
            "needs_pagination_confirmation": False,
            "pagination": None,
            "sources": None,
            "disambiguation": None,
        }
        self.error = error
        self.calls = []

    async def ask(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


@pytest.mark.asyncio
async def test_ask_chatbot_returns_service_response_and_forwards_choice():
    service = FakeChatbotService()
    query = ChatQuery.model_validate(
        {
            "question": "Quanto recebeu o Brasil?",
            "page": 2,
            "page_size": 5,
            "confirm_pagination": True,
            "disambiguationChoice": {
                "id": 1,
                "name": "Brazil",
                "kind": "country",
            },
        }
    )

    payload = await ask_chatbot(query, service)

    assert payload["answer"] == "ok"
    assert service.calls[0]["page"] == 2
    assert service.calls[0]["page_size"] == 5
    assert service.calls[0]["confirm_pagination"] is True
    assert service.calls[0]["disambiguation_choice"] == {
        "id": 1,
        "name": "Brazil",
        "kind": "country",
    }


@pytest.mark.asyncio
async def test_ask_chatbot_maps_unsafe_sql_error_to_bad_request():
    service = FakeChatbotService(error=UnsafeSQLQueryError("sql inseguro"))
    query = ChatQuery(question="teste")

    with pytest.raises(HTTPException) as exc_info:
        await ask_chatbot(query, service)

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
    assert exc_info.value.detail == "sql inseguro"


@pytest.mark.asyncio
async def test_ask_chatbot_maps_value_error_to_bad_request():
    service = FakeChatbotService(error=ValueError("entrada invalida"))
    query = ChatQuery(question="teste")

    with pytest.raises(HTTPException) as exc_info:
        await ask_chatbot(query, service)

    assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST
    assert exc_info.value.detail == "entrada invalida"


@pytest.mark.asyncio
async def test_ask_chatbot_maps_unexpected_error_to_internal_server_error():
    service = FakeChatbotService(error=RuntimeError("boom"))
    query = ChatQuery(question="teste")

    with pytest.raises(HTTPException) as exc_info:
        await ask_chatbot(query, service)

    assert exc_info.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Unexpected chatbot error."
