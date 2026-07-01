import pytest

from src.cfcgs_tracker.service_layer.services.chatbot import ChatbotService


class FakeAgent:
    def __init__(self) -> None:
        self.calls = []

    async def ask(self, **kwargs):
        self.calls.append(kwargs)
        return {"answer": "ok", "session_id": kwargs["session_id"]}


@pytest.mark.asyncio
async def test_chatbot_service_generates_session_id_when_missing():
    service = ChatbotService(session=None)
    service.agent = FakeAgent()

    response = await service.ask(question="teste")

    assert response["answer"] == "ok"
    assert service.agent.calls[0]["question"] == "teste"
    assert service.agent.calls[0]["session_id"]


@pytest.mark.asyncio
async def test_chatbot_service_normalizes_blank_session_and_forwards_flags():
    service = ChatbotService(session=None)
    service.agent = FakeAgent()

    await service.ask(
        question="teste",
        session_id="   ",
        page=2,
        page_size=20,
        confirm_pagination=True,
        disambiguation_choice={"id": 1, "name": "Brazil", "kind": "country"},
    )

    call = service.agent.calls[0]
    assert call["page"] == 2
    assert call["page_size"] == 20
    assert call["confirm_pagination"] is True
    assert call["disambiguation_choice"]["name"] == "Brazil"


@pytest.mark.asyncio
async def test_chatbot_service_reuses_non_blank_session_id():
    service = ChatbotService(session=None)
    service.agent = FakeAgent()

    await service.ask(question="teste", session_id="thread-123")

    assert service.agent.calls[0]["session_id"] == "thread-123"
