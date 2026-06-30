from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.cfcgs_tracker.adapters.llm.chatbot_agent import (
    ClimateFinanceChatbotAgent,
)


class ChatbotService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.agent = ClimateFinanceChatbotAgent(session)

    async def ask(
        self,
        *,
        question: str,
        session_id: str | None = None,
        page: int = 1,
        page_size: int = 10,
        confirm_pagination: bool = False,
        disambiguation_choice: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        effective_session_id = session_id.strip() if session_id else ""
        if not effective_session_id:
            effective_session_id = str(uuid4())

        return await self.agent.ask(
            question=question,
            session_id=effective_session_id,
            page=page,
            page_size=page_size,
            confirm_pagination=confirm_pagination,
            disambiguation_choice=disambiguation_choice,
        )
