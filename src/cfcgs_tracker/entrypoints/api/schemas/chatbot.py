from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatQuery(BaseModel):
    question: str
    session_id: str = "default"
    page: int = Field(1, ge=1)
    page_size: int = Field(10, ge=1, le=100)
    confirm_pagination: bool = False
    disambiguation_choice: dict[str, str] | None = None


class PaginationResult(BaseModel):
    page: int
    page_size: int
    total_rows: int
    has_more: bool
    rows: list[dict[str, Any]]


class ChatSource(BaseModel):
    name: str
    url: str


class DisambiguationOption(BaseModel):
    name: str
    kind: str


class DisambiguationPayload(BaseModel):
    message: str
    options: list[DisambiguationOption]
    mode: str = "select"


class ChatResponse(BaseModel):
    answer: str
    needs_pagination_confirmation: bool = False
    pagination: PaginationResult | None = None
    sources: list[ChatSource] | None = None
    disambiguation: DisambiguationPayload | None = None
