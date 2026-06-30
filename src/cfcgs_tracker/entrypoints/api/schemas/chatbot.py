from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, Field


class ChatQuery(BaseModel):
    question: str
    session_id: str | None = Field(default_factory=lambda: str(uuid4()))
    page: int = Field(1, ge=1)
    page_size: int = Field(10, ge=1, le=100)
    confirm_pagination: bool = False
    disambiguation_choice: DisambiguationOption | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "disambiguation_choice",
            "disambiguationChoice",
            "resolved_entity",
            "resolvedEntity",
        ),
    )


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
    id: int | None = None
    name: str
    kind: str


class DisambiguationPayload(BaseModel):
    message: str
    options: list[DisambiguationOption]
    mode: str = "select"


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    needs_pagination_confirmation: bool = False
    pagination: PaginationResult | None = None
    sources: list[ChatSource] | None = None
    disambiguation: DisambiguationPayload | None = None
