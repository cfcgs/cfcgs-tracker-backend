from http import HTTPStatus
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.cfcgs_tracker.adapters.llm.chatbot_agent import UnsafeSQLQueryError
from src.cfcgs_tracker.entrypoints.api.dependencies import get_chatbot_service
from src.cfcgs_tracker.entrypoints.api.schemas.chatbot import (
    ChatQuery,
    ChatResponse,
)
from src.cfcgs_tracker.service_layer.services.chatbot import ChatbotService

router = APIRouter(prefix="/chatbot", tags=["chatbot"])
ChatbotServiceDep = Annotated[ChatbotService, Depends(get_chatbot_service)]
logger = logging.getLogger("uvicorn.error")


@router.post(
    "/query",
    response_model=ChatResponse,
    status_code=HTTPStatus.OK,
)
async def ask_chatbot(
    query: ChatQuery,
    service: ChatbotServiceDep,
):
    try:
        logger.info(
            "[chatbot] incoming request | session_id=%s | page=%s | page_size=%s | confirm_pagination=%s | question=%s",
            query.session_id,
            query.page,
            query.page_size,
            query.confirm_pagination,
            query.question,
        )
        return await service.ask(
            question=query.question,
            session_id=query.session_id,
            page=query.page,
            page_size=query.page_size,
            confirm_pagination=query.confirm_pagination,
            disambiguation_choice=query.disambiguation_choice,
        )
    except UnsafeSQLQueryError as exc:
        logger.warning(
            "[chatbot] bad request unsafe sql | session_id=%s | question=%s | detail=%s",
            query.session_id,
            query.question,
            str(exc),
        )
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        logger.warning(
            "[chatbot] bad request value error | session_id=%s | question=%s | detail=%s",
            query.session_id,
            query.question,
            str(exc),
        )
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "[chatbot] unexpected error | session_id=%s | question=%s",
            query.session_id,
            query.question,
        )
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Unexpected chatbot error.",
        ) from exc
