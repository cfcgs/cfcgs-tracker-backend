from http import HTTPStatus
from typing import Annotated
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.cfcgs_tracker.database.database import get_session
from src.cfcgs_tracker.chatbot.agent import ClimateDataAgent
from src.cfcgs_tracker.schemas import ChatResponse, ChatQuery

router = APIRouter(prefix="/chatbot", tags=["chatbot"])
T_Session = Annotated[Session, Depends(get_session)]

@router.post("/query", response_model=ChatResponse, status_code=HTTPStatus.OK)
def ask_chatbot(query: ChatQuery, session: T_Session):
    """
    Recebe uma pergunta do frontend, passa para o agente e retorna a resposta.
    """
    agent = ClimateDataAgent(db_session=session)
    answer = agent.run(question=query.question)
    return {"answer": answer}