from contextlib import asynccontextmanager

from fastapi import FastAPI
from http import HTTPStatus

from sqlalchemy.orm import Session
from starlette.middleware.cors import CORSMiddleware

from src.cfcgs_tracker.database.database import engine
from src.cfcgs_tracker.database.seeding import create_initial_regions
from src.cfcgs_tracker.schemas import Message
from src.cfcgs_tracker.routers import (
    funds,
    fund_types,
    fund_focuses,
    fund_projects,
    commitments,
    countries,
    regions,
    chatbot,
    projects
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Aplicação iniciando...")
    with Session(engine) as session:
        create_initial_regions(session)
    yield
    print("Aplicação encerrando.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chatbot.router)
app.include_router(regions.router)
app.include_router(countries.router)
app.include_router(commitments.router)
app.include_router(funds.router)
app.include_router(fund_types.router)
app.include_router(fund_focuses.router)
app.include_router(fund_projects.router)
app.include_router(projects.router)


@app.get("/", status_code=HTTPStatus.OK, response_model=Message)
def read_root():
    return {"message": "Hello World"}
