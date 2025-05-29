from fastapi import FastAPI
from http import HTTPStatus

from starlette.middleware.cors import CORSMiddleware

from src.cfcgs_tracker.schemas import Message
from src.cfcgs_tracker.routers import funds, fund_types, fund_focuses

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(funds.router)
app.include_router(fund_types.router)
app.include_router(fund_focuses.router)
@app.get("/", status_code=HTTPStatus.OK, response_model=Message)
def read_root():
    return {"message": "Hello World"}

