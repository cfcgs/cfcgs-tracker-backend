from collections import defaultdict, deque
import asyncio
import time

from fastapi import FastAPI
from http import HTTPStatus

from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.cfcgs_tracker.entrypoints.api.routers import (
    auth,
    beneficiary_countries,
    chatbot,
    fund_focuses,
    funding_providers,
    fund_types,
    imports,
    projects,
    records,
    users,
)
from src.cfcgs_tracker.entrypoints.api.schemas.common import Message
from src.cfcgs_tracker.settings import Settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        max_requests: int,
        window_seconds: int,
        enabled: bool = True,
        target_path: str = "/chatbot/query",
    ) -> None:
        super().__init__(app)
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = max(1, int(window_seconds))
        self.enabled = enabled
        self.target_path = target_path
        self._lock = asyncio.Lock()
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if not self.enabled or request.method == "OPTIONS":
            return await call_next(request)

        if not request.url.path.endswith(self.target_path):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time.monotonic()

        async with self._lock:
            queue = self._requests[client_ip]
            while queue and (now - queue[0]) > self.window_seconds:
                queue.popleft()

            if len(queue) >= self.max_requests:
                retry_after = self._compute_retry_after(queue, now)
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Limite de requisições atingido. Tente novamente em alguns instantes."
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            queue.append(now)

        return await call_next(request)

    def _compute_retry_after(self, queue: deque[float], now: float) -> int:
        if not queue:
            return self.window_seconds
        remaining = self.window_seconds - (now - queue[0])
        return max(1, int(remaining))

    def _get_client_ip(self, request: Request) -> str:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
        if request.client:
            return request.client.host
        return "unknown"


app = FastAPI(root_path="/api")
settings = Settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.CHATBOT_RATE_LIMIT_REQUESTS,
    window_seconds=settings.CHATBOT_RATE_LIMIT_WINDOW_SECONDS,
    enabled=settings.CHATBOT_RATE_LIMIT_ENABLED,
)

app.include_router(users.router)
app.include_router(auth.router)
app.include_router(imports.router)
app.include_router(fund_types.router)
app.include_router(fund_focuses.router)
app.include_router(funding_providers.router)
app.include_router(beneficiary_countries.router)
app.include_router(records.router)
app.include_router(projects.router)
app.include_router(chatbot.router)


@app.get("/", status_code=HTTPStatus.OK, response_model=Message)
def read_root():
    return {"message": "Hello World"}
