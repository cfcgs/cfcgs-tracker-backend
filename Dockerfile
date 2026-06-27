FROM python:3.12-slim

LABEL authors="jordany"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONPATH=/cfcgs-tracker-backend

WORKDIR /cfcgs-tracker-backend

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config installer.max-workers 10 \
    && python -m pip config set global.timeout 600 \
    && poetry install --only main --no-root --no-ansi

COPY alembic.ini entrypoint.sh ./
COPY migrations ./migrations
COPY src ./src

RUN chmod +x /cfcgs-tracker-backend/entrypoint.sh

EXPOSE 8000

CMD ["./entrypoint.sh"]
