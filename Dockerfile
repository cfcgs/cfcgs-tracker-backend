FROM python:3.12-slim
LABEL authors="jordany"
ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR app/
ENV PYTHONPATH=/app
COPY . .

RUN pip install poetry

RUN poetry config installer.max-workers 10
RUN python -m pip config set global.timeout 600
RUN poetry install --no-interaction --no-ansi

EXPOSE 8000
CMD poetry run alembic upgrade head && poetry run uvicorn --host 0.0.0.0 --app-dir /cfcgs-tracker/src cfcgs_tracker.app:app