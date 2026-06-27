#!/bin/sh
set -eu

# executa as migrações do banco de dados
poetry run alembic upgrade head
poetry run python -m src.cfcgs_tracker.entrypoints.cli.seed_admin

# Inicia a aplicação
poetry run uvicorn --host 0.0.0.0 --port 8000 --app-dir /cfcgs-tracker-backend/src cfcgs_tracker.app:app
