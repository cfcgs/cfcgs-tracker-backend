#!/bin/sh

# executa as migraç`oes do banco de dados
poetry run alembic upgrade head

# Inicia a aplicação
poetry run uvicorn --host 0.0.0.0 --port 8000 --app-dir /cfcgs-tracker/src cfcgs_tracker.app:app