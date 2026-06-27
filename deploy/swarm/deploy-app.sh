#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)

: "${STACK_NAME:=cfcgs}"
: "${DOMAIN:?DOMAIN must be set}"
: "${BACKEND_IMAGE:?BACKEND_IMAGE must be set}"
: "${FRONTEND_IMAGE:?FRONTEND_IMAGE must be set}"
: "${DB_USER:?DB_USER must be set}"
: "${DB_PASSWORD:?DB_PASSWORD must be set}"
: "${DB_NAME:?DB_NAME must be set}"
: "${EXPECTED_COLUMNS:?EXPECTED_COLUMNS must be set}"
: "${SIMILARITY_THRESHOLD:?SIMILARITY_THRESHOLD must be set}"
: "${REGIONS:?REGIONS must be set}"
: "${GEMINI_API_KEY:?GEMINI_API_KEY must be set}"
: "${SECRET_KEY:?SECRET_KEY must be set}"
: "${ALGORITHM:?ALGORITHM must be set}"
: "${ACCESS_TOKEN_EXPIRE_MINUTES:?ACCESS_TOKEN_EXPIRE_MINUTES must be set}"
: "${REFRESH_TOKEN_EXPIRE_DAYS:?REFRESH_TOKEN_EXPIRE_DAYS must be set}"
: "${INITIAL_ADMIN_USERNAME:?INITIAL_ADMIN_USERNAME must be set}"
: "${INITIAL_ADMIN_EMAIL:?INITIAL_ADMIN_EMAIL must be set}"
: "${INITIAL_ADMIN_PASSWORD:?INITIAL_ADMIN_PASSWORD must be set}"
: "${CHATBOT_RATE_LIMIT_REQUESTS:=30}"
: "${CHATBOT_RATE_LIMIT_WINDOW_SECONDS:=60}"
: "${CHATBOT_RATE_LIMIT_ENABLED:=true}"
: "${HEATMAP_DYNAMIC_FILTERS_ENABLED:=false}"

docker network create --driver overlay --attachable traefik-public >/dev/null 2>&1 || true
docker stack deploy --with-registry-auth -c "$ROOT_DIR/deploy/swarm/app-stack.yml" "$STACK_NAME"
