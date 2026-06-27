#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)

: "${ACME_EMAIL:?ACME_EMAIL must be set}"

docker network create --driver overlay --attachable traefik-public >/dev/null 2>&1 || true
docker stack deploy -c "$ROOT_DIR/deploy/swarm/traefik-stack.yml" traefik
