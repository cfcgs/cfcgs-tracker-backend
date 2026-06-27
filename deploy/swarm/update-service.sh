#!/bin/sh
set -eu

SERVICE_NAME=${1:-}
IMAGE_NAME=${2:-}

: "${STACK_NAME:=cfcgs}"

if [ -z "$SERVICE_NAME" ] || [ -z "$IMAGE_NAME" ]; then
    echo "Usage: $0 <service-name> <image-name>"
    exit 1
fi

docker service update \
    --with-registry-auth \
    --image "$IMAGE_NAME" \
    "${STACK_NAME}_${SERVICE_NAME}"
