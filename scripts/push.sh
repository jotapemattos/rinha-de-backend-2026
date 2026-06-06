#!/usr/bin/env bash
set -euo pipefail

IMAGE="jotapemattos/rinha-de-backend-2026"
TAG="${TAG:-latest}"
SEED="${SEED:-42}"

echo "Building and pushing ${IMAGE}:${TAG} for linux/amd64 (SEED=${SEED})..."

docker buildx build \
  --platform linux/amd64 \
  --build-arg "SEED=${SEED}" \
  -t "${IMAGE}:${TAG}" \
  --push \
  .

echo "Done — pushed docker.io/${IMAGE}:${TAG}"
