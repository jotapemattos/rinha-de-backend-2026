#!/usr/bin/env bash
set -euo pipefail

IMAGE="jotapemattos/rinha-de-backend-2026"
TAG="${TAG:-latest}"

echo "Building and pushing ${IMAGE}:${TAG} for linux/amd64..."

docker buildx build \
  --platform linux/amd64 \
  -t "${IMAGE}:${TAG}" \
  --push \
  .

echo "Done — pushed docker.io/${IMAGE}:${TAG}"
