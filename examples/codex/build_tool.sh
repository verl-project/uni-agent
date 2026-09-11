#!/usr/bin/env bash
# Build and optionally push the Codex native sidecar image.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${TOOL_IMAGE:-codex-tool}"
IMAGE_TAG="${TOOL_TAG:-0.147.0-direct-stdin}"
CODEX_VERSION="${CODEX_VERSION:-0.147.0}"
NPM_REGISTRY="${NPM_REGISTRY:-}"
REGISTRY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="$2"; shift 2 ;;
    --version) CODEX_VERSION="$2"; shift 2 ;;
    --npm-registry) NPM_REGISTRY="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done
ARGS=(--build-arg "CODEX_VERSION=${CODEX_VERSION}")
if [[ -n "$NPM_REGISTRY" ]]; then ARGS+=(--build-arg "NPM_REGISTRY=${NPM_REGISTRY}"); fi
echo "==> Building ${IMAGE_NAME}:${IMAGE_TAG} (Codex ${CODEX_VERSION})"
docker build -f "$SCRIPT_DIR/Dockerfile.codex-tool" -t "${IMAGE_NAME}:${IMAGE_TAG}" "${ARGS[@]}" "$SCRIPT_DIR"
if [[ -n "$REGISTRY" ]]; then
  FULL_TAG="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
  docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "$FULL_TAG"
  docker push "$FULL_TAG"
  echo "Pushed: $FULL_TAG"
fi
echo "Tool image ready: ${IMAGE_NAME}:${IMAGE_TAG}"
