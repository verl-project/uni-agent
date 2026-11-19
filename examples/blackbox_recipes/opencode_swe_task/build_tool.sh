#!/usr/bin/env bash
# Build the OpenCode tool image and optionally publish it to a registry.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${TOOL_IMAGE:-opencode-tool}"
IMAGE_TAG="${TOOL_TAG:-1.19.3}"
REGISTRY=""

# Pass any remaining options directly to docker build (for example, --build-arg).
if [[ "${1:-}" == "--registry" ]]; then
  REGISTRY="${2:?Usage: build_tool.sh [--registry registry.example.com/team] [docker build options]}"
  shift 2
fi

IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
docker build --platform linux/amd64 --target tool \
  --file "$SCRIPT_DIR/Dockerfile.opencode-tool" --tag "$IMAGE" "$@" "$SCRIPT_DIR"

if [[ -n "$REGISTRY" ]]; then
  FULL_TAG="${REGISTRY}/${IMAGE}"
  docker tag "$IMAGE" "$FULL_TAG"
  docker push "$FULL_TAG"
fi
