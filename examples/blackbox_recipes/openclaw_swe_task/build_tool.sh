#!/usr/bin/env bash
# Build and optionally publish the OpenClaw sidecar image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${TOOL_IMAGE:-openclaw-tool}"
IMAGE_TAG="${TOOL_TAG:-2026.9.2}"
OPENCLAW_VERSION="${OPENCLAW_VERSION:-2026.9.2}"
REGISTRY=""
BUILD_NETWORK="${BUILD_NETWORK:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="$2"; shift 2 ;;
    --tag) IMAGE_TAG="$2"; shift 2 ;;
    --version) OPENCLAW_VERSION="$2"; shift 2 ;;
    --network) BUILD_NETWORK="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
BUILD_ARGS=(
  --target tool
  --file "$SCRIPT_DIR/Dockerfile.openclaw-tool"
  --tag "$IMAGE"
  --build-arg "OPENCLAW_VERSION=${OPENCLAW_VERSION}"
)
if [[ -n "${BUILD_NETWORK}" ]]; then
  BUILD_ARGS+=(--network "${BUILD_NETWORK}")
fi

echo "==> Building OpenClaw tool image ${IMAGE} (OpenClaw ${OPENCLAW_VERSION})"
docker build "${BUILD_ARGS[@]}" "$SCRIPT_DIR"
docker image inspect "$IMAGE" --format "{{.Id}} {{.Size}}"

if [[ -n "${REGISTRY}" ]]; then
  FULL_TAG="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
  docker tag "$IMAGE" "$FULL_TAG"
  docker push "$FULL_TAG"
  echo "Pushed: $FULL_TAG"
fi
echo "Tool image ready: ${IMAGE}"
