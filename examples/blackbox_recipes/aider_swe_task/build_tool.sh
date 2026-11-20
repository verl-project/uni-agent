#!/usr/bin/env bash
# Build and optionally publish the Aider sidecar tool image.
#
# Usage:
#   bash examples/blackbox_recipes/aider_swe_task/build_tool.sh
#   bash examples/blackbox_recipes/aider_swe_task/build_tool.sh --platform linux/arm64
#   bash examples/blackbox_recipes/aider_swe_task/build_tool.sh --pip-index https://pypi.tuna.tsinghua.edu.cn/simple/
#   bash examples/blackbox_recipes/aider_swe_task/build_tool.sh --registry registry.example.com/team
#
# --platform defaults to the build host's native architecture. On ARM hosts,
# pass --platform linux/arm64 (or omit it) for a native build, or
# --platform linux/amd64 with emulation enabled for x86 sandboxes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${TOOL_IMAGE:-aider-tool}"
IMAGE_TAG="${TOOL_TAG:-0.86.2}"
AIDER_VERSION="${AIDER_VERSION:-0.86.2}"
REGISTRY=""
PLATFORM="${PLATFORM:-}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry) REGISTRY="$2"; shift 2 ;;
        --platform) PLATFORM="$2"; shift 2 ;;
        --pip-index) PIP_INDEX_URL="$2"; shift 2 ;;
        --version) AIDER_VERSION="$2"; IMAGE_TAG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
BUILD_ARGS=(
    --file "$SCRIPT_DIR/Dockerfile.aider-tool"
    --tag "$IMAGE"
    --build-arg "AIDER_VERSION=${AIDER_VERSION}"
)
if [[ -n "${PLATFORM}" ]]; then
    BUILD_ARGS+=(--platform "${PLATFORM}")
fi
if [[ -n "${PIP_INDEX_URL}" ]]; then
    BUILD_ARGS+=(--build-arg PIP_INDEX_URL="${PIP_INDEX_URL}")
fi

echo "==> Building Aider tool image ${IMAGE} (aider-chat ${AIDER_VERSION})"
docker build "${BUILD_ARGS[@]}" "$SCRIPT_DIR"
docker image inspect "$IMAGE" --format '{{.Os}}/{{.Architecture}} {{.Size}} bytes'

if [[ -n "${REGISTRY}" ]]; then
    FULL_TAG="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    docker tag "$IMAGE" "$FULL_TAG"
    docker push "$FULL_TAG"
    echo "Pushed: ${FULL_TAG}"
fi
echo "Tool image ready: ${IMAGE}"
