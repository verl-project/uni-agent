#!/usr/bin/env bash
# Build the pinned Hermes sidecar image.  Push only when --registry is explicit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${TOOL_IMAGE:-hermes-agent-tool}"
IMAGE_TAG="${TOOL_TAG:-20260915}"
REGISTRY=""
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
HERMES_COMMIT="${HERMES_COMMIT:-5eb99eb2844b22ebb723711b8e6a0bbb80bb5f04}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry) REGISTRY="$2"; shift 2 ;;
        --pip-index) PIP_INDEX_URL="$2"; shift 2 ;;
        --tag) IMAGE_TAG="$2"; shift 2 ;;
        --hermes-commit) HERMES_COMMIT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

BUILD_ARGS=(--build-arg "HERMES_COMMIT=${HERMES_COMMIT}")
if [[ -n "${PIP_INDEX_URL}" ]]; then
    BUILD_ARGS+=(--build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}")
fi

echo "==> Building Hermes tool image ${IMAGE_NAME}:${IMAGE_TAG}"
docker build \
    -f "${SCRIPT_DIR}/Dockerfile.hermes-tool" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${BUILD_ARGS[@]}" \
    "${SCRIPT_DIR}"

if [[ -n "${REGISTRY}" ]]; then
    FULL_TAG="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo "==> Tagging and pushing ${FULL_TAG}"
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_TAG}"
    docker push "${FULL_TAG}"
fi

echo "Hermes tool image ready: ${IMAGE_NAME}:${IMAGE_TAG}"
if [[ -n "${REGISTRY}" ]]; then
    echo "Remote sandbox image: ${FULL_TAG}"
fi
