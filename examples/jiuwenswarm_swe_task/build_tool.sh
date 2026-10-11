#!/usr/bin/env bash
# Build the jiuwenswarm sidecar tool image.
#
# The image uses python-build-standalone to build an isolated Python runtime
# with workswarm + run_agent.sh, copied into a minimal `FROM scratch` final
# stage rooted at /opt/jiuwenswarm. It is mounted into the SWE-bench sandbox
# at /opt/jiuwenswarm, so the sandbox base image does not need Python for
# the sidecar tool runtime.
#
# Usage:
#   bash examples/jiuwenswarm_swe_task/build_tool.sh
#   bash examples/jiuwenswarm_swe_task/build_tool.sh --pip-index https://pypi.tuna.tsinghua.edu.cn/simple/
#   bash examples/jiuwenswarm_swe_task/build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${TOOL_IMAGE:-jiuwenswarm-tool}"
IMAGE_TAG="${TOOL_TAG:-latest}"

# Parse args
REGISTRY=""
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry) REGISTRY="$2"; shift 2 ;;
        --pip-index) PIP_INDEX_URL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BUILD_ARGS=()
if [[ -n "${PIP_INDEX_URL}" ]]; then
    BUILD_ARGS+=(--build-arg PIP_INDEX_URL="${PIP_INDEX_URL}")
fi

echo "==> Building jiuwenswarm tool image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker build \
    -f "${SCRIPT_DIR}/Dockerfile.jiuwenswarm-tool" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${BUILD_ARGS[@]}" \
    "${SCRIPT_DIR}/"

if [[ -n "${REGISTRY}" ]]; then
    FULL_TAG="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo "==> Tagging and pushing: ${FULL_TAG}"
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_TAG}"
    docker push "${FULL_TAG}"
    echo "    Pushed."
fi

echo ""
echo "Tool image ready: ${IMAGE_NAME}:${IMAGE_TAG}"
if [[ -n "${REGISTRY}" ]]; then
    echo "  Remote sandbox: ${FULL_TAG}"
fi
