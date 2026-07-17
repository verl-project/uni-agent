#!/usr/bin/env bash
# Build the Claude Code sidecar tool image.
#
# The image installs the @anthropic-ai/claude-code npm package into a minimal
# `FROM scratch` layer rooted at /opt/claude-code. It is mounted into the
# SWE-bench sandbox at /opt/claude-code, so the sandbox base image does not need
# Node or npm to run the agent.
#
# Usage:
#   bash examples/blackbox_recipes/claude_code/build_tool.sh
#   bash examples/blackbox_recipes/claude_code/build_tool.sh --npm-registry https://registry.npmmirror.com
#   bash examples/blackbox_recipes/claude_code/build_tool.sh --tool-version latest
#   bash examples/blackbox_recipes/claude_code/build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${TOOL_IMAGE:-claude-code-tool}"
IMAGE_TAG="${TOOL_TAG:-latest}"
TOOL_VERSION="${TOOL_VERSION:-latest}"

# Parse args
REGISTRY=""
NPM_REGISTRY="${NPM_REGISTRY:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry) REGISTRY="$2"; shift 2 ;;
        --npm-registry) NPM_REGISTRY="$2"; shift 2 ;;
        --tool-version) TOOL_VERSION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BUILD_ARGS=(--build-arg "TOOL_VERSION=${TOOL_VERSION}")
if [[ -n "${NPM_REGISTRY}" ]]; then
    BUILD_ARGS+=(--build-arg "NPM_REGISTRY=${NPM_REGISTRY}")
fi

echo "==> Building claude_code tool image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker build \
    -f "${SCRIPT_DIR}/Dockerfile.claude-code-tool" \
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
