#!/usr/bin/env bash
# Build the DeepSeek Harness sidecar tool image.
#
# The image installs the @deepseek-ai/dsh npm package plus a node runtime into a
# `FROM scratch` layer rooted at /opt/dsh, and bakes minimal_patch.yml at
# /opt/dsh/minimal_patch.yml. It is mounted into the SWE-bench sandbox at
# /opt/dsh, so the sandbox base image does not need node to run the agent.
#
# Usage:
#   bash examples/deepseek_harness/build_tool.sh
#   bash examples/deepseek_harness/build_tool.sh --npm-registry https://registry.npmmirror.com
#   bash examples/deepseek_harness/build_tool.sh --dsh-version 0.1.0-rc.8
#   bash examples/deepseek_harness/build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${TOOL_IMAGE:-dsh-tool}"
IMAGE_TAG="${TOOL_TAG:-latest}"
# Pinned to the version minimal_patch.yml was measured against (its row IDs are
# dsh internals); override with --dsh-version to build against another release.
DSH_VERSION="${DSH_VERSION:-0.1.0-rc.8}"
BASE_IMAGE="${BASE_IMAGE:-}"

# Parse args
REGISTRY=""
NPM_REGISTRY="${NPM_REGISTRY:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry) REGISTRY="$2"; shift 2 ;;
        --npm-registry) NPM_REGISTRY="$2"; shift 2 ;;
        --dsh-version) DSH_VERSION="$2"; shift 2 ;;
        --base-image) BASE_IMAGE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BUILD_ARGS=(--build-arg "DSH_VERSION=${DSH_VERSION}")
if [[ -n "${NPM_REGISTRY}" ]]; then
    BUILD_ARGS+=(--build-arg "NPM_REGISTRY=${NPM_REGISTRY}")
fi
if [[ -n "${BASE_IMAGE}" ]]; then
    BUILD_ARGS+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
fi

echo "==> Building deepseek_harness tool image: ${IMAGE_NAME}:${IMAGE_TAG} (dsh@${DSH_VERSION})"
docker build \
    -f "${SCRIPT_DIR}/Dockerfile.dsh-tool" \
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
