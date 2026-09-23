#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${TOOL_IMAGE:-swr.cn-east-3.myhuaweicloud.com/openyuanrong/openclaw-tool:2026.9.2}"
# Build only. Authentication and publication are separate explicit operations.
BUILD_ARGS=(--target tool --file "$SCRIPT_DIR/Dockerfile.openclaw-tool" --tag "$IMAGE")
if [[ -n "${BUILD_NETWORK:-}" ]]; then
  BUILD_ARGS+=(--network "${BUILD_NETWORK}")
fi
docker build "${BUILD_ARGS[@]}" "$SCRIPT_DIR"
docker image inspect "$IMAGE" --format "{{.Id}} {{.Size}}"
