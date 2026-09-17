#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${BUILD_TARGET:-tool}"
case "$TARGET" in tool|runtime) ;; *) echo "BUILD_TARGET必须为tool或runtime" >&2; exit 2;; esac
if [[ "$TARGET" == tool ]]; then
  IMAGE="${TOOL_IMAGE:-swr.cn-east-3.myhuaweicloud.com/openyuanrong/openclaw-tool:2026.9.2}"
else
  IMAGE="${RUNTIME_IMAGE:-openclaw-recipe-runtime:2026.9.2}"
fi
# Build only. Authentication and publication are separate explicit operations.
BUILD_ARGS=(--target "$TARGET" --file "$SCRIPT_DIR/Dockerfile.openclaw-tool" --tag "$IMAGE")
if [[ -n "${BUILD_NETWORK:-}" ]]; then
  BUILD_ARGS+=(--network "${BUILD_NETWORK}")
fi
docker build "${BUILD_ARGS[@]}" "$SCRIPT_DIR"
docker image inspect "$IMAGE" --format "{{.Id}} {{.Size}}"
