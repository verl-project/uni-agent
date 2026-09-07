#!/usr/bin/env bash
# Codex sidecar entrypoint. The outer OpenYuanRong sandbox is the security boundary.
set -uo pipefail

TOOL_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PROJECT_DIR="${CODEX_PROJECT_DIR:-${PWD}}"
CODEX_HOME="${CODEX_HOME:-/tmp/codex-home}"
MODEL="${CODEX_MODEL:?missing CODEX_MODEL}"
API_BASE="${CODEX_API_BASE:?missing CODEX_API_BASE}"
API_KEY="${CODEX_API_KEY:-EMPTY}"

mkdir -p "${CODEX_HOME}"
cat >"${CODEX_HOME}/config.toml" <<EOF
model_provider = "gateway"
model = "${MODEL}"
disable_response_storage = true
check_for_update_on_startup = false

[model_providers.gateway]
name = "Uni-Agent Gateway"
base_url = "${API_BASE}"
wire_api = "responses"
requires_openai_auth = true
EOF

export CODEX_HOME
export OPENAI_API_KEY="${API_KEY}"
export CODEX_MANAGED_PACKAGE_ROOT="${CODEX_MANAGED_PACKAGE_ROOT:-${TOOL_ROOT}}"
export NO_PROXY="*"
export no_proxy="*"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

cd "${PROJECT_DIR}"

# Prompt comes from stdin. With no positional prompt, Codex exec reads piped stdin.
exec "${TOOL_ROOT}/bin/codex" exec \
  --json \
  --ephemeral \
  --skip-git-repo-check \
  --dangerously-bypass-approvals-and-sandbox \
  --cd "${PROJECT_DIR}" \
  --model "${MODEL}" \
  "$@"
