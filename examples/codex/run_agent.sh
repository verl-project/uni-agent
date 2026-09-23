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

# Keep the caller's proxy configuration and bypass it only for the gateway
# authority used by this episode.
gateway_host="${CODEX_API_BASE#*://}"
gateway_host="${gateway_host%%/*}"
gateway_host="${gateway_host%%:*}"
if [[ -n "${gateway_host}" ]]; then
  export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${gateway_host}"
  export no_proxy="${no_proxy:+${no_proxy},}${gateway_host}"
fi

cd "${PROJECT_DIR}"

# Prompt comes from stdin. With no positional prompt, Codex exec reads piped stdin.
exec "${TOOL_ROOT}/vendor/x86_64-unknown-linux-musl/bin/codex" exec \
  --json \
  --ephemeral \
  --skip-git-repo-check \
  --dangerously-bypass-approvals-and-sandbox \
  --cd "${PROJECT_DIR}" \
  --model "${MODEL}" \
  "$@"
