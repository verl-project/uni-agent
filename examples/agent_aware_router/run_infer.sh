#!/usr/bin/env bash
# KV-cache-aware router agent inference (verl framework + TQ path).
# Thin wrapper: exports the Ray-worker env, forwards all flags to
# run_infer.py. See `bash run_infer.sh --help` for the full flag set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# uni_agent resolves from the repo-root package; verl from its installed package.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ── Ray-worker env (read via os.environ, not CLI flags) ──────────────────
# OpenYuanrong sandbox creds (only provider with reverse-tunnel support).
export OPENYUANRONG_SERVER_ADDRESS="${OPENYUANRONG_SERVER_ADDRESS:-}"
export OPENYUANRONG_TOKEN="${OPENYUANRONG_TOKEN:-}"
export OPENYUANRONG_TUNNEL_SSL_VERIFY="${OPENYUANRONG_TUNNEL_SSL_VERIFY:-0}"
export SANDBOX_NAME_PREFIX="${SANDBOX_NAME_PREFIX:-mini-swe-}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export RL_INSIGHT_SERVER_URL="${RL_INSIGHT_SERVER_URL:-http://127.0.0.1:18080}"

python examples/agent_aware_router/run_infer.py "$@"
