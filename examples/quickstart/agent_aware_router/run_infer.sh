#!/usr/bin/env bash
# Agent Aware Router agent inference (verl framework + TQ path).
# Thin wrapper: exports the Ray-worker env, forwards all flags to
# run_infer.py. See `bash run_infer.sh --help` for the full flag set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

python examples/agent_aware_router/run_infer.py "$@"
