#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACT_DIR="${PLAN_E_ARTIFACT_DIR:-artifacts/plan-e/$STAMP}"
GPU_ID="${PLAN_E_GPU_ID:-${CUDA_VISIBLE_DEVICES:-2}}"
mkdir -p "$ARTIFACT_DIR"

run_step() {
  local name="$1"
  shift
  echo "==> $name"
  {
    echo "$ $*"
    "$@"
  } 2>&1 | tee "$ARTIFACT_DIR/$name.log"
}

run_step env_contract pytest llm_router/tests/integration/test_env_contract.py -v -s --tb=short
run_step cpu_regression pytest llm_router/ -v --tb=short
run_step ruff_check ruff check llm_router

if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" run_step gpu_manager_parity \
    pytest llm_router/tests/test_manager_parity.py -v -s --tb=short
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" run_step mooncake_store \
  pytest llm_router/connector/tests/test_mooncake_store.py -v -s --tb=short

echo "Plan E validation logs: $ARTIFACT_DIR"
