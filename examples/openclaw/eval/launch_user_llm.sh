#!/usr/bin/env bash
#
# Launch a vLLM OpenAI-compatible API server for the role LLM.
#
# Required environment variables:
#   MODEL_PATH      - Absolute path to model weights directory
#
# Optional environment variables:
#   HOST            - Bind address       (default: 0.0.0.0)
#   PORT            - Listen port        (default: 30001)
#   TP_SIZE         - Tensor parallel    (default: 8)
#   MAX_TOKENS      - Max total tokens   (default: 32768)
#   MODEL_NAME      - served-model-name  (default: qwen3-4b-user-llm)
#   VLLM_API_KEY    - API key for auth   (default: none, no auth)
#   DTYPE           - Model dtype        (default: auto)
#
# Usage:
#   export MODEL_PATH="/data/models/Qwen/Qwen3-4B"
#   bash launch_user_llm.sh

set -euo pipefail

if [ -z "${MODEL_PATH:-}" ]; then
    echo "Error: MODEL_PATH is not set." >&2
    echo "Usage: MODEL_PATH=/path/to/Qwen3-4B bash $0" >&2
    exit 1
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30001}"
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
MODEL_NAME="${MODEL_NAME:-qwen3-4b-user-llm}"
API_KEY="${VLLM_API_KEY:-}"
DTYPE="${DTYPE:-auto}"

API_KEY_ARGS=()
if [ -n "${API_KEY}" ]; then
    API_KEY_ARGS=(--api-key "${API_KEY}")
fi

echo "============================================"
echo "  vLLM OpenAI API Server"
echo "  Model:  ${MODEL_PATH}"
echo "  Host:   ${HOST}:${PORT}"
echo "  TP:     ${TP_SIZE}"
echo "  DType:  ${DTYPE}"
echo "============================================"

# vLLM >= 0.6 CLI
vllm serve "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_TOKENS}" \
    --served-model-name "${MODEL_NAME}" \
    --dtype "${DTYPE}" \
    "${API_KEY_ARGS[@]}"
