#!/usr/bin/env bash
set -euo pipefail

# ========= Configurable settings =========
ROOT="/path/frameworks/uni-agent"
MODEL_PATH="/path/models/Qwen3-VL-4B-Instruct"
DATASET="${ROOT}/examples/openclaw/eval/GSM8K.json"

# Eval scale. Use a small smoke run first, then increase as needed.
NUM_PROBLEMS="${NUM_PROBLEMS:-36}"
MAX_TURNS="${MAX_TURNS:-8}"

# RL training endpoint.
RL_HOST="0.0.0.0"
RL_PORT="30000"
RL_API_KEY="rl-local-token"

# user_llm(vLLM) endpoint.
USER_LLM_HOST="0.0.0.0"
USER_LLM_PORT="30001"
USER_LLM_API_KEY="user-llm-local-token"
USER_LLM_MODEL_NAME="qwen3-vl-2b-user-llm"

# Total RL rollout steps. Set large enough to cover the full eval run.
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-2000}"

# Bound the prompt sent to the external role LLM.
OPENCLAW_DRIVER_HISTORY_MAX_CHARS="${OPENCLAW_DRIVER_HISTORY_MAX_CHARS:-20000}"
OPENCLAW_AGENT_REPLY_MAX_CHARS="${OPENCLAW_AGENT_REPLY_MAX_CHARS:-12000}"

# Sampling params.
OPENCLAW_DRIVER_TEMPERATURE="${OPENCLAW_DRIVER_TEMPERATURE:-0.7}"
OPENCLAW_DRIVER_TOP_P="${OPENCLAW_DRIVER_TOP_P:-0.8}"
OPENCLAW_DRIVER_MAX_TOKENS="${OPENCLAW_DRIVER_MAX_TOKENS:-2048}"
OPENCLAW_DRIVER_REPETITION_PENALTY="${OPENCLAW_DRIVER_REPETITION_PENALTY:-1.1}"
OPENCLAW_AGENT_TEMPERATURE="${OPENCLAW_AGENT_TEMPERATURE:-0.7}"
OPENCLAW_AGENT_TOP_P="${OPENCLAW_AGENT_TOP_P:-0.8}"
OPENCLAW_AGENT_MAX_TOKENS="${OPENCLAW_AGENT_MAX_TOKENS:-4096}"
OPENCLAW_AGENT_REPETITION_PENALTY="${OPENCLAW_AGENT_REPETITION_PENALTY:-1.1}"

# ========= Pre-flight checks =========
cd "${ROOT}"
export PYTHONPATH="${ROOT}"

if [[ ! -f "${DATASET}" ]]; then
  echo "[ERROR] DATASET not found: ${DATASET}"
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi

mkdir -p logs
RL_LOG="logs/rl_eval_rl_server.log"
USER_LLM_LOG="logs/rl_eval_user_llm.log"

stop_process_group() {
  local name="$1"
  local pid="${2:-}"

  if [[ -z "${pid}" ]]; then
    return
  fi

  if ! kill -0 -- "-${pid}" 2>/dev/null; then
    return
  fi

  echo "[CLEANUP] stopping ${name} process group: ${pid}"
  kill -TERM -- "-${pid}" 2>/dev/null || true

  for _ in $(seq 1 15); do
    if ! kill -0 -- "-${pid}" 2>/dev/null; then
      return
    fi
    sleep 1
  done

  echo "[CLEANUP] force stopping ${name} process group: ${pid}"
  kill -KILL -- "-${pid}" 2>/dev/null || true
}

stop_stale_processes() {
  pkill -TERM -f "uni_agent.openclaw.rl.train_entry" 2>/dev/null || true
  pkill -TERM -f "vllm serve" 2>/dev/null || true
  pkill -TERM -f "VLLM::EngineCore" 2>/dev/null || true
  pkill -TERM -f "VLLM::Worker" 2>/dev/null || true
  sleep 2
  pkill -KILL -f "uni_agent.openclaw.rl.train_entry" 2>/dev/null || true
  pkill -KILL -f "vllm serve" 2>/dev/null || true
  pkill -KILL -f "VLLM::EngineCore" 2>/dev/null || true
  pkill -KILL -f "VLLM::Worker" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  echo "[CLEANUP] stopping background services..."
  stop_process_group "user_llm(vLLM)" "${USER_LLM_PID:-}"
  stop_process_group "RL service" "${RL_PID:-}"
  stop_stale_processes
  ray stop --force >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

echo "[STEP 0] Cleaning up stale processes"
stop_stale_processes
ray stop --force >/dev/null 2>&1 || true
sleep 2

# ========= Start RL service (GPU 0,1,2) =========
echo "[STEP 1] Starting RL service (GPU 0,1,2)"
setsid env \
  CUDA_VISIBLE_DEVICES=0,1,2 \
  NUM_GPUS=3 \
  ACTOR_GPUS=1 \
  ROLLOUT_GPUS=1 \
  PRM_GPUS=1 \
  MODEL_PATH="${MODEL_PATH}" \
  PRM_MODEL_PATH="${MODEL_PATH}" \
  OPENCLAW_RL_HOST="${RL_HOST}" \
  OPENCLAW_RL_PORT="${RL_PORT}" \
  OPENCLAW_RL_API_KEY="${RL_API_KEY}" \
  TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS}" \
  PPO_MINI_BSZ=8 \
  MICRO_BATCH_SIZE=1 \
  PYTHONPATH="${PYTHONPATH}" \
  bash examples/openclaw/train_rl.sh > "${RL_LOG}" 2>&1 &
RL_PID=$!

echo "[WAIT] Waiting for RL service readiness..."
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${RL_PORT}/healthz" >/dev/null; then
    echo "[OK] RL service is ready: http://127.0.0.1:${RL_PORT}"
    break
  fi
  sleep 2
  if [[ $i -eq 180 ]]; then
    echo "[ERROR] RL service startup timed out. Check log: ${RL_LOG}"
    exit 1
  fi
done

# ========= Start user_llm(vLLM) (GPU 3) =========
echo "[STEP 2] Starting user_llm(vLLM) (GPU 3)"
setsid env \
  CUDA_VISIBLE_DEVICES=3 \
  MODEL_PATH="${MODEL_PATH}" \
  HOST="${USER_LLM_HOST}" \
  PORT="${USER_LLM_PORT}" \
  TP_SIZE=1 \
  MAX_TOKENS=32768 \
  MODEL_NAME="${USER_LLM_MODEL_NAME}" \
  VLLM_API_KEY="${USER_LLM_API_KEY}" \
  PYTHONPATH="${PYTHONPATH}" \
  bash examples/openclaw/eval/launch_user_llm.sh > "${USER_LLM_LOG}" 2>&1 &
USER_LLM_PID=$!

echo "[WAIT] Waiting for user_llm(vLLM) readiness..."
for i in $(seq 1 180); do
  if curl -sf -H "Authorization: Bearer ${USER_LLM_API_KEY}" "http://127.0.0.1:${USER_LLM_PORT}/v1/models" >/dev/null; then
    echo "[OK] user_llm is ready: http://127.0.0.1:${USER_LLM_PORT}/v1"
    break
  fi
  sleep 2
  if [[ $i -eq 180 ]]; then
    echo "[ERROR] user_llm startup timed out. Check log: ${USER_LLM_LOG}"
    exit 1
  fi
done

# ========= Run the three eval stages =========
echo "[STEP 3] Running eval (student -> TA -> teacher)"
cd "${ROOT}/examples/openclaw/eval"

# RL service under evaluation.
export OPENCLAW_GATEWAY_URL="http://127.0.0.1:${RL_PORT}"
export OPENCLAW_GATEWAY_TOKEN="${RL_API_KEY}"
export OPENCLAW_ENDPOINT_MODE="gateway"
export OPENCLAW_AGENT_MODEL="default"
export OPENCLAW_WORKSPACE="${HOME}/.openclaw/workspace"

# External role LLM served by local vLLM.
export OPENAI_BASE_URL="http://127.0.0.1:${USER_LLM_PORT}/v1"
export OPENAI_API_KEY="${USER_LLM_API_KEY}"
export EXTERNAL_MODEL="${USER_LLM_MODEL_NAME}"
export OPENCLAW_DRIVER_HISTORY_MAX_CHARS
export OPENCLAW_AGENT_REPLY_MAX_CHARS
export OPENCLAW_DRIVER_TEMPERATURE OPENCLAW_DRIVER_TOP_P OPENCLAW_DRIVER_MAX_TOKENS OPENCLAW_DRIVER_REPETITION_PENALTY
export OPENCLAW_AGENT_TEMPERATURE OPENCLAW_AGENT_TOP_P OPENCLAW_AGENT_MAX_TOKENS OPENCLAW_AGENT_REPETITION_PENALTY

echo "[RUN] student_chat.py"
python student_chat.py --dataset "${DATASET}" --num-problems "${NUM_PROBLEMS}" --max-turns "${MAX_TURNS}"

echo "[RUN] TA_chat.py"
python TA_chat.py --dataset "${DATASET}" --num-problems "${NUM_PROBLEMS}" --max-turns "${MAX_TURNS}"

echo "[RUN] teacher_chat.py"
python teacher_chat.py --dataset "${DATASET}" --num-problems "${NUM_PROBLEMS}" --max-turns "${MAX_TURNS}"

echo "[DONE] RL eval pipeline completed"
echo "  RL log: ${ROOT}/${RL_LOG}"
echo "  user_llm log: ${ROOT}/${USER_LLM_LOG}"
echo "  Eval outputs: ${ROOT}/examples/openclaw/eval/results_student.txt, results_TA.txt, results_teacher.txt"