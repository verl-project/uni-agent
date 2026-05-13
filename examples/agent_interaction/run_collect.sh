#!/usr/bin/env bash
# Collect rollouts for offline tail-latency analysis.
#
# Layout mirrors examples/agent_train/wb_train_sync.sh so the rollout
# distribution matches what training sees, but the driver loops
# `--num-steps` times and never runs backward/optimizer.
#
# After this finishes, run:
#   python examples/agent_interaction/analyze_tail.py \
#       --log-dir "${LOG_DIR}" --top-k 10 \
#       --out-dir "${LOG_DIR}/tail_analysis"
#
# Usage:
#   bash examples/agent_interaction/run_collect.sh
# Override any var via env, e.g.:
#   NUM_STEPS=10 TRAIN_BATCH_SIZE=32 bash examples/agent_interaction/run_collect.sh
set -xeuo pipefail

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}

# ---- data & model (aligned with wb_train_sync.sh) --------------------------
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-30B-A3B-Instruct-xml-template"}
DATA_FILE=${DATA_FILE:-"${RAY_DATA_HOME}/data/swe_agent/swe_rebench_filtered.parquet"}
RUNTIME_ENV=${RUNTIME_ENV:-"/mnt/hdfs/went/data/swe_agent/runtime_env.yaml"}
AGENT_CONFIG_PATH=${AGENT_CONFIG_PATH:-"examples/agent_interaction/agent_config.yaml"}

# ---- cluster shape (8 nodes x 8 GPUs) --------------------------------------
NNODES=${NNODES:-8}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-8}
ENGINE=${ENGINE:-"vllm"}

# ---- batch shape (aligned with training: 64 prompts x 8 responses = 512) ---
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
N_PER_PROMPT=${N_PER_PROMPT:-8}
NUM_STEPS=${NUM_STEPS:-5}
START_STEP=${START_STEP:-0}     # set to resume; with same SHUFFLE_SEED + dataset,
                                # samples will match a continuous run starting at 0.
MAX_TURNS=${MAX_TURNS:-100}
PROMPT_LENGTH=${PROMPT_LENGTH:-4096}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-65536}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}

# ---- output layout ---------------------------------------------------------
# agent_config.yaml's `log_dir` is the worker-local scratch path (typically /tmp).
# After every step the driver pins a ray task on each live node that moves
# <log_dir>/<run_id>/ into ${HDFS_ROOT}/step_<idx>/, so HDFS holds the final,
# step-grouped collection. Local /tmp is drained as we go.
HDFS_ROOT=${HDFS_ROOT:-"/mnt/hdfs/went/logs/swebench_qwen3_coder_infer/run_$(date +%Y%m%d_%H%M%S)"}

ray job submit --no-wait \
    --runtime-env "${RUNTIME_ENV}" \
    --working-dir . \
    -- python3 examples/agent_interaction/parallel_infer.py \
    --data-path "${DATA_FILE}" \
    --model-path "${MODEL_PATH}" \
    --agent-config-path "${AGENT_CONFIG_PATH}" \
    --engine "${ENGINE}" \
    --nnodes "${NNODES}" \
    --n-gpus-per-node "${NGPUS_PER_NODE}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --n "${N_PER_PROMPT}" \
    --num-steps "${NUM_STEPS}" \
    --start-step "${START_STEP}" \
    --shuffle \
    --seed "${SHUFFLE_SEED}" \
    --max-turns "${MAX_TURNS}" \
    --prompt-length "${PROMPT_LENGTH}" \
    --response-length "${RESPONSE_LENGTH}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --hdfs-root "${HDFS_ROOT}" \
    --max-samples 0
