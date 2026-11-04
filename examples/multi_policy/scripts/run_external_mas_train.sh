#!/usr/bin/env bash
set -euo pipefail

# Launch three-policy external MAS training in separate_async mode.
# Set the model directories and verl-compatible Parquet datasets before running:
#   POLICY_1_MODEL_PATH=/path/to/model_1 \
#   POLICY_2_MODEL_PATH=/path/to/model_2 \
#   POLICY_3_MODEL_PATH=/path/to/model_3 \
#   TRAIN_DATA=/path/to/train.parquet VAL_DATA=/path/to/val.parquet \
#   bash examples/multi_policy/scripts/run_external_mas_train.sh
# Training and resource settings below have defaults and can be overridden
# through environment variables. The external MAS command is configured in
# examples/multi_policy/config/multi_policy.yaml.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

RAY_ADDRESS="${RAY_ADDRESS:-auto}"
POLICY_1_MODEL_PATH="${POLICY_1_MODEL_PATH:?Set POLICY_1_MODEL_PATH to a model directory}"
POLICY_2_MODEL_PATH="${POLICY_2_MODEL_PATH:?Set POLICY_2_MODEL_PATH to a model directory}"
POLICY_3_MODEL_PATH="${POLICY_3_MODEL_PATH:?Set POLICY_3_MODEL_PATH to a model directory}"

TRAIN_DATA="${TRAIN_DATA:?Set TRAIN_DATA to a training parquet file}"
VAL_DATA="${VAL_DATA:?Set VAL_DATA to a validation parquet file}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-30}"
# Keep TRAIN_BATCH_SIZE = PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-1}"
if (( PARAMETER_SYNC_STEP <= 0 )); then
    echo "ERROR: PARAMETER_SYNC_STEP must be positive" >&2
    exit 1
fi
if (( TRAIN_BATCH_SIZE % PARAMETER_SYNC_STEP != 0 )); then
    echo "ERROR: TRAIN_BATCH_SIZE must be divisible by PARAMETER_SYNC_STEP" >&2
    exit 1
fi
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$((TRAIN_BATCH_SIZE / PARAMETER_SYNC_STEP))}"
ROLLOUT_N="${ROLLOUT_N:-8}"
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-2}"
PROMPT_LENGTH="${PROMPT_LENGTH:-16384}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((PROMPT_LENGTH + RESPONSE_LENGTH + 512))}"

if (( TRAIN_BATCH_SIZE != PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE )); then
    echo "ERROR: TRAIN_BATCH_SIZE must equal PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE" >&2
    exit 1
fi

POLICY_1_NNODES="${POLICY_1_NNODES:-1}"
POLICY_1_N_GPUS_PER_NODE="${POLICY_1_N_GPUS_PER_NODE:-4}"
POLICY_1_FSDP_SIZE="${POLICY_1_FSDP_SIZE:-4}"
POLICY_1_ROLLOUT_NNODES="${POLICY_1_ROLLOUT_NNODES:-2}"
POLICY_1_ROLLOUT_N_GPUS_PER_NODE="${POLICY_1_ROLLOUT_N_GPUS_PER_NODE:-2}"
POLICY_1_TENSOR_PARALLEL_SIZE="${POLICY_1_TENSOR_PARALLEL_SIZE:-2}"

POLICY_2_NNODES="${POLICY_2_NNODES:-1}"
POLICY_2_N_GPUS_PER_NODE="${POLICY_2_N_GPUS_PER_NODE:-8}"
POLICY_2_FSDP_SIZE="${POLICY_2_FSDP_SIZE:-8}"
POLICY_2_ROLLOUT_NNODES="${POLICY_2_ROLLOUT_NNODES:-2}"
POLICY_2_ROLLOUT_N_GPUS_PER_NODE="${POLICY_2_ROLLOUT_N_GPUS_PER_NODE:-4}"
POLICY_2_TENSOR_PARALLEL_SIZE="${POLICY_2_TENSOR_PARALLEL_SIZE:-4}"

POLICY_3_NNODES="${POLICY_3_NNODES:-1}"
POLICY_3_N_GPUS_PER_NODE="${POLICY_3_N_GPUS_PER_NODE:-4}"
POLICY_3_FSDP_SIZE="${POLICY_3_FSDP_SIZE:-4}"
POLICY_3_ROLLOUT_NNODES="${POLICY_3_ROLLOUT_NNODES:-2}"
POLICY_3_ROLLOUT_N_GPUS_PER_NODE="${POLICY_3_ROLLOUT_N_GPUS_PER_NODE:-2}"
POLICY_3_TENSOR_PARALLEL_SIZE="${POLICY_3_TENSOR_PARALLEL_SIZE:-2}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/examples/multi_policy/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/external_mas_${TIMESTAMP}.log}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/multi_policy/external_mas_${TIMESTAMP}}"

PYTHON="${PYTHON:-python}"

mkdir -p "${LOG_DIR}" "$(dirname "${LOG_PATH}")"
cd "${REPO_ROOT}"

export PYTHON
export RAY_ADDRESS
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

for var in POLICY_1_MODEL_PATH POLICY_2_MODEL_PATH POLICY_3_MODEL_PATH; do
    val="${!var}"
    if [[ -z "${val}" ]]; then
        echo "ERROR: ${var} is empty" >&2
        exit 1
    fi
    if [[ "${val}" == *'~'* ]]; then
        echo "ERROR: ${var} contains an unexpanded '~': '${val}'" >&2
        exit 1
    fi
    if [[ ! -d "${val}" ]]; then
        echo "ERROR: ${var} directory does not exist: '${val}'" >&2
        exit 1
    fi
done

for data_file in "${TRAIN_DATA}" "${VAL_DATA}"; do
    if [[ ! -f "${data_file}" ]]; then
        echo "ERROR: verl dataset not found: '${data_file}'" >&2
        exit 1
    fi
done

echo "=== Multi-policy external MAS training (separate_async) ==="
echo "Ray address:      ${RAY_ADDRESS}"
echo "Nodes:            p1 train=${POLICY_1_NNODES}/rollout=${POLICY_1_ROLLOUT_NNODES}, p2 train=${POLICY_2_NNODES}/rollout=${POLICY_2_ROLLOUT_NNODES}, p3 train=${POLICY_3_NNODES}/rollout=${POLICY_3_ROLLOUT_NNODES}"
echo "Train GPUs/node:  p1=${POLICY_1_N_GPUS_PER_NODE}, p2=${POLICY_2_N_GPUS_PER_NODE}, p3=${POLICY_3_N_GPUS_PER_NODE} (FSDP: p1=${POLICY_1_FSDP_SIZE}, p2=${POLICY_2_FSDP_SIZE}, p3=${POLICY_3_FSDP_SIZE}, TP: p1=${POLICY_1_TENSOR_PARALLEL_SIZE}, p2=${POLICY_2_TENSOR_PARALLEL_SIZE}, p3=${POLICY_3_TENSOR_PARALLEL_SIZE})"
echo "Rollout GPUs/node: p1=${POLICY_1_ROLLOUT_N_GPUS_PER_NODE}, p2=${POLICY_2_ROLLOUT_N_GPUS_PER_NODE}, p3=${POLICY_3_ROLLOUT_N_GPUS_PER_NODE} (standalone)"
echo "Steps:            ${TOTAL_TRAINING_STEPS}, batch=${TRAIN_BATCH_SIZE}, rollout_n=${ROLLOUT_N}, warmup=${NUM_WARMUP_BATCHES}"
echo "Sequence:         prompt=${PROMPT_LENGTH}, response=${RESPONSE_LENGTH}, max_model_len=${MAX_MODEL_LEN}"
echo "Policy 1 model:   ${POLICY_1_MODEL_PATH}"
echo "Policy 2 model:   ${POLICY_2_MODEL_PATH}"
echo "Policy 3 model:   ${POLICY_3_MODEL_PATH}"
echo "Train data:       ${TRAIN_DATA}"
echo "Val data:         ${VAL_DATA}"
echo "Log file:         ${LOG_PATH}"
echo "============================================"

"${PYTHON}" -u -m examples.multi_policy.main \
    --config-name=multi_policy \
    --config-path="${REPO_ROOT}/examples/multi_policy/config" \
    \
    data.train_files="['${TRAIN_DATA}']" \
    data.val_files="['${VAL_DATA}']" \
    "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
    "data.val_batch_size=${TRAIN_BATCH_SIZE}" \
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}" \
    "trainer.default_local_dir=${CKPT_DIR}" \
    "trainer.v1.trainer_mode=separate_async" \
    "trainer.v1.separate_async.parameter_sync_step=${PARAMETER_SYNC_STEP}" \
    "trainer.v1.separate_async.num_warmup_batches=${NUM_WARMUP_BATCHES}" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" \
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}" \
    \
    "policies.policy_1.ppo_trainer_overrides.trainer.nnodes=${POLICY_1_NNODES}" \
    "policies.policy_1.ppo_trainer_overrides.trainer.n_gpus_per_node=${POLICY_1_N_GPUS_PER_NODE}" \
    "policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.model.path=${POLICY_1_MODEL_PATH}" \
    "policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.actor.fsdp_config.fsdp_size=${POLICY_1_FSDP_SIZE}" \
    "policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH}" \
    "policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH}" \
    "policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}" \
    "policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.nnodes=${POLICY_1_ROLLOUT_NNODES}" \
    "policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.n_gpus_per_node=${POLICY_1_ROLLOUT_N_GPUS_PER_NODE}" \
    "policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.rollout.tensor_model_parallel_size=${POLICY_1_TENSOR_PARALLEL_SIZE}" \
    \
    "policies.policy_2.ppo_trainer_overrides.trainer.nnodes=${POLICY_2_NNODES}" \
    "policies.policy_2.ppo_trainer_overrides.trainer.n_gpus_per_node=${POLICY_2_N_GPUS_PER_NODE}" \
    "policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.model.path=${POLICY_2_MODEL_PATH}" \
    "policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.actor.fsdp_config.fsdp_size=${POLICY_2_FSDP_SIZE}" \
    "policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH}" \
    "policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH}" \
    "policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}" \
    "policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.nnodes=${POLICY_2_ROLLOUT_NNODES}" \
    "policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.n_gpus_per_node=${POLICY_2_ROLLOUT_N_GPUS_PER_NODE}" \
    "policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.rollout.tensor_model_parallel_size=${POLICY_2_TENSOR_PARALLEL_SIZE}" \
    \
    "policies.policy_3.ppo_trainer_overrides.trainer.nnodes=${POLICY_3_NNODES}" \
    "policies.policy_3.ppo_trainer_overrides.trainer.n_gpus_per_node=${POLICY_3_N_GPUS_PER_NODE}" \
    "policies.policy_3.ppo_trainer_overrides.actor_rollout_ref.model.path=${POLICY_3_MODEL_PATH}" \
    "policies.policy_3.ppo_trainer_overrides.actor_rollout_ref.actor.fsdp_config.fsdp_size=${POLICY_3_FSDP_SIZE}" \
    "policies.policy_3.ppo_trainer_overrides.actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH}" \
    "policies.policy_3.ppo_trainer_overrides.actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH}" \
    "policies.policy_3.ppo_trainer_overrides.actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}" \
    "policies.policy_3.ppo_trainer_overrides.actor_rollout_ref.rollout.nnodes=${POLICY_3_ROLLOUT_NNODES}" \
    "policies.policy_3.ppo_trainer_overrides.actor_rollout_ref.rollout.n_gpus_per_node=${POLICY_3_ROLLOUT_N_GPUS_PER_NODE}" \
    "policies.policy_3.ppo_trainer_overrides.actor_rollout_ref.rollout.tensor_model_parallel_size=${POLICY_3_TENSOR_PARALLEL_SIZE}" \
    "$@" \
    2>&1 | tee "${LOG_PATH}"
