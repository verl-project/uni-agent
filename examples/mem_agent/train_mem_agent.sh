#!/usr/bin/env bash
# MemAgent FSDP2 training with the verl v1 separate-async trainer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"
: "${MODEL_PATH:=Qwen3-4B}"
TRAIN_FILE="./hotpotqa/hotpotqa_train_32k.parquet"
: "${VAL_FILE:=./hotpotqa/hotpotqa_dev.parquet}"
: "${CONDA_ENV_DIR:=/root/.miniconda3/envs/xxx}"
: "${PYTHON_BIN:=${CONDA_ENV_DIR}/bin/python3}"
: "${RAY_BIN:=${CONDA_ENV_DIR}/bin/ray}"
: "${GPU_IDS:=0,1,2,3,4,5,6,7}"
: "${RAY_PREFLIGHT_TIMEOUT:=30}"

for required_path in "${MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done
for executable in "${PYTHON_BIN}" "${RAY_BIN}"; do
    if [[ ! -x "${executable}" ]]; then
        echo "Required executable is missing: ${executable}" >&2
        exit 1
    fi
done


TASK_CONFIG="${TASK_CONFIG:-examples/mem_agent/task_config.yaml}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"

PROJECT_NAME="${PROJECT_NAME:-mem_agent}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mem_agent_v1_$(date +%Y%m%d_%H%M)}"
CKPTS_DIR="${CKPTS_DIR:-${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-${REPO_ROOT}/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"

# separate_async uses disjoint trainer and rollout resource pools.
TRAINER_NNODES="${TRAINER_NNODES:-1}"
TRAINER_GPUS_PER_NODE="${TRAINER_GPUS_PER_NODE:-4}"
ROLLOUT_NNODES="${ROLLOUT_NNODES:-1}"
ROLLOUT_GPUS_PER_NODE="${ROLLOUT_GPUS_PER_NODE:-4}"
ROLLOUT_TP="${ROLLOUT_TP:-4}"

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
GPU_COUNT="${#GPU_ID_ARRAY[@]}"
REQUESTED_GPUS=$((TRAINER_NNODES * TRAINER_GPUS_PER_NODE + ROLLOUT_NNODES * ROLLOUT_GPUS_PER_NODE))
if ((REQUESTED_GPUS != GPU_COUNT)); then
    echo "Trainer + rollout request ${REQUESTED_GPUS} GPUs, but GPU_IDS=${GPU_IDS} contains ${GPU_COUNT}" >&2
    exit 1
fi

PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-4}"
PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-2}"
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE))}"

if ((TRAIN_BATCH_SIZE != PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE)); then
    echo "TRAIN_BATCH_SIZE must equal PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE for separate_async" >&2
    exit 1
fi

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
CONTEXT_CHUNK_SIZE="${CONTEXT_CHUNK_SIZE:-5000}"

GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
CONCURRENCY="${CONCURRENCY:-32}"
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-8}"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

if ! "${RAY_BIN}" status >/dev/null 2>&1; then
    echo "Starting a local Ray cluster on physical GPUs ${GPU_IDS}..."
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${RAY_BIN}" start --head --num-gpus="${GPU_COUNT}"
fi

GPU_IDS="${GPU_IDS}" GPU_COUNT="${GPU_COUNT}" RAY_PREFLIGHT_TIMEOUT="${RAY_PREFLIGHT_TIMEOUT}" \
    "${PYTHON_BIN}" - <<'PY'
import os

import ray

expected = set(os.environ["GPU_IDS"].split(","))
ray.init(address="auto", logging_level="ERROR")


@ray.remote(num_gpus=int(os.environ["GPU_COUNT"]))
def visible_gpu_ids() -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES", "")


gpu_probe = visible_gpu_ids.remote()
try:
    visible = ray.get(gpu_probe, timeout=float(os.environ["RAY_PREFLIGHT_TIMEOUT"]))
except ray.exceptions.GetTimeoutError:
    ray.cancel(gpu_probe, force=True)
    raise SystemExit("Timed out waiting for all requested GPUs; another Ray job may still be using them.") from None
actual = set(visible.split(","))
ray.shutdown()
if actual != expected:
    raise SystemExit(
        f"Ray cluster exposes physical GPUs {sorted(actual)}, expected {sorted(expected)}. "
        "Stop the existing Ray cluster or set GPU_IDS to match it."
    )
print(f"Ray GPU preflight passed: {','.join(sorted(actual, key=int))}")
PY

"${RAY_BIN}" job submit --no-wait \
    --working-dir="${REPO_ROOT}" \
    --runtime-env-json="{\"env_vars\": {\"NCCL_DEBUG\": \"INFO\", \"NCCL_P2P_DISABLE\": \"1\", \"NCCL_IB_DISABLE\": \"1\", \"RAY_DEDUP_LOGS\": \"0\"}}" \
    -- "${PYTHON_BIN}" -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches="${NUM_WARMUP_BATCHES}" \
    trainer.v1.separate_async.parameter_sync_step="${PARAMETER_SYNC_STEP}" \
    transfer_queue.enable=True \
    data.train_files="['${TRAIN_FILE}']" \
    data.val_files="['${VAL_FILE}']" \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    ++data.apply_chat_template_kwargs.enable_thinking=False \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.custom_cls.path=pkg://examples.mem_agent.dataset \
    data.custom_cls.name=HotpotQAMemAgentDataset \
    ++data.context_chunk_size="${CONTEXT_CHUNK_SIZE}" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.nnodes="${ROLLOUT_NNODES}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${ROLLOUT_GPUS_PER_NODE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.7 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.num_workers="${NUM_AGENT_WORKERS}" \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count="${GATEWAY_COUNT}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${AGENT_LOG_DIR}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="${CONCURRENCY}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=all \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="${SERVED_MODEL_NAME}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False \
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=False \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="['console','tensorboard']" \
    trainer.nnodes="${TRAINER_NNODES}" \
    trainer.n_gpus_per_node="${TRAINER_GPUS_PER_NODE}" \
    trainer.val_before_train=False \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${CKPTS_DIR}" \
    "$@"
