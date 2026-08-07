#!/usr/bin/env bash
# MemAgent FSDP2 training with the verl v1 separate-async trainer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

: "${MODEL_PATH:?Set MODEL_PATH to the policy checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the training Parquet file}"
: "${VAL_FILE:?Set VAL_FILE to the validation Parquet file}"

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

GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
CONCURRENCY="${CONCURRENCY:-32}"
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-8}"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

ray job submit --no-wait \
    --working-dir="${REPO_ROOT}" \
    --runtime-env-json="{\"env_vars\": {\"NCCL_DEBUG\": \"INFO\", \"NCCL_P2P_DISABLE\": \"1\", \"NCCL_IB_DISABLE\": \"1\", \"RAY_DEDUP_LOGS\": \"0\"}}" \
    -- python3 -m verl.trainer.main_ppo \
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
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
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
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True \
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
