#!/usr/bin/env bash
# Minimal single-node rollout -> reward -> FSDP2 policy-update smoke test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
NUM_GPUS="${NUM_GPUS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((NUM_GPUS * 2))}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_N="${ROLLOUT_N:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-rloo}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-320}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-192}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-1024}"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/pipeline_smoke}"
DEFAULT_TRAIN_FILE="${DATA_DIR}/train.parquet"
DEFAULT_VAL_FILE="${DATA_DIR}/val.parquet"
TRAIN_FILE="${TRAIN_FILE:-${DEFAULT_TRAIN_FILE}}"
VAL_FILE="${VAL_FILE:-${DEFAULT_VAL_FILE}}"
TASK_CONFIG="${TASK_CONFIG:-examples/quickstart/training/task_config_pipeline_smoke.yaml}"
PROJECT_NAME="${PROJECT_NAME:-uni-agent-pipeline-smoke}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-0.6b-${NUM_GPUS}gpu}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-${REPO_ROOT}/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
# Each invocation gets its own evidence directory, including custom log roots.
mkdir -p "${AGENT_LOG_DIR}"
AGENT_LOG_DIR="$(mktemp -d "${AGENT_LOG_DIR}/run-XXXXXXXX")"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-${AGENT_LOG_DIR}/train.log}"
VERIFY_TRAINING_SIGNALS="${VERIFY_TRAINING_SIGNALS:-1}"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

"${PYTHON_BIN}" -c 'import ray, torch, transfer_queue, uni_agent, verl, vllm'
AVAILABLE_GPUS="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if ((NUM_GPUS < 1 || NUM_GPUS > AVAILABLE_GPUS)); then
    echo "NUM_GPUS=${NUM_GPUS}, but this Python environment sees ${AVAILABLE_GPUS} GPU(s)." >&2
    exit 1
fi
if ((TRAIN_BATCH_SIZE < NUM_GPUS || TRAIN_BATCH_SIZE % NUM_GPUS != 0)); then
    echo "TRAIN_BATCH_SIZE must be at least NUM_GPUS and divisible by NUM_GPUS." >&2
    exit 1
fi
if ((ROLLOUT_TP < 1 || NUM_GPUS % ROLLOUT_TP != 0)); then
    echo "ROLLOUT_TP must be positive and divide NUM_GPUS." >&2
    exit 1
fi
if [[ "${VERIFY_TRAINING_SIGNALS}" != "0" && "${VERIFY_TRAINING_SIGNALS}" != "1" ]]; then
    echo "VERIFY_TRAINING_SIGNALS must be 0 or 1." >&2
    exit 1
fi

if [[ "${TRAIN_FILE}" == "${DEFAULT_TRAIN_FILE}" && "${VAL_FILE}" == "${DEFAULT_VAL_FILE}" ]]; then
    "${PYTHON_BIN}" -m uni_agent.tasks.pipeline_smoke.preprocess \
        --output-dir "${DATA_DIR}" \
        --train-size 16 \
        --val-size 4
elif [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
    echo "Custom TRAIN_FILE and VAL_FILE must both exist." >&2
    exit 1
fi
mkdir -p "${AGENT_LOG_DIR}"
mkdir -p "$(dirname "${TRAIN_LOG_FILE}")"

echo "Model: ${MODEL_PATH}"
echo "GPUs: ${NUM_GPUS}; rollout TP: ${ROLLOUT_TP}"
echo "Training steps: ${TOTAL_TRAINING_STEPS}; train batch: ${TRAIN_BATCH_SIZE}; rollout n: ${ROLLOUT_N}"
echo "Advantage estimator: ${ADV_ESTIMATOR}"
echo "Task tools: finish only (no host command execution)"

set +e
"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches=1 \
    transfer_queue.enable=True \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    ++data.apply_chat_template_kwargs.enable_thinking=False \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    ++actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.num_workers="$((NUM_GPUS * 2))" \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count="${NUM_GPUS}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${AGENT_LOG_DIR}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="$((NUM_GPUS * 4))" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=all \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="$(basename "${MODEL_PATH}")" \
    ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False \
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=True \
    reward.reward_manager.name=dapo \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.device=cuda \
    ray_kwargs.ray_init.runtime_env.py_executable="${PYTHON_BIN}" \
    "$@" 2>&1 | tee "${TRAIN_LOG_FILE}"
TRAIN_PIPE_STATUS=("${PIPESTATUS[@]}")
set -e

if ((TRAIN_PIPE_STATUS[0] != 0)); then
    exit "${TRAIN_PIPE_STATUS[0]}"
fi
if ((TRAIN_PIPE_STATUS[1] != 0)); then
    echo "Failed to write training log to ${TRAIN_LOG_FILE}." >&2
    exit "${TRAIN_PIPE_STATUS[1]}"
fi
if [[ "${VERIFY_TRAINING_SIGNALS}" == "1" ]]; then
    "${PYTHON_BIN}" -m uni_agent.tasks.pipeline_smoke.verify_training \
        --log-file "${TRAIN_LOG_FILE}" \
        --agent-log-dir "${AGENT_LOG_DIR}" \
        --expected-step "${TOTAL_TRAINING_STEPS}"
fi
