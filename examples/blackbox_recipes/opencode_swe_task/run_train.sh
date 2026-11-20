#!/usr/bin/env bash
# Megatron + V1 async training for the OpenCode SWE-rebench recipe on an
# E2B-compatible sandbox backend (Ascend NPU defaults).
#
# Each rollout runs uni_agent.framework.task_runner.run_task: the sample's task
# is resolved against TASK_CONFIG (OpenCode 1.19.3 mounted at /opt/opencode,
# E2B sandbox), the agent talks to its Gateway session URL directly, and the
# SWE-rebench verifier returns the reward. There is no reverse tunnel: the
# sandbox network must reach the Ray node IPs and Gateway ports.
#
# Usage (from any directory):
#   E2B_API_KEY=... E2B_DOMAIN=... \
#   MODEL_PATH=/models/Qwen3.6-35B-A3B TRAIN_DATA=/data/swe_rebench.parquet \
#   bash examples/blackbox_recipes/opencode_swe_task/run_train.sh [extra hydra overrides]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${REPO_ROOT}"

# ── Model & data ─────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-${HOME}/models/Qwen3.6-35B-A3B}"
TRAIN_DATA="${TRAIN_DATA:-${HOME}/data/swe_agent/swe_rebench_filtered.parquet}"
VAL_DATA="${VAL_DATA:-${TRAIN_DATA}}"
RUNTIME_ENV="${RUNTIME_ENV:-}"

# ── V1 trainer ───────────────────────────────────────────────────────────
TRAINER_MODE="${TRAINER_MODE:-colocate_async}"
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
RAY_SUBMIT_MODE="${RAY_SUBMIT_MODE:-job}"
RAY_INIT_ADDRESS="${RAY_INIT_ADDRESS:-auto}"
RAY_STATUS_TIMEOUT="${RAY_STATUS_TIMEOUT:-5}"

# ── Hardware ─────────────────────────────────────────────────────────────
NNODES="${NNODES:-2}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-16}"

# ── Algorithm ────────────────────────────────────────────────────────────
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"
KL_COEF="${KL_COEF:-0.0}"
USE_KL_LOSS="${USE_KL_LOSS:-False}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
CLIP_RATIO_C="${CLIP_RATIO_C:-10.0}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
BYPASS_MODE="${BYPASS_MODE:-True}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
LOSS_MODE="${LOSS_MODE:-vanilla}"
# Dynamic-sampling filter; disable for tiny smoke datasets where every group can
# share one reward and the filter would keep asking for more batches.
FILTER_GROUPS="${FILTER_GROUPS:-False}"

# ── Sequence lengths ─────────────────────────────────────────────────────
# Keep PROMPT_LENGTH + RESPONSE_LENGTH >= agent.context_limit in TASK_CONFIG.
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-61440}"
MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))

# ── Rollout ──────────────────────────────────────────────────────────────
ENGINE="${ENGINE:-vllm}"
GEN_TP="${GEN_TP:-4}"
N="${N:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
VAL_TOP_K="${VAL_TOP_K:--1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.7}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB:-2048}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"

# ── Megatron parallelism (MoE) ───────────────────────────────────────────
TRAIN_TP="${TRAIN_TP:-4}"
TRAIN_PP="${TRAIN_PP:-2}"
TRAIN_CP="${TRAIN_CP:-4}"
TRAIN_EP="${TRAIN_EP:-8}"
TRAIN_ETP="${TRAIN_ETP:-1}"
OFFLOAD="${OFFLOAD:-True}"
OPTIMIZER_OFFLOAD_FRACTION="${OFFLOAD_FRACTION:-1.0}"
USE_MBRIDGE="${USE_MBRIDGE:-True}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"

# ── Agent-framework rollout ──────────────────────────────────────────────
TASK_CONFIG="${TASK_CONFIG:-examples/blackbox_recipes/opencode_swe_task/task_config_opencode_e2b.yaml}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
GATEWAY_COUNT="${GATEWAY_COUNT:-8}"
MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-128}"
# agent.run_timeout (3600) + eval_timeout (900) + sandbox start/teardown.
SESSION_TIMEOUT_SECONDS="${SESSION_TIMEOUT_SECONDS:-5400}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
MASK_UNFINISHED_EPISODE="${MASK_UNFINISHED_EPISODE:-True}"
TRAJECTORY_SELECTION="${TRAJECTORY_SELECTION:-longest}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-${HOME}/uni_agent_logs/opencode_swe}"
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-8}"

# ── E2B-compatible sandbox backend ──────────────────────────────────────
# Read by the e2b SDK inside each rollout worker. E2B_API_KEY is required;
# E2B_DOMAIN or E2B_API_URL/E2B_SANDBOX_URL select the internal backend.
E2B_ENV_KEYS=(E2B_API_KEY E2B_DOMAIN E2B_API_URL E2B_SANDBOX_URL E2B_DEBUG E2B_HTTP_VERSION)
if [[ -z "${E2B_API_KEY:-}" ]]; then
    echo "E2B_API_KEY is not set" >&2
    exit 1
fi

# ── Logging & checkpointing ──────────────────────────────────────────────
PROJECT_NAME="${PROJECT_NAME:-opencode_swe_rebench}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opencode_$(date +%Y%m%d_%H%M)}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
CKPTS_DIR="${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"

export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

echo "=== OpenCode SWE-rebench training (E2B) ==="
echo "Model:       ${MODEL_PATH} (served as ${SERVED_MODEL_NAME})"
echo "Train data:  ${TRAIN_DATA}"
echo "Val data:    ${VAL_DATA}"
echo "Task config: ${TASK_CONFIG}"
echo "Resources:   ${NNODES}x${N_GPUS_PER_NODE} ${TRAINER_MODE}, gen_tp=${GEN_TP}"
echo "Megatron:    tp=${TRAIN_TP} pp=${TRAIN_PP} cp=${TRAIN_CP} ep=${TRAIN_EP} etp=${TRAIN_ETP}"
echo "Batch:       bsz=${TRAIN_BATCH_SIZE}, n=${N}, mini_bsz=${PPO_MINI_BATCH_SIZE}"
echo "Sequence:    prompt=${PROMPT_LENGTH}, response=${RESPONSE_LENGTH}"
echo "E2B:         domain=${E2B_DOMAIN:-<default>} api_url=${E2B_API_URL:-<default>}"
echo "============================================"

ACTOR_PPO_MAX_TOKEN_LEN=$((MAX_MODEL_LEN / TRAIN_CP))
INFER_PPO_MAX_TOKEN_LEN=$((MAX_MODEL_LEN / TRAIN_CP))

RUNTIME_ENV_ARGS=()
if [[ -n "${RUNTIME_ENV}" ]]; then
    RUNTIME_ENV_ARGS=(--runtime-env "${RUNTIME_ENV}")
fi

# `ray job submit` starts the driver through the cluster Job Agent, which does not
# inherit this shell, so sandbox credentials ride verl's ray.init runtime_env.
# Values stay quoted so Hydra keeps them as strings. TRANSFER_QUEUE_ENABLE must
# pre-exist in the struct dict because verl assigns it itself.
RAY_INIT_ENV_ARGS=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.TRANSFER_QUEUE_ENABLE=\"\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT=\"2400\""
)
for key in "${E2B_ENV_KEYS[@]}"; do
    if [[ -n "${!key:-}" ]]; then
        RAY_INIT_ENV_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.${key}=\"${!key}\"")
    fi
done
for key in VLLM_USE_V1 TASK_QUEUE_ENABLE CPU_AFFINITY_CONF OMP_PROC_BIND VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE; do
    case "${key}" in
        OMP_PROC_BIND) default=false ;;
        *) default=1 ;;
    esac
    RAY_INIT_ENV_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.${key}=\"${!key:-${default}}\"")
done

RUNNER_PREFIX="+actor_rollout_ref.rollout.custom.agent_framework"
RUNNER_ARGS=(
    "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter"
    "${RUNNER_PREFIX}.gateway_count=${GATEWAY_COUNT}"
    "${RUNNER_PREFIX}.log_dir=${AGENT_LOG_DIR}"
    "${RUNNER_PREFIX}.mask_unfinished_episode=${MASK_UNFINISHED_EPISODE}"
    "${RUNNER_PREFIX}.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task"
    "${RUNNER_PREFIX}.agent_runners.task.dispatch_mode=ray_task"
    "${RUNNER_PREFIX}.agent_runners.task.max_concurrent_sessions=${MAX_CONCURRENT_SESSIONS}"
    "${RUNNER_PREFIX}.agent_runners.task.session_timeout_seconds=${SESSION_TIMEOUT_SECONDS}"
    "${RUNNER_PREFIX}.agent_runners.task.trajectory_selection=${TRAJECTORY_SELECTION}"
    "${RUNNER_PREFIX}.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG}"
    "${RUNNER_PREFIX}.agent_runners.task.runner_kwargs.model_name=${SERVED_MODEL_NAME}"
    "${RUNNER_PREFIX}.agent_runners.task.runner_kwargs.report_reward=True"
)

WORKING_DIR="${WORKING_DIR:-${REPO_ROOT}}"

MAIN_CMD=(
    python3 -m verl.trainer.main_ppo
    --config-name=ppo_megatron_trainer
    hydra.searchpath=[pkg://verl.trainer.config]
    +ray_kwargs.ray_init.address="${RAY_INIT_ADDRESS}"
    "${RAY_INIT_ENV_ARGS[@]}"
    trainer.use_v1=True
    trainer.v1.trainer_mode="${TRAINER_MODE}"
    trainer.v1.${TRAINER_MODE}.num_warmup_batches=${NUM_WARMUP_BATCHES}
    trainer.device=npu
    transfer_queue.enable=True
    transfer_queue.metrics.enabled=True
    actor_rollout_ref.nccl_timeout=9600
    actor_rollout_ref.hybrid_engine=True
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=False
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=${MAX_MODEL_LEN}
    data.train_files="['${TRAIN_DATA}']"
    data.val_files="['${VAL_DATA}']"
    data.prompt_key=prompt
    data.truncation=error
    data.return_raw_chat=True
    data.filter_overlong_prompts=True
    data.trust_remote_code=True
    data.dataloader_num_workers=0
    data.max_prompt_length=${PROMPT_LENGTH}
    data.max_response_length=${RESPONSE_LENGTH}
    data.train_max_samples=${TRAIN_MAX_SAMPLES}
    data.val_max_samples=${VAL_MAX_SAMPLES}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.gen_batch_size=${TRAIN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    actor_rollout_ref.rollout.n=${N}
    actor_rollout_ref.rollout.name=${ENGINE}
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    +actor_rollout_ref.rollout.enable_sleep_mode=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.temperature=${TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${TOP_P}
    actor_rollout_ref.rollout.top_k=${TOP_K}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MB}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=\"FULL_DECODE_ONLY\""
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=true
    +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=false
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
    actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER}
    actor_rollout_ref.rollout.agent.num_workers=${NUM_AGENT_WORKERS}
    "${RUNNER_ARGS[@]}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN}
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.checkpoint.strict=False
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE}
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=False
    actor_rollout_ref.actor.megatron.param_offload=${OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${OFFLOAD}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${TRAIN_EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${TRAIN_ETP}
    actor_rollout_ref.actor.megatron.use_mbridge=${USE_MBRIDGE}
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=False
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN}
    actor_rollout_ref.ref.megatron.param_offload=${OFFLOAD}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${TRAIN_CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${TRAIN_EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${TRAIN_ETP}
    actor_rollout_ref.ref.megatron.use_remove_padding=False
    algorithm.adv_estimator=${ADV_ESTIMATOR}
    algorithm.use_kl_in_reward=${USE_KL_IN_REWARD}
    algorithm.kl_ctrl.kl_coef=${KL_COEF}
    algorithm.rollout_correction.bypass_mode=${BYPASS_MODE}
    algorithm.filter_groups.enable=${FILTER_GROUPS}
    algorithm.filter_groups.metric=acc
    algorithm.filter_groups.max_inflight_gen_batches=1
    reward.reward_manager.name=dapo
    reward.custom_reward_function.path=pkg://uni_agent.framework.task_runner
    reward.custom_reward_function.name=score_from_runner_result
    trainer.logger='["console"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.resume_mode=auto
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE}
)
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
    MAIN_CMD+=(trainer.total_training_steps=${TOTAL_TRAINING_STEPS})
fi
MAIN_CMD+=("$@")

if [[ "${PRINT_CMD_ONLY:-0}" == "1" ]]; then
    # Redact whole raw arguments before shell quoting (spaces/newlines in a key
    # otherwise leave suffixes visible). Covers caller-supplied overrides too.
    for arg in "${MAIN_CMD[@]}"; do
        if [[ "$arg" == *E2B_API_KEY=* ]]; then
            arg="${arg%%E2B_API_KEY=*}E2B_API_KEY=<redacted>"
        fi
        printf '%q ' "$arg"
    done
    echo
    exit 0
fi

if [[ "${RAY_SUBMIT_MODE}" == "job" ]]; then
    if ! timeout "${RAY_STATUS_TIMEOUT}" ray status &>/dev/null; then
        echo "No running Ray cluster found; start Ray on every node first." >&2
        exit 1
    fi
    ray job submit --no-wait --working-dir="${WORKING_DIR}" "${RUNTIME_ENV_ARGS[@]}" -- "${MAIN_CMD[@]}"
elif [[ "${RAY_SUBMIT_MODE}" == "local" ]]; then
    "${MAIN_CMD[@]}"
else
    echo "Unknown RAY_SUBMIT_MODE=${RAY_SUBMIT_MODE}; expected job or local" >&2
    exit 1
fi
