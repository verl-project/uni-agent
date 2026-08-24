#!/usr/bin/env bash
# Megatron + V1 async training for the blackbox mini-swe-agent recipe.
#
# mini-swe-agent runs *inside* the sandbox from a prebuilt tool image (mounted
# at /opt/mini-swe-agent) and talks to the policy gateway through a reverse
# tunnel. This recipe uses the new unified runner bridge:
#
#     uni_agent.framework.task_runner.run_task
#
# which resolves each sample's task from task_config_mini_swe_agent.yaml
# (agent + sandbox defaults), deep-merges the sample values and the runtime
# model binding, and returns the reward via report_reward=True.
#
# Usage:
#   bash examples/mini_swe_agent/run_train.sh
#
# All configurable via environment variables (see defaults below).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

# ── Model & data ─────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-${HOME}/models/Qwen3.5-9B}"
TRAIN_DATA="${TRAIN_DATA:-${HOME}/data/swe_agent/swe_rebench_filtered.parquet}"
VAL_DATA="${VAL_DATA:-${HOME}/data/swe_agent/swe_bench_verified.parquet}"
RUNTIME_ENV="${RUNTIME_ENV:-}"

# ── V1 trainer ───────────────────────────────────────────────────────────
TRAINER_MODE="${TRAINER_MODE:-separate_async}"
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-4}"
RAY_SUBMIT_MODE="${RAY_SUBMIT_MODE:-job}"
RAY_INIT_ADDRESS="${RAY_INIT_ADDRESS:-auto}"
RAY_STATUS_TIMEOUT="${RAY_STATUS_TIMEOUT:-5}"

# ── Hardware ─────────────────────────────────────────────────────────────
NNODES="${NNODES:-${NNODES_TRAIN:-4}}"
PHYSICAL_GPUS_PER_NODE="${PHYSICAL_GPUS_PER_NODE:-8}"
if [[ "${TRAINER_MODE}" == "separate_async" ]]; then
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${TRAIN_NGPUS_PER_NODE:-8}}"
    ROLLOUT_NNODES="${ROLLOUT_NNODES:-${NNODES_ROLLOUT:-${NNODES}}}"
    ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE:-${NGPUS_PER_NODE_ROLLOUT:-8}}"
else
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${TRAIN_NGPUS_PER_NODE:-${PHYSICAL_GPUS_PER_NODE}}}"
    ROLLOUT_NNODES="${ROLLOUT_NNODES:-${NNODES_ROLLOUT:-0}}"
    ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE:-${NGPUS_PER_NODE_ROLLOUT:-${N_GPUS_PER_NODE}}}"
fi

# ── Algorithm ────────────────────────────────────────────────────────────
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"
KL_COEF="${KL_COEF:-0.0}"
USE_KL_LOSS="${USE_KL_LOSS:-False}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-4e-4}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-4e-4}"
CLIP_RATIO_C="${CLIP_RATIO_C:-10.0}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
BY_PASS_MODE="${BY_PASS_MODE:-True}"        # rollout_correction.bypass_mode
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
LOSS_MODE="${LOSS_MODE:-gspo}"

# ── Sequence lengths ─────────────────────────────────────────────────────
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-131072}"
MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))

# ── Rollout parameters ───────────────────────────────────────────────────
ENGINE="${ENGINE:-vllm}"
GEN_TP="${GEN_TP:-${TP:-${ROLLOUT_NGPUS_PER_NODE}}}"
N="${N:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
VAL_TOP_K="${VAL_TOP_K:--1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.7}"
UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB:-2048}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"

# ── Megatron training parallelism ────────────────────────────────────────
if [[ "${TRAINER_MODE}" == "separate_async" ]]; then
    TRAIN_TP="${TRAIN_TP:-${TP:-${N_GPUS_PER_NODE}}}"
else
    TRAIN_TP="${TRAIN_TP:-${TP:-8}}"
fi
TRAIN_PP="${TRAIN_PP:-2}"
TRAIN_CP="${TRAIN_CP:-4}"
OFFLOAD="${OFFLOAD:-True}"
OPTIMIZER_OFFLOAD_FRACTION="${OFFLOAD_FRACTION:-1.0}"
USE_MBRIDGE="${USE_MBRIDGE:-True}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
# Per-GPU micro batch size.
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"

# ── Agent-framework rollout (unified run_task bridge) ────────────────────
# mini-swe-agent knobs (step_limit/run_timeout/conda_env) and the tool-image
# mount are configured in TASK_CONFIG (task_config_mini_swe_agent.yaml).
TASK_CONFIG="${TASK_CONFIG:-examples/mini_swe_agent/task_config_mini_swe_agent.yaml}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"   # gateway tool-call parser; must match the model chat template
GATEWAY_COUNT="${GATEWAY_COUNT:-8}"
MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-256}"
# Hard cap per-session runtime (seconds). A runner that hangs without raising
# (e.g. remote sandbox OOM-killed without surfacing an error) otherwise holds its
# concurrency slot forever and stalls the whole training batch.
SESSION_TIMEOUT_SECONDS="${SESSION_TIMEOUT_SECONDS:-1800}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
# The agent reports finished explicitly (exit_status == "Submitted"); set True to
# exclude unfinished episodes from the loss (paired with the finished field in agent.py).
MASK_UNFINISHED_EPISODE="${MASK_UNFINISHED_EPISODE:-True}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-/home/${USER}/uni_agent_logs}"
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-8}"

RUNNER_ARGS=(
    "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter"
    "+actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT}"
    "+actor_rollout_ref.rollout.custom.agent_framework.log_dir=${AGENT_LOG_DIR}"
    "+actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task"
    "+actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task"
    "+actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${MAX_CONCURRENT_SESSIONS}"
    "+actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.session_timeout_seconds=${SESSION_TIMEOUT_SECONDS}"
    "+actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG}"
    "+actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${SERVED_MODEL_NAME}"
    "+actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True"
    "+actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=${MASK_UNFINISHED_EPISODE}"
    "+actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False"
)

# ── OpenYuanrong (remote sandbox) ───────────────────────────────────────
# OPENYUANRONG_SERVER_ADDRESS / OPENYUANRONG_TOKEN are required by the provider.
# Canonical SWE image refs are mapped to the sandbox registry via `image_map`
# in the Task Config (task_config_mini_swe_agent.yaml), not here.
OPENYUANRONG_SERVER_ADDRESS="${OPENYUANRONG_SERVER_ADDRESS:-}"
OPENYUANRONG_TOKEN="${OPENYUANRONG_TOKEN:-}"
OPENYUANRONG_TUNNEL_SSL_VERIFY="${OPENYUANRONG_TUNNEL_SSL_VERIFY:-0}"
SANDBOX_NAME_PREFIX="${SANDBOX_NAME_PREFIX:-mini-swe-}"

# ── Logging & checkpointing ──────────────────────────────────────────────
PROJECT_NAME="${PROJECT_NAME:-swe_agent_blackbox}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-swe_agent_$(date +%Y%m%d_%H%M)}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
CKPTS_DIR="${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-${MAX_SAMPLES:--1}}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-${MAX_SAMPLES:--1}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-500}"
# rl-insight collector endpoint. Leave empty to disable rl_insight (the logger
# list below is guarded on this being set).
RL_INSIGHT_SERVER_URL="${RL_INSIGHT_SERVER_URL:-}"

export OPENYUANRONG_SERVER_ADDRESS
export OPENYUANRONG_TOKEN
export OPENYUANRONG_TUNNEL_SSL_VERIFY
export SANDBOX_NAME_PREFIX
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export RL_INSIGHT_SERVER_URL
# Logger list: console always; rl_insight only when its endpoint is configured,
# so an empty RL_INSIGHT_SERVER_URL does not enable a logger that cannot connect.
LOGGER='["console"]'
if [[ -n "${RL_INSIGHT_SERVER_URL}" ]]; then
    LOGGER='["console","rl_insight"]'
fi
# NCCL tuning for the multi-node NPU cluster (previously set via the job
# runtime-env-json, now forwarded through verl's ray.init runtime_env below).
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

echo "=== SWE-Agent Blackbox Megatron Async Training ==="
echo "Model:       ${MODEL_PATH}"
echo "Train data:  ${TRAIN_DATA}"
echo "Val data:    ${VAL_DATA}"
echo "Engine:      ${ENGINE} (gen_tp=${GEN_TP}, train_tp=${TRAIN_TP})"
echo "Task config: ${TASK_CONFIG}"
echo "Tool parser: ${TOOL_PARSER}"
echo "Mask:        mask_unfinished_episode=${MASK_UNFINISHED_EPISODE}"
echo "Batch:       n=${N}, mini_bsz=${PPO_MINI_BATCH_SIZE}"
echo "Sequence:    prompt=${PROMPT_LENGTH}, response=${RESPONSE_LENGTH}"
echo "Trainer:     V1 ${TRAINER_MODE}"
if [[ "${TRAINER_MODE}" == "separate_async" ]]; then
    echo "Resources:   trainer=${NNODES}x${N_GPUS_PER_NODE}, rollout=${ROLLOUT_NNODES}x${ROLLOUT_NGPUS_PER_NODE}"
else
    echo "Resources:   colocated=${NNODES}x${N_GPUS_PER_NODE}"
fi
echo "Samples:     train_max=${TRAIN_MAX_SAMPLES}, val_max=${VAL_MAX_SAMPLES}"
echo "==================================================="

# ── Compute derived parameters ───────────────────────────────────────────
ACTOR_PPO_MAX_TOKEN_LEN=$(( (PROMPT_LENGTH + RESPONSE_LENGTH) / TRAIN_CP ))
INFER_PPO_MAX_TOKEN_LEN=$(( (PROMPT_LENGTH + RESPONSE_LENGTH) / TRAIN_CP ))

# Job-level runtime env is NOT set here: it would conflict with verl main_ppo's
# own ray.init runtime_env (both set TRANSFER_QUEUE_ENABLE / PYTHONPATH, and Ray
# refuses to merge duplicated keys). All env vars ride the verl ray.init
# runtime_env via config `ray_kwargs.ray_init.runtime_env.env_vars.*` (injected
# below in MAIN_CMD). A custom YAML runtime env is still honored.
RUNTIME_ENV_ARGS=()
if [ -n "${RUNTIME_ENV}" ]; then
    RUNTIME_ENV_ARGS=(--runtime-env "${RUNTIME_ENV}")
fi

# Env vars forwarded to every Ray actor through verl's ray.init runtime_env.
# Only the (fixed) keys below go here, and all values are quoted strings (so
# hydra keeps them as str, not int -- Ray requires Dict[str,str]). These keys
# are declared statically:
#   TRANSFER_QUEUE_ENABLE / NCCL_P2P_DISABLE / NCCL_SHM_DISABLE /
#   SANDBOX_NAME_PREFIX / RL_INSIGHT_SERVER_URL / OPENYUANRONG_*
#
# The OPENYUANRONG_* credentials MUST ride the runtime_env: `ray job submit`
# launches the driver via the cluster-side Job Agent, which does NOT inherit
# the submitting shell's environment, so a plain `export` would never reach
# the rollout workers. Empty values are skipped (the provider has defaults).
# Pass secrets via a Ray secret provider in production when available.
#
# PYTHONPATH is omitted here: Ray injects it from the job working_dir; the actor
# PYTHONPATH is set by verl's get_ppo_ray_runtime_env.
RAY_INIT_ENV_ARGS=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE=\"1\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SHM_DISABLE=\"1\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_NAME_PREFIX=\"${SANDBOX_NAME_PREFIX}\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.RL_INSIGHT_SERVER_URL=\"${RL_INSIGHT_SERVER_URL}\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENYUANRONG_SERVER_ADDRESS=\"${OPENYUANRONG_SERVER_ADDRESS}\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENYUANRONG_TOKEN=\"${OPENYUANRONG_TOKEN}\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.OPENYUANRONG_TUNNEL_SSL_VERIFY=\"${OPENYUANRONG_TUNNEL_SSL_VERIFY}\""
)
# TRANSFER_QUEUE_ENABLE is a REQUIRED key here: verl main_ppo overwrites it to
# "1" itself when transfer_queue.enable=True. It must already exist in the
# (struct) env_vars dict or verl's assignment crashes ("Key ... is not in
# struct").
RAY_INIT_ENV_ARGS+=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.TRANSFER_QUEUE_ENABLE=\"\""
)

# ── Ensure Ray is running ────────────────────────────────────────────────
if [[ "${TRAINER_MODE}" == "separate_async" ]]; then
    TOTAL_GPUS=$(( NNODES * N_GPUS_PER_NODE + ROLLOUT_NNODES * ROLLOUT_NGPUS_PER_NODE ))
else
    TOTAL_GPUS=$(( NNODES * N_GPUS_PER_NODE ))
fi
if ! timeout "${RAY_STATUS_TIMEOUT}" ray status &>/dev/null; then
    echo "Starting Ray cluster (${TOTAL_GPUS} GPUs)..."
    # NPU env head resources; for GPU env switch to --num-gpus="${TOTAL_GPUS}"
    ray start --head --resources="{\"NPU\": ${TOTAL_GPUS}}" --disable-usage-stats
else
    echo "Ray cluster already running."
fi

# ── Launch ────────────────────────────────────────────────────────────────
WORKING_DIR="${WORKING_DIR:-$(pwd)}"

MAIN_CMD=(
    python3 -m verl.trainer.main_ppo
    --config-name=ppo_megatron_trainer
    hydra.searchpath=[pkg://verl.trainer.config]
    +ray_kwargs.ray_init.address="${RAY_INIT_ADDRESS}"
    "${RAY_INIT_ENV_ARGS[@]}"
    trainer.use_v1=True
    trainer.v1.trainer_mode="${TRAINER_MODE}"
    trainer.v1.separate_async.num_warmup_batches=${NUM_WARMUP_BATCHES}
    trainer.v1.separate_async.parameter_sync_step=${PARAMETER_SYNC_STEP}
    transfer_queue.enable=True
    transfer_queue.metrics.enabled=True
    actor_rollout_ref.nccl_timeout=9600
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=False
    data.train_files="['${TRAIN_DATA}']"
    data.val_files="['${VAL_DATA}']"
    data.prompt_key=prompt
    data.truncation=left
    data.return_raw_chat=True
    data.filter_overlong_prompts=True
    data.trust_remote_code=True
    data.dataloader_num_workers=0
    data.max_prompt_length=${PROMPT_LENGTH}
    data.max_response_length=${RESPONSE_LENGTH}
    data.train_max_samples=${TRAIN_MAX_SAMPLES}
    data.val_max_samples=${VAL_MAX_SAMPLES}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    actor_rollout_ref.rollout.n=${N}
    actor_rollout_ref.rollout.name=${ENGINE}
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    +actor_rollout_ref.rollout.enable_sleep_mode=True
    actor_rollout_ref.rollout.calculate_log_probs=True
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
    actor_rollout_ref.rollout.nnodes=${ROLLOUT_NNODES}
    actor_rollout_ref.rollout.n_gpus_per_node=${ROLLOUT_NGPUS_PER_NODE}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.disable_log_stats=False
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=\"FULL_DECODE_ONLY\""
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode=align"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=true"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=true"
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=100
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
    actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER}
    actor_rollout_ref.rollout.agent.num_workers=${NUM_AGENT_WORKERS}
    "${RUNNER_ARGS[@]}"
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.checkpoint.strict=False
    +actor_rollout_ref.actor.use_rollout_log_probs=True
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
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=False
    actor_rollout_ref.actor.megatron.param_offload=${OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${OFFLOAD}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${TRAIN_CP}
    actor_rollout_ref.actor.megatron.use_mbridge=${USE_MBRIDGE}
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.megatron.param_offload=${OFFLOAD}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${TRAIN_CP}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN}
    algorithm.adv_estimator=${ADV_ESTIMATOR}
    algorithm.use_kl_in_reward=${USE_KL_IN_REWARD}
    algorithm.kl_ctrl.kl_coef=${KL_COEF}
    algorithm.rollout_correction.bypass_mode=${BY_PASS_MODE}
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.logger="${LOGGER}"
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE}
    "$@"
)

if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
    MAIN_CMD+=(trainer.total_training_steps=${TOTAL_TRAINING_STEPS})
fi

if [[ "${RAY_SUBMIT_MODE}" == "job" ]]; then
    ray job submit --no-wait --working-dir="${WORKING_DIR}" "${RUNTIME_ENV_ARGS[@]}" -- "${MAIN_CMD[@]}"
elif [[ "${RAY_SUBMIT_MODE}" == "local" ]]; then
    "${MAIN_CMD[@]}"
else
    echo "Unknown RAY_SUBMIT_MODE=${RAY_SUBMIT_MODE}; expected job or local" >&2
    exit 1
fi
