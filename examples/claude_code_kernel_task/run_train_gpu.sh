#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

# NVIDIA/Megatron variant of this recipe. It preserves the synchronous trainer
# topology of the legacy GPU launcher while using stock verl and UniAgent APIs.
# Keep NCCL/NIC/CUDA settings in the Ray runtime environment so every worker
# receives the same values.
RECIPE_DIR="examples/claude_code_kernel_task"
NNODES=${NNODES:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${N_GPUS:-8}}

project_name=${PROJECT_NAME:-"Triton-Agent-sync"}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H%M)_exp"}
DATA_HOME=${DATA_HOME:-"${HOME}"}
MODEL_PATH=${MODEL_PATH:-"${DATA_HOME}/models/Qwen3.6-35B-A3B"}
CKPTS_DIR=${CKPTS_DIR:-"${DATA_HOME}/ckpts/${project_name}/${exp_name}"}
mkdir -p "${CKPTS_DIR}"
LOG_DIR=${LOG_DIR:-"${DATA_HOME}/logs/${project_name}"}
mkdir -p "${LOG_DIR}"
AGENT_LOG_DIR="${LOG_DIR}/${exp_name}"
TRAIN_FILE=${TRAIN_FILE:-"${DATA_HOME}/data/triton-agent/train.parquet"}
VAL_FILE=${VAL_FILE:-"${DATA_HOME}/data/triton-agent/validation.parquet"}
RUNTIME_ENV=${RUNTIME_ENV:-}
WORKING_DIR=${WORKING_DIR:-"${REPO_ROOT}"}
TASK_CONFIG=${TASK_CONFIG:-"${RECIPE_DIR}/task_config_kernel_bench.yaml"}
TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}
GATEWAY_COUNT=${GATEWAY_COUNT:-2}
AGENT_WORKERS=${AGENT_WORKERS:-4}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "${MODEL_PATH}")"}

# Session budget includes setup/evaluation/cleanup as well as Claude execution.
CLAUDE_RUN_TIMEOUT=${CLAUDE_RUN_TIMEOUT:-7200}
if [[ ! "${CLAUDE_RUN_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CLAUDE_RUN_TIMEOUT must be positive integer seconds" >&2
  exit 2
fi
SESSION_TIMEOUT_SECONDS=${SESSION_TIMEOUT_SECONDS:-$((CLAUDE_RUN_TIMEOUT + 600))}
if [[ ! "${SESSION_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CLAUDE_RUN_TIMEOUT and SESSION_TIMEOUT_SECONDS must be positive integer seconds" >&2
  exit 2
fi
if (( SESSION_TIMEOUT_SECONDS <= CLAUDE_RUN_TIMEOUT )); then
  echo "SESSION_TIMEOUT_SECONDS must exceed CLAUDE_RUN_TIMEOUT to reserve setup and cleanup time" >&2
  exit 2
fi

# Training/rollout uses NVIDIA GPUs. Operator verification runs in per-session
# containers on the remote Ascend hosts below.
REMOTE_DOCKER_HOSTS=${REMOTE_DOCKER_HOSTS:?set comma-separated Docker endpoints, preferably ssh://user@host}
REMOTE_DOCKER_HOSTS_PARSER="${REMOTE_DOCKER_HOSTS//,/\\\\,}"
EVALUATOR_NPU_DEVICE_IDS=${EVALUATOR_NPU_DEVICE_IDS:?set comma-separated evaluator NPU IDs}
EVALUATOR_NPU_DEVICE_IDS_PARSER="${EVALUATOR_NPU_DEVICE_IDS//,/\\\\,}"
EVALUATOR_NPU_LOCK_DIR=${EVALUATOR_NPU_LOCK_DIR:-/var/lock/triton-agent-npu}
EVALUATOR_NPU_LOCK_TIMEOUT=${EVALUATOR_NPU_LOCK_TIMEOUT:-1200}
IFS=',' read -r -a evaluator_devices <<< "${EVALUATOR_NPU_DEVICE_IDS}"
IFS=',' read -r -a remote_docker_hosts <<< "${REMOTE_DOCKER_HOSTS}"
MAX_CONCURRENT_SESSIONS=${MAX_CONCURRENT_SESSIONS:-$((${#evaluator_devices[@]} * ${#remote_docker_hosts[@]} * 4))}
# Algorithm and sequence lengths.
loss_agg_mode=${LOSS_AGG_MODE:-token-mean}

max_prompt_length=${MAX_PROMPT_LENGTH:-184320}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
max_model_len=${MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}
total_len=$((max_prompt_length + max_response_length))
if (( total_len > max_model_len )); then
  echo "MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH must not exceed MAX_MODEL_LEN" >&2
  exit 2
fi

use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
gen_tp=${GEN_TP:-4}
train_tp=${TP:-2}
train_pp=${PP:-1}
train_cp=${CP:-8}
train_ep=${EP:-8}
train_etp=${ETP:-1}
actor_ppo_max_token_len=${PPO_MAX_TOKEN_LEN_PER_GPU:-28672}
infer_ppo_max_token_len=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${actor_ppo_max_token_len}}
seed=${VERL_DETERMINISTIC_SEED:-1234}
routing_replay_mode=${ROUTING_REPLAY_MODE:-disabled}
enable_rollout_routing_replay=${ENABLE_ROLLOUT_ROUTING_REPLAY:-False}

train_prompt_bsz=${BATCH_SIZE:-16}
val_prompt_bsz=${VAL_BATCH_SIZE:-128}
n_resp_per_prompt=${ROLLOUT_N:-14}
val_resp_per_prompt=${VAL_ROLLOUT_N:-1}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-4}
actor_lr=${ACTOR_LR:-1e-6}
test_freq=${TEST_FREQ:-10}
save_freq=${SAVE_FREQ:-10}
total_epochs=${TOTAL_EPOCHS:-100}
val_before_train=${VAL_BEFORE_TRAIN:-False}
gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL:-0.72}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-60}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}

RUNTIME_ENV_ARGS=()
if [[ -n "${RUNTIME_ENV}" ]]; then
  RUNTIME_ENV_ARGS=(--runtime-env "${RUNTIME_ENV}")
fi

MAIN_CMD=(
  python3 -m verl.trainer.main_ppo
  --config-name=ppo_megatron_trainer
  trainer.use_v1=True
  trainer.v1.trainer_mode=sync
  transfer_queue.enable=True
  data.train_files="${TRAIN_FILE}"
  data.val_files="${VAL_FILE}"
  data.prompt_key=prompt
  data.filter_overlong_prompts=True
  data.truncation=error
  data.max_prompt_length=${max_prompt_length}
  data.max_response_length=${max_response_length}
  data.train_batch_size=${train_prompt_bsz}
  data.val_batch_size=${val_prompt_bsz}
  data.return_raw_chat=True
  actor_rollout_ref.rollout.n=${n_resp_per_prompt}
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  algorithm.kl_ctrl.kl_coef=0.001
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.002
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
  actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.optim.lr=${actor_lr}
  actor_rollout_ref.actor.optim.lr_decay_style=constant
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0
  +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
  actor_rollout_ref.actor.optim.use_precision_aware_optimizer=False
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
  actor_rollout_ref.actor.megatron.use_mbridge=True
  actor_rollout_ref.actor.megatron.seed=${seed}
  actor_rollout_ref.actor.megatron.dtype=bfloat16
  actor_rollout_ref.actor.megatron.router_replay.mode=${routing_replay_mode}
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
  actor_rollout_ref.actor.megatron.use_remove_padding=True
  actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max=False
  actor_rollout_ref.actor.megatron.param_offload=True
  actor_rollout_ref.actor.megatron.optimizer_offload=False
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
  actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.001
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0
  algorithm.rollout_correction.bypass_mode=True
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
  actor_rollout_ref.actor.entropy_from_logits_chunk_size=2048
  actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
  ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER}
  actor_rollout_ref.rollout.agent.num_workers=${AGENT_WORKERS}
  ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter
  ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT}
  "++actor_rollout_ref.rollout.custom.agent_framework.log_dir=${AGENT_LOG_DIR}"
  ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False
  ++actor_rollout_ref.rollout.custom.agent_framework.trajectory_postprocessor_fqn=examples.claude_code_kernel_task.trajectory_processor.process_trajectories
  ++actor_rollout_ref.rollout.custom.agent_framework.trajectory_postprocessor_kwargs.selection=best
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=examples.claude_code_kernel_task.runner.run_triton_task
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${MAX_CONCURRENT_SESSIONS}
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.session_timeout_seconds=${SESSION_TIMEOUT_SECONDS}
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=all
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG}
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${SERVED_MODEL_NAME}
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.max_response_length=${max_response_length}
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.claude_run_timeout=${CLAUDE_RUN_TIMEOUT}
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.remote_docker_hosts=${REMOTE_DOCKER_HOSTS_PARSER}"
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.evaluator_npu_device_ids=${EVALUATOR_NPU_DEVICE_IDS_PARSER}"
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.evaluator_npu_lock_dir=${EVALUATOR_NPU_LOCK_DIR}
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.evaluator_npu_lock_timeout=${EVALUATOR_NPU_LOCK_TIMEOUT}
  actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
  actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
  actor_rollout_ref.rollout.prompt_length=${max_prompt_length}
  actor_rollout_ref.rollout.response_length=${max_response_length}
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.enable_prefix_caching=True
  actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
  actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
  actor_rollout_ref.rollout.max_model_len=${max_model_len}
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0
  actor_rollout_ref.rollout.val_kwargs.top_k=-1
  actor_rollout_ref.rollout.val_kwargs.do_sample=True
  actor_rollout_ref.rollout.val_kwargs.n=${val_resp_per_prompt}
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.enable_rollout_routing_replay=${enable_rollout_routing_replay}
  actor_rollout_ref.rollout.dtype=bfloat16
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=${TOOL_PARSER}
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
  actor_rollout_ref.nccl_timeout=9600
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.free_cache_engine=True
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
  actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
  actor_rollout_ref.ref.megatron.use_mbridge=True
  actor_rollout_ref.ref.megatron.seed=${seed}
  actor_rollout_ref.ref.megatron.dtype=bfloat16
  actor_rollout_ref.ref.megatron.use_remove_padding=True
  actor_rollout_ref.ref.megatron.pad_bshd_to_minibatch_max=False
  actor_rollout_ref.ref.megatron.param_offload=True
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
  actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}
  actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
  trainer.critic_warmup=0
  trainer.logger=['console']
  trainer.project_name="${project_name}"
  trainer.experiment_name="${exp_name}"
  trainer.val_before_train=${val_before_train}
  trainer.device=cuda
  trainer.total_epochs=${total_epochs}
  trainer.resume_mode=auto
  trainer.log_val_generations=10
  trainer.default_local_dir="${CKPTS_DIR}"
  trainer.nnodes=${NNODES}
  trainer.n_gpus_per_node=${NGPUS_PER_NODE}
  trainer.save_freq=${save_freq}
  trainer.test_freq=${test_freq}
  "$@"
)

ray job submit --working-dir="${WORKING_DIR}" "${RUNTIME_ENV_ARGS[@]}" \
  -- env RAY_OVERRIDE_JOB_RUNTIME_ENV=1 "${MAIN_CMD[@]}" 2>&1 | tee -i "${LOG_DIR}/${exp_name}.log"
