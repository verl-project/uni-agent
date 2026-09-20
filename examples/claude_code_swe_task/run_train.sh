#!/usr/bin/env bash
set -xeuo pipefail

project_name=${PROJECT_NAME:-"cc-yuanrong-qwen3p5-9b-grpo"}
exp_name=${EXP_NAME:-"swe_agent_$(date +%Y%m%d_%H%M)"}

MODEL_PATH=${MODEL_PATH:-"${DATA_DIR}/models/Qwen3.5-9B"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/data/uni_agent/swe_rebench_filtered.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_DIR}/data/uni_agent/swe_bench_verified.parquet"}

RUNTIME_ENV=${RUNTIME_ENV:-"${RUNTIME_DIR}/data/uni_agent/runtime_env_openyuanrong.yaml"}
CKPTS_DIR=${CKPTS_DIR:-"${RUNTIME_DIR}/ckpts/${project_name}/${exp_name}"}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${RUNTIME_DIR}/logs/${project_name}/${exp_name}"}

# Must be launched from the repository root so Ray packages both `verl/` and `uni_agent/`.
# --- Agent-framework rollout (replaces the swe_agent agent-loop) --------------
TASK_CONFIG="${TASK_CONFIG:-examples/claude_code_swe_task/task_config_claude_code_openyuanrong.yaml}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
GATEWAY_COUNT="${GATEWAY_COUNT:-8}"
CONCURRENCY="${CONCURRENCY:-256}"
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-32}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
MASK_UNFINISHED_EPISODE="${MASK_UNFINISHED_EPISODE:-False}"
TRAJECTORY_SELECTION="${TRAJECTORY_SELECTION:-longest}"

rollout_mode="${ROLLOUT_MODE:-async}"
rollout_name="${ROLLOUT_NAME:-vllm}"

# Algorithm parameters.
adv_estimator="${ADV_ESTIMATOR:-grpo}"

use_kl_in_reward="${USE_KL_IN_REWARD:-False}"
kl_coef="${KL_COEF:-0.0}"
use_kl_loss="${USE_KL_LOSS:-False}"
kl_loss_coef="${KL_LOSS_COEF:-0.0}"

clip_ratio_low="${CLIP_RATIO_LOW:-0.2}"
clip_ratio_high="${CLIP_RATIO_HIGH:-0.28}"
clip_ratio_c="${CLIP_RATIO_C:-10.0}"

loss_agg_mode="${LOSS_AGG_MODE:-token-mean}"
loss_mode="${LOSS_MODE:-vanilla}"

# Rollout-correction parameters.
bypass_mode="${BYPASS_MODE:-True}"
bypass_loss_type="${BYPASS_LOSS_TYPE:-ppo_clip}"
rollout_is="${ROLLOUT_IS:-null}"
rollout_is_threshold="${ROLLOUT_IS_THRESHOLD:-2.0}"
rollout_is_batch_normalize="${ROLLOUT_IS_BATCH_NORMALIZE:-False}"
rollout_rs="${ROLLOUT_RS:-null}"
rollout_rs_threshold="${ROLLOUT_RS_THRESHOLD:-null}"

# Sampling parameters.
temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:--1}"
val_temperature="${VAL_TEMPERATURE:-1.0}"
val_top_p="${VAL_TOP_P:-0.95}"
val_top_k="${VAL_TOP_K:--1}"

# Response-length parameters.
max_prompt_length="${MAX_PROMPT_LENGTH:-8000}"
max_response_length="${MAX_RESPONSE_LENGTH:-128000}"
max_model_len=$((max_prompt_length + max_response_length))
max_num_batched_tokens=$((max_prompt_length + max_response_length))

# Performance-related parameters. Qwen3.5 Megatron remains in BSHD, so
# remove-padding and dynamic batching stay disabled.
use_dynamic_bsz="${USE_DYNAMIC_BSZ:-False}"
gen_tp="${ROLLOUT_TP:-${GEN_TP:-2}}"
train_tp="${TRAIN_TP:-${TP:-2}}"
train_pp="${TRAIN_PP:-${PP:-2}}"
train_cp="${TRAIN_CP:-${CP:-8}}"
rollout_max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-20}"
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"

optimizer_offload_fraction="${OFFLOAD_FRACTION:-1.0}"
actor_lr="${ACTOR_LR:-1e-6}"

USE_MBRIDGE="${USE_MBRIDGE:-True}"
VANILLA_MBRIDGE="${VANILLA_MBRIDGE:-False}"
USE_DIST_CKPT="${USE_DIST_CKPT:-False}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-True}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"

# V1 colocate_async topology.
NNODES="${NNODES:-2}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-16}"

train_prompt_bsz="${TRAIN_PROMPT_BSZ:-64}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
train_prompt_mini_bsz="${PPO_MINI_BATCH_SIZE:-16}"
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
num_warmup_batches="${NUM_WARMUP_BATCHES:-1}"

total_epochs="${TOTAL_EPOCHS:-10}"
total_training_steps="${TOTAL_TRAINING_STEPS:-200}"
save_freq="${SAVE_FREQ:-10}"
test_freq="${TEST_FREQ:-10}"

actor_ppo_max_token_len=$((max_model_len / train_cp))
infer_ppo_max_token_len=$((max_model_len / train_cp))
export TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT=2400

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE="${VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE:-1}"

ray job submit --no-wait --runtime-env $RUNTIME_ENV \
    -- env RAY_OVERRIDE_JOB_RUNTIME_ENV=1 TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT=${TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT} \
    python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches="${num_warmup_batches}" \
    transfer_queue.enable=True \
    transfer_queue.metrics.enabled=True \
    trainer.device=npu \
    actor_rollout_ref.nccl_timeout=9600 \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings="${max_model_len}" \
    "data.train_files=['${TRAIN_FILE}']" \
    "data.val_files=['${TEST_FILE}']" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.train_batch_size="${train_prompt_bsz}" \
    data.gen_batch_size="${train_prompt_bsz}" \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    actor_rollout_ref.actor.policy_loss.loss_mode="${loss_mode}" \
    actor_rollout_ref.actor.checkpoint.strict=False \
    algorithm.adv_estimator="${adv_estimator}" \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_inflight_gen_batches=1 \
    algorithm.use_kl_in_reward="${use_kl_in_reward}" \
    algorithm.kl_ctrl.kl_coef="${kl_coef}" \
    actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
    actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
    actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
    actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
    actor_rollout_ref.actor.clip_ratio_c="${clip_ratio_c}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=False \
    actor_rollout_ref.actor.loss_agg_mode="${loss_agg_mode}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${train_prompt_mini_bsz}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ppo_micro_batch_size_per_gpu}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_ppo_max_token_len}" \
    actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True \
    actor_rollout_ref.actor.optim.main_grads_dtype=bf16 \
    '+actor_rollout_ref.actor.megatron.override_ddp_config={grad_reduce_in_fp32: false, overlap_grad_reduce: true, bucket_size: 50000000}' \
    actor_rollout_ref.actor.optim.lr="${actor_lr}" \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction="${optimizer_offload_fraction}" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge="${USE_MBRIDGE}" \
    actor_rollout_ref.actor.megatron.vanilla_mbridge="${VANILLA_MBRIDGE}" \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing="${USE_DIST_CKPT}" \
    actor_rollout_ref.actor.megatron.param_offload="${PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.megatron.optimizer_offload="${OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${train_tp}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${train_pp}" \
    actor_rollout_ref.actor.megatron.context_parallel_size="${train_cp}" \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max=True \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    algorithm.rollout_correction.bypass_mode="${bypass_mode}" \
    algorithm.rollout_correction.rollout_is="${rollout_is}" \
    algorithm.rollout_correction.rollout_is_threshold="${rollout_is_threshold}" \
    algorithm.rollout_correction.rollout_is_batch_normalize="${rollout_is_batch_normalize}" \
    algorithm.rollout_correction.rollout_rs="${rollout_rs}" \
    algorithm.rollout_correction.rollout_rs_threshold="${rollout_rs_threshold}" \
    algorithm.rollout_correction.loss_type="${bypass_loss_type}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode="${bypass_mode}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is="${rollout_is}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold="${rollout_is_threshold}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_batch_normalize="${rollout_is_batch_normalize}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs="${rollout_rs}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs_threshold="${rollout_rs_threshold}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type="${bypass_loss_type}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${log_prob_micro_batch_size_per_gpu}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format="${TOOL_PARSER}" \
    actor_rollout_ref.rollout.agent.num_workers="${NUM_AGENT_WORKERS}" \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count="${GATEWAY_COUNT}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${AGENT_LOG_DIR}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="${CONCURRENCY}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection="${TRAJECTORY_SELECTION}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="${SERVED_MODEL_NAME}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True \
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode="${MASK_UNFINISHED_EPISODE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_gpu_memory_utilization}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
    actor_rollout_ref.rollout.prompt_length="${max_prompt_length}" \
    actor_rollout_ref.rollout.response_length="${max_response_length}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="${max_num_batched_tokens}" \
    actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
    actor_rollout_ref.rollout.max_num_seqs="${rollout_max_num_seqs}" \
    actor_rollout_ref.rollout.temperature="${temperature}" \
    actor_rollout_ref.rollout.top_p="${top_p}" \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${val_temperature}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${val_top_p}" \
    actor_rollout_ref.rollout.val_kwargs.top_k="${val_top_k}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name="${rollout_name}" \
    actor_rollout_ref.rollout.mode="${rollout_mode}" \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"' \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode=align \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=false \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${log_prob_micro_batch_size_per_gpu}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    actor_rollout_ref.ref.megatron.param_offload="${PARAM_OFFLOAD}" \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${train_tp}" \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${train_pp}" \
    actor_rollout_ref.ref.megatron.context_parallel_size="${train_cp}" \
    actor_rollout_ref.ref.megatron.use_remove_padding=False \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path=pkg://uni_agent.framework.task_runner \
    reward.custom_reward_function.name=score_from_runner_result \
    'trainer.logger=["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.save_freq="${save_freq}" \
    trainer.test_freq="${test_freq}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.total_training_steps="${total_training_steps}"
