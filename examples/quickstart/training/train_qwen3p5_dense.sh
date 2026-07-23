#!/usr/bin/env bash
set -xeuo pipefail

project_name=${PROJECT_NAME:-"Uni-Agent-Qwen3.5-4B-megatron"}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H)_exp"}

MODEL_PATH=${MODEL_PATH:-"${DATA_DIR}/models/Qwen3.5-4B"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/data/uni_agent/swe_rebench_filtered_1150.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_DIR}/data/uni_agent/swe_bench_verified.parquet"}

RUNTIME_ENV=${RUNTIME_ENV:-"${RUNTIME_DIR}/data/uni_agent/runtime_env.yaml"}
CKPTS_DIR=${CKPTS_DIR:-"${RUNTIME_DIR}/ckpts/${project_name}/${exp_name}"}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${RUNTIME_DIR}/logs/${project_name}/${exp_name}"}
# Must be launched from the repository root so Ray packages both `verl/` and `uni_agent/`.
# --- Agent-framework rollout (replaces the swe_agent agent-loop) --------------
# Run-wide task base (agent + sandbox + sampling), loaded from this YAML by
# uni_agent.framework.task_runner.run_task and deep-merged onto each row's task.
# Same file-path idea as the old agent_loop_config_path; new (task-config) schema.
TASK_CONFIG=${TASK_CONFIG:-"examples/quickstart/training/task_config_claude_code.yaml"}
TOOL_PARSER=${TOOL_PARSER:-"qwen3_coder"}    # gateway tool-call parser; MUST match the model chat template
GATEWAY_COUNT=${GATEWAY_COUNT:-8}            # gateway actors fronting the engine
CONCURRENCY=${CONCURRENCY:-256}              # max in-flight rollout sessions (runner cap)
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "${MODEL_PATH}")"}

rollout_mode=${ROLLOUT_MODE:-"async"}
rollout_name=${ROLLOUT_NAME:-"vllm"} # sglang or vllm

# Algorithm parameters
adv_estimator=${ADV_ESTIMATOR:-grpo}

use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.0}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}

clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
clip_ratio_c=${CLIP_RATIO_C:-10.0}

# Response length parameters
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 8))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 128))}
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-False}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-$((1024 * 4))}  # unused
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}

loss_agg_mode=${LOSS_AGG_MODE:-"token-mean"}
loss_mode=${LOSS_MODE:-vanilla}

# Algorithm
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
val_temperature=${VAL_TEMPERATURE:-1.0}
val_top_p=${VAL_TOP_P:-0.95}
val_top_k=${VAL_TOP_K:--1}

# Performance Related Parameter
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
offload=${OFFLOAD:-False}
gen_tp=${GEN_TP:-2}
train_tp=${TP:-4}
train_pp=${PP:-1}
train_cp=${CP:-2}
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))

optimizer_offload_fraction=${OFFLOAD_FRACTION:-1.0}

# install mbridge
# pip3 install git+https://github.com/ISEEKYAN/mbridge
USE_MBRIDGE=${USE_MBRIDGE:-True}
USE_DIST_CKPT=${USE_DIST_CKPT:-False}

# V1 colocate_async topology. colocate_async colocates actor + rollout on the same
# GPUs (rollout replicas sleep during the train step), so NNODES is the TOTAL node
# count (replaces the old fully-async NNODES_ROLLOUT + NNODES_TRAIN split).
NNODES=${NNODES:-8}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# parameter_sync_step defaults to 1 for colocate_async, so train_batch_size
# (prompts/step) only needs to be > 0 (the old async mode used train_batch_size=0).
# num_warmup_batches pre-fills the rollout pipeline before the first train step.
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-32}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-16}
num_warmup_batches=${NUM_WARMUP_BATCHES:-1}
lr_decay_steps=${LR_DECAY_STEPS:-2000}
test_freq=${TEST_FREQ:-10}

# ============================================================================
# Rollout correction is disabled by default for the standard GRPO + PPO
# baseline. Override these variables to enable behavior-anchor or decoupled
# rollout-correction experiments.
# ============================================================================
bypass_mode=${BYPASS_MODE:-False}                                # True => old_log_prob = rollout_log_prob
bypass_loss_type=${BYPASS_LOSS_TYPE:-ppo_clip}                   # ppo_clip | reinforce
rollout_is=${ROLLOUT_IS:-null}                                   # PPO clip already applies the IS ratio
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0}                # single float => TIS upper clamp; "lo_hi" string => IcePop
rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORMALIZE:-False}  # normalize IS weights to mean=1.0 within a batch
rollout_rs=${ROLLOUT_RS:-null}                                   # no rejection sampling
rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD:-null}

ray job submit --no-wait --runtime-env $RUNTIME_ENV \
    -- python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches=${num_warmup_batches} \
    transfer_queue.enable=True \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${lr_decay_steps} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=$USE_DIST_CKPT \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    algorithm.rollout_correction.bypass_mode=${bypass_mode} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold="${rollout_rs_threshold}" \
    algorithm.rollout_correction.loss_type=${bypass_loss_type} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=${bypass_mode} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=${rollout_is} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs=${rollout_rs} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs_threshold="${rollout_rs_threshold}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=${bypass_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    +actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model'] \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER} \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT} \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir=${AGENT_LOG_DIR} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${CONCURRENCY} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=longest \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${SERVED_MODEL_NAME} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True \
    ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.nccl_timeout=9600 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.save_freq=10 \
    trainer.total_epochs=10 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.test_freq="${test_freq}"
