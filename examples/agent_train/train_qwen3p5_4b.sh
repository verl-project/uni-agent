#!/usr/bin/env bash
set -xeuo pipefail

# Qwen3.5-4B agentic RL on SWE tasks via the uni_agent framework, running on
# verl's V1 unified trainer in `colocate_async` mode: the actor and an async
# rollout share the same GPUs (rollout replicas sleep during the train step),
# and rollout output flows through TransferQueue.
#
# The framework plugs in through `agent_loop_manager_class`: verl.trainer.main_ppo's
# TaskRunnerV1 honors that key, forces transfer_queue.enable, and calls our adapter's
# `generate_sequences` (fire-and-forget) which posts each session's reward into TQ as
# rm_scores. Because reward.reward_model.enable is False, the trainer's
# reward_loop_worker_handles are non-None, so `_compute_reward_colocate` is skipped and
# our posted rm_scores are used verbatim (no DAPO ground_truth needed).
#
# Launch from the repo root so Ray packages both `verl/` and `uni_agent/`.

RAY_DATA_HOME=/mnt/hdfs/yyding
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
# colocate_async colocates actor + rollout, so every node runs both. This is the
# TOTAL node count (replaces the old fully-async NNODES_ROLLOUT + NNODES_TRAIN split).
NNODES=${NNODES:-4}

project_name='Uni-Agent-Qwen3p5-4B'
exp_name=$(date +%Y%m%d%H)_exp

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3.5-4B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${RAY_DATA_HOME}/logs/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/uni_agent/swe_rebench_filtered_1150.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/uni_agent/swe_bench_verified.parquet"}
RUNTIME_ENV=${RUNTIME_ENV:-"${RAY_DATA_HOME}/data/uni_agent/runtime_env.yaml"}

# --- Agent-framework rollout (replaces agent_loop_config_path) ----------------
# Run-wide task base (agent + sandbox + sampling), loaded from this YAML by
# uni_agent.framework.task_runner.run_task and deep-merged onto each data row's task.
TASK_CONFIG=${TASK_CONFIG:-"examples/agent_train/task_config.yaml"}
TOOL_PARSER=${TOOL_PARSER:-"qwen3_coder"}
GATEWAY_COUNT=${GATEWAY_COUNT:-8}
CONCURRENCY=${CONCURRENCY:-1024}

rollout_mode="async"
rollout_name="vllm" # sglang or vllm

# Algorithm parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Response length parameters
max_prompt_length=$((1024 * 8))
max_response_length=$((1024 * 128))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))  # unused
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"
loss_mode=vanilla

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1
val_temperature=1.0
val_top_p=0.95
val_top_k=-1

# Performance Related Parameter
use_dynamic_bsz=True
offload=False
gen_tp=2
train_tp=4
train_pp=1
train_cp=2
train_ep=1
train_etp=1
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))

optimizer_offload_fraction=1.0

# install mbridge
# pip3 install git+https://github.com/ISEEKYAN/mbridge
USE_MBRIDGE=True
USE_DIST_CKPT=False

# V1 colocate_async batching. parameter_sync_step defaults to 1 for colocate_async,
# so train_batch_size (prompts/step) only needs to be > 0 (unlike the old async mode
# which used train_batch_size=0 + gen_batch_size=1). num_warmup_batches pre-fills the
# rollout pipeline before the first train step for actor/rollout overlap.
train_prompt_bsz=${train_prompt_bsz:-32}
n_resp_per_prompt=8
train_prompt_mini_bsz=16
num_warmup_batches=${num_warmup_batches:-1}
lr_decay_steps=${lr_decay_steps:-2000}
test_freq=10
save_freq=10
total_epochs=20

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
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
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
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp} \
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
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    +actor_rollout_ref.actor.checkpoint.save_contents=['hf_model'] \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER} \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT} \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir=${AGENT_LOG_DIR} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=inline_async \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${CONCURRENCY} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${MODEL_PATH} \
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
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp} \
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
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.test_freq=${test_freq}
