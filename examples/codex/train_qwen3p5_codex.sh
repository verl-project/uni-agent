#!/usr/bin/env bash
set -euo pipefail

project_name=${PROJECT_NAME:-"Uni-Agent-Codex-Qwen3.5-9B"}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H)_exp"}

MODEL_PATH=${MODEL_PATH:-"${DATA_DIR}/models/Qwen/Qwen3.5-9B"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/data/uni_agent/swe_bench_verified_openyuanrong.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_DIR}/data/uni_agent/swe_bench_verified_openyuanrong.parquet"}

RUNTIME_ENV=${RUNTIME_ENV:-"${RUNTIME_DIR}/data/uni_agent/runtime_env.yaml"}
CKPTS_DIR=${CKPTS_DIR:-"${RUNTIME_DIR}/ckpts/${project_name}/${exp_name}"}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${RUNTIME_DIR}/logs/${project_name}/${exp_name}"}
TASK_CONFIG=${TASK_CONFIG:-"examples/codex/task_config_codex.yaml"}
TOOL_PARSER=${TOOL_PARSER:-"qwen3_coder"}
GATEWAY_COUNT=${GATEWAY_COUNT:-1}
CONCURRENCY=${CONCURRENCY:-1}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "${MODEL_PATH}")"}
SESSION_TIMEOUT_SECONDS=${SESSION_TIMEOUT_SECONDS:-1800}
MASK_UNFINISHED_EPISODE=${MASK_UNFINISHED_EPISODE:-True}

max_prompt_length=${MAX_PROMPT_LENGTH:-8192}
max_response_length=${MAX_RESPONSE_LENGTH:-122880}
MAX_TOKENS_PER_TURN=${MAX_TOKENS_PER_TURN:-8192}
gen_tp=${GEN_TP:-8}
train_tp=${TP:-8}
train_pp=${PP:-1}
train_cp=${CP:-1}
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_prompt_bsz=${TRAIN_PROMPT_BSZ:-1}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-1}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-1}
num_warmup_batches=${NUM_WARMUP_BATCHES:-0}
lr_decay_steps=${LR_DECAY_STEPS:-10000}
test_freq=${TEST_FREQ:--1}
save_freq=${SAVE_FREQ:--1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}

# Qwen3.5 text-only recipe settings.
VLLM_REASONING_PARSER=${VLLM_REASONING_PARSER:-qwen3}

if ! [[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CONCURRENCY must be a positive integer, got ${CONCURRENCY}" >&2
    exit 2
fi
if ! [[ "${MAX_TOKENS_PER_TURN}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_TOKENS_PER_TURN must be a positive integer, got ${MAX_TOKENS_PER_TURN}" >&2
    exit 2
fi
ray job submit --no-wait --runtime-env $RUNTIME_ENV \
    -- env RAY_OVERRIDE_JOB_RUNTIME_ENV=1 \
    python3 -m verl.trainer.main_ppo \
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
    ++data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${lr_decay_steps} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    +actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model'] \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=100 \
    ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER} \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT} \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir=${AGENT_LOG_DIR} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${CONCURRENCY} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.session_timeout_seconds=${SESSION_TIMEOUT_SECONDS} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=longest \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${SERVED_MODEL_NAME} \
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=${MASK_UNFINISHED_EPISODE} \
    ++actor_rollout_ref.rollout.custom.agent_framework.max_tokens_per_turn=${MAX_TOKENS_PER_TURN} \
    ++actor_rollout_ref.rollout.custom.agent_framework.served_model_name=${SERVED_MODEL_NAME} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.62 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=NONE" \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode=align" \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=true" \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=true" \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=${VLLM_REASONING_PARSER}" \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.nccl_timeout=9600 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.ref.megatron.param_offload=False \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path=pkg://uni_agent.framework.task_runner \
    reward.custom_reward_function.name=score_from_runner_result \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.test_freq="${test_freq}"
