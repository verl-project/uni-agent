#!/usr/bin/env bash
# This script is a demo for GRPO training of DeepSeek-V4-Flash using VeOmniEngine.
#
# Environment:
#   - transformers==5.9.0
#   - vllm==0.24.0
#   - veomni==0.1.12

set -xeuo pipefail

# ---- user-adjustable ----
trainer_nnodes=${NNODES:-16}
n_gpus_per_node=${N_GPUS_PER_NODE:-8}

project_name='Uni-Agent-DeepSeek-V4'
exp_name='deepseek_v4_flash_exp'

# ===================================== Algorithm =====================================
adv_estimator=grpo
loss_mode=vanilla
loss_agg_mode=seq-mean-token-mean

# reference policy
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.28

actor_lr=2e-6
critic_lr=2e-6
gae_gamma=1.0
gae_lam=0.95
critic_warmup=0

bypass_mode=False
calculate_entropy=False
if [[ $bypass_mode == True ]]; then
    calculate_entropy=True
fi

# Rollout correction, applied inside the bypass_mode policy loss.
# With both rollout_is and rollout_rs null this is numerically identical to
# plain PPO-clip anchored at pi_rollout, and only emits the rollout_corr/* diagnostics.
# Enable RS once the observed mismatch justifies it: seq_mean_k1 (length-normalized)
# is the only sequence-level option usable at these response lengths.
# Comma-separated multi-option specs must be single-quoted for Hydra (see run_qwen2_5_7b_fsdp_multi_rs.sh).
loss_type=ppo_clip
rollout_is=token
rollout_is_threshold=2.0
rollout_rs=null
rollout_rs_threshold=null

# ===================================== Data/Model =====================================
: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-V4-Flash-0731 model directory}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the Parquet produced by prepare_data.py with --harness}"
experiment_dir="${OUTPUT_DIR:-$PWD/outputs}/$project_name/$exp_name"

# Uni-Agent rollout config
TOOL_PARSER=${TOOL_PARSER:-"deepseek_v4"}
GATEWAY_COUNT=${GATEWAY_COUNT:-16}
CONCURRENCY=${CONCURRENCY:-1024}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-openai/DeepSeek-V4-Flash-0731}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"$experiment_dir/logs"}
# Unfinished episodes (react `token_limit` / `max_steps` / errors) stay trainable: burning the
# 147k context is a policy behavior we want to penalize, not exogenous noise. Masking them zeroed
# their gradient while their r=0 still lowered the GRPO baseline.
MASK_UNFINISHED_EPISODE=${MASK_UNFINISHED_EPISODE:-False}
TRAJ_SELECTION=${TRAJ_SELECTION:-longest}

train_batch_size=64
ppo_mini_batch_size=64
n_resp_per_prompt=8
n_resp_per_prompt_val=16
enable_thinking=${ENABLE_THINKING:-True}
reasoning_effort=${REASONING_EFFORT:-low}
drop_thinking=${DROP_THINKING:-False}

# Training config
usp_size=${usp_size:-8}
expert_size=${expert_size:-8}
use_remove_padding=True
use_dynamic_bsz=True
max_prompt_length=$((1024 * 16))
max_response_length=$((1024 * 144))
actor_max_token_len_per_gpu=$((1024 * 24))

# Inference config
rollout_name=vllm
infer_tp=8
infer_dp=1
infer_ep=8
infer_dcp=1
gpu_memory_utilization=0.6
# ---- end user-adjustable ----

# ---- no user adjustment needed below ----
# VeOmni config
ENGINE_CONFIG=(
    model_engine=veomni
    actor_rollout_ref.actor.veomni.enable_fsdp_offload=True
    # actor_rollout_ref.actor.veomni.param_offload=True
    # actor_rollout_ref.actor.veomni.optimizer_offload=True
    actor_rollout_ref.actor.veomni.enable_async_activation_offload=True
    actor_rollout_ref.actor.veomni.enable_full_shard=True
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=$usp_size
    actor_rollout_ref.actor.veomni.expert_parallel_size=$expert_size
    actor_rollout_ref.actor.veomni.moe_implementation=fused_triton
    actor_rollout_ref.actor.veomni.attn_implementation=eager
    actor_rollout_ref.actor.veomni.dsa_indexer_implementation=tilelang
    actor_rollout_ref.actor.veomni.dsa_attention_implementation=tilelang
    actor_rollout_ref.actor.veomni.mhc_implementation=tilelang
    actor_rollout_ref.actor.veomni.qat_implementation=fp8_blockwise
    actor_rollout_ref.actor.veomni.router_replay.mode=R3
    actor_rollout_ref.actor.veomni.cross_entropy_loss_implementation=chunk_loss
)

# Actor model config
ACTOR_CONFIG=(
    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=$use_remove_padding
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.actor.optim.lr=$actor_lr
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=$bypass_mode
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=${loss_type}
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=${rollout_is}
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=${rollout_is_threshold}
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs=${rollout_rs}
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold}
    actor_rollout_ref.actor.calculate_entropy=$calculate_entropy
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz
    actor_rollout_ref.actor.pad_to_length=True
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_max_token_len_per_gpu}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${actor_max_token_len_per_gpu}
)

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=$rollout_name
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp
    actor_rollout_ref.rollout.data_parallel_size=$infer_dp
    actor_rollout_ref.rollout.expert_parallel_size=$infer_ep
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.decode_context_parallel_size=$infer_dcp
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.prompt_length=$max_prompt_length
    actor_rollout_ref.rollout.response_length=$max_response_length
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization
    actor_rollout_ref.rollout.n=$n_resp_per_prompt
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8
)

AGENT_CONFIG=(
    transfer_queue.enable=True
    actor_rollout_ref.rollout.multi_turn.enable=True
    ++actor_rollout_ref.rollout.multi_turn.format=$TOOL_PARSER
    actor_rollout_ref.rollout.agent.num_workers=8
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=$GATEWAY_COUNT
    "++actor_rollout_ref.rollout.custom.agent_framework.log_dir=$AGENT_LOG_DIR"
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=$CONCURRENCY
    "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=$SERVED_MODEL_NAME"
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=$MASK_UNFINISHED_EPISODE
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=$TRAJ_SELECTION
)

DATA=(
    algorithm.adv_estimator=$adv_estimator
    algorithm.use_kl_in_reward=$use_kl_in_reward
    algorithm.kl_ctrl.kl_coef=$kl_coef
    algorithm.gamma=$gae_gamma
    algorithm.lam=$gae_lam
    algorithm.rollout_correction.bypass_mode=$bypass_mode
    algorithm.filter_groups.enable=True
    algorithm.filter_groups.metric=acc
    algorithm.rollout_correction.rollout_is=${rollout_is}
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold}
    algorithm.rollout_correction.rollout_is_batch_normalize=False
    algorithm.rollout_correction.rollout_rs=${rollout_rs}
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold}
    algorithm.rollout_correction.loss_type=${loss_type}
    "data.train_files=$TRAIN_FILE"
    "data.val_files=${VAL_FILE:-$TRAIN_FILE}"
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.train_batch_size=$train_batch_size
    data.max_prompt_length=$max_prompt_length
    data.max_response_length=$max_response_length
    data.filter_overlong_prompts=False
    data.filter_overlong_prompts_workers=64
    data.truncation='error'
    +data.apply_chat_template_kwargs.enable_thinking=$enable_thinking
    +data.apply_chat_template_kwargs.reasoning_effort=$reasoning_effort
    +data.apply_chat_template_kwargs.drop_thinking=$drop_thinking
)

TRAINER=(
    trainer.use_v1=True
    trainer.v1.trainer_mode=colocate_async
    trainer.v1.colocate_async.num_warmup_batches=1
    trainer.critic_warmup=$critic_warmup
    'trainer.logger=[console,wandb]'
    trainer.project_name=$project_name
    trainer.experiment_name=$exp_name
    trainer.n_gpus_per_node=${n_gpus_per_node}
    trainer.nnodes=$trainer_nnodes
    trainer.val_before_train=False
    trainer.log_val_generations=100
    trainer.save_freq=10
    trainer.test_freq=-1
    "trainer.default_local_dir=$experiment_dir/ckpts"
    trainer.total_epochs=10
    trainer.total_training_steps=500
)

########################### launch ###########################
python -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${TRAINER[@]}" \
    "${ENGINE_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${AGENT_CONFIG[@]}" \
    "$@"
