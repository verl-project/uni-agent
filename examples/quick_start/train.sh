#!/usr/bin/env bash
# Quickstart: single-node GRPO on 8 dummy samples with Qwen3-0.6B.
# Debug-only -- verifies the pipeline runs; produces no meaningful training.
set -xeuo pipefail

project_name='Uni-Agent-Quickstart'
exp_name='GRPO-Dummy-Quickstart'

MODEL_PATH=${MODEL_PATH:-"$HOME/models/Qwen3-0.6B"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/swe_agent/dummy_agent_train.parquet"}
RUNTIME_ENV=${RUNTIME_ENV:-"examples/quick_start/runtime_env.yaml"}
# Must be launched from the repository root so Ray packages both `verl/` and `uni_agent/`.
AGENT_CONFIG_PATH=${AGENT_CONFIG_PATH:-"examples/quick_start/agent_config.yaml"}

gen_tp=1
train_tp=1
hybrid_engine=True

max_prompt_length=4096
max_response_length=2048

ray job submit --no-wait --runtime-env $RUNTIME_ENV \
    -- python3 -m verl.trainer.main_ppo \
    --config-name='ppo_megatron_trainer.yaml' \
    hydra.searchpath=[pkg://verl.trainer.config] \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TRAIN_FILE}" \
    data.prompt_key=prompt \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=True \
    data.train_batch_size=8 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_CONFIG_PATH} \
    actor_rollout_ref.rollout.agent.default_agent_loop=swe_agent \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.hybrid_engine=${hybrid_engine} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward.reward_manager.name=dapo \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE:-1} \
    trainer.test_freq=1000
