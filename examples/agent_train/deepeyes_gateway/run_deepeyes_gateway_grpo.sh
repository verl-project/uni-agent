#!/usr/bin/env bash

set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UNI_AGENT_REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
CONFIG_DIR="${UNI_AGENT_REPO_ROOT}/examples/agent_train/deepeyes_gateway/configs"
cd "${UNI_AGENT_REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export VERL_FORCE_TQ_NESTED_READBACK="${VERL_FORCE_TQ_NESTED_READBACK:-1}"
export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE:-http://127.0.0.1:18901/v1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

PROJECT_NAME="${PROJECT_NAME:-deepeyes_gateway_sync_real_data}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35_4b_deepeyes_gateway_grpo}"
MODEL_PATH="${MODEL_PATH:-/data1/models/Qwen/Qwen3.5-4B}"
TRAIN_FILE="${TRAIN_FILE:-/data1/datasets/deepeyes/data/data_0.1.2_visual_toolbox_v2.parquet}"
VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"

python3 -m verl.trainer.main_ppo_sync \
  --config-path="${CONFIG_DIR}" \
  --config-name=deepeyes_gateway_grpo \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  data.train_files="${TRAIN_FILE}" \
  "data.val_files=[${VAL_FILE}]" \
  data.train_batch_size=14 \
  data.max_prompt_length=4096 \
  data.max_response_length=1024 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=7 \
  trainer.nnodes=1 \
  'trainer.logger=[console,wandb,tensorboard]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.use_remove_padding=True \
  '+actor_rollout_ref.model.override_config.attn_implementation=eager' \
  actor_rollout_ref.actor.ppo_mini_batch_size=14 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.response_length=1024 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.dtype=float16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.custom.agent_framework.gateway_count=7 \
  actor_rollout_ref.rollout.custom.agent_framework.tool_config_path="${CONFIG_DIR}/image_zoom_in_tool_config.yaml" \
  actor_rollout_ref.rollout.custom.agent_framework.agent_runner_kwargs.max_turns=5
