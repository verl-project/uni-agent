#!/usr/bin/env bash
# OpenClaw RL training (fully_async).
# GPU mapping:
#   ACTOR_GPUS -> trainer.n_gpus_per_node
#   ROLLOUT_GPUS -> rollout.n_gpus_per_node
#   PRM_GPUS -> reward.reward_model.n_gpus_per_node
set -xeuo pipefail

# -------- GPU allocation (sums must be <= NUM_GPUS) --------
NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
PRM_GPUS=${PRM_GPUS:-2}

if (( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS > NUM_GPUS )); then
  echo "ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS must be <= NUM_GPUS"
  echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, PRM_GPUS=${PRM_GPUS}, NUM_GPUS=${NUM_GPUS}"
  exit 1
fi

# -------- model / cluster (runtime-only overrides) --------
MODEL_PATH=${MODEL_PATH:-"/path/models/Qwen3-VL-4B-Instruct"}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-"${MODEL_PATH}"}
PROJECT_NAME=${PROJECT_NAME:-"OpenClaw-RL"}
EXP_NAME=${EXP_NAME:-"openclaw-rl"}
CKPTS_DIR=${CKPTS_DIR:-"/path/outputs/checkpoints/openclaw/${EXP_NAME}"}
NNODES=${NNODES:-1}

# -------- rl proxy knobs (read by the rollouter from env) --------
export OPENCLAW_RL_HOST=${OPENCLAW_RL_HOST:-"0.0.0.0"}
export OPENCLAW_RL_PORT=${OPENCLAW_RL_PORT:-30000}
export OPENCLAW_RL_API_KEY=${OPENCLAW_RL_API_KEY:-""}

# -------- sizes / PRM / sampling --------
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-4096}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-2048}
PPO_MINI_BSZ=${PPO_MINI_BSZ:-8}          # == required_samples per trainer step (require_batches=1)
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
LR=${LR:-1e-6}
PRM_M=${PRM_M:-3}
PRM_TEMPERATURE=${PRM_TEMPERATURE:-0.6}
PRM_MAX_TOKENS=${PRM_MAX_TOKENS:-2048}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-1000}

python3 -m uni_agent.openclaw.rl.train_entry \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LEN} \
  actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LEN} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_GPUS} \
  actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
  actor_rollout_ref.rollout.top_p=${TOP_P} \
  actor_rollout_ref.rollout.top_k=${TOP_K} \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN)) \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BSZ} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
  actor_rollout_ref.actor.optim.lr=${LR} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
  rollout.nnodes=${NNODES} \
  rollout.n_gpus_per_node=${ROLLOUT_GPUS} \
  rollout.total_rollout_steps=${TOTAL_ROLLOUT_STEPS} \
  reward.reward_model.model_path="${PRM_MODEL_PATH}" \
  reward.reward_model.n_gpus_per_node=${PRM_GPUS} \
  reward.reward_model.nnodes=${NNODES} \
  reward.reward_model.rollout.tensor_model_parallel_size=${PRM_GPUS} \
  +reward.reward_model.openclaw.prm_m=${PRM_M} \
  +reward.reward_model.openclaw.temperature=${PRM_TEMPERATURE} \
  +reward.reward_model.openclaw.max_tokens=${PRM_MAX_TOKENS} \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.logger=['console','tensorboard'] \
  trainer.nnodes=${NNODES} \
  trainer.n_gpus_per_node=${ACTOR_GPUS} \
  trainer.default_local_dir="${CKPTS_DIR}" \
  "$@"
