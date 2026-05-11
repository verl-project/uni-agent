#!/usr/bin/env bash
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

# ================= basic =================
NNODES_ROLLOUT=${NNODES_ROLLOUT:-4}
NNODES_TRAIN=${NNODES_TRAIN:-4}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

ppo_mini_batch_size=16
rollout_n=8

# ================= path =================
WORKING_DIR=${PWD}
DATA_ROOT=${DATA_ROOT:-${HOME}/uni_agent_data}
PROMETHEUS_FILE=${PROMETHEUS_FILE:-/tmp/ray/session_latest/metrics/prometheus/prometheus.yml}

RUNTIME_ENV=$WORKING_DIR/examples/search_agent/runtime_env.yaml

# ================= wandb =================
project_name=search_agent
experiment_datetime=$(date +%Y%m%d_%H%M)
experiment_name=search-async-n$rollout_n-$experiment_datetime

# ================= data =================
asearcher_train=$DATA_ROOT/data/asearcher_uni_processed/train.parquet
asearcher_test=$DATA_ROOT/data/asearcher_uni_processed/test.parquet
filter_overlong_prompts=False

train_files="['$asearcher_train']"
test_files="['$asearcher_test']"
val_before_train=True

# ================= ckpt =================
model_name=Qwen3-30B-A3B-Thinking-2507
model_path=$DATA_ROOT/model/${model_name}
#model_path=$DATA_ROOT/model/Qwen3-30B-A3B-Instruct-2507
save_path=$DATA_ROOT/ckpts/$experiment_name

# ================= agent config =================
AGENT_CONFIG_PATH=$WORKING_DIR/examples/search_agent/agent_config.yaml

# Resolve Ray head IP for localwiki service
get_ray_head_ip() {
    local ip
    ip=$(python3 -c "import ray; ray.init(address='auto'); print(ray.util.get_node_ip_address())" 2>/dev/null) && { echo "$ip"; return 0; }
    if [ -n "${RAY_HEAD_IP:-}" ]; then
        echo "${RAY_HEAD_IP}"
        return 0
    fi
    echo "ERROR: Cannot determine Ray head IP. Set RAY_HEAD_IP or ensure Ray is running." >&2
    return 1
}

RAY_HEAD_IP=$(get_ray_head_ip)

if [[ "${RAY_HEAD_IP}" == *:* ]] && [[ "${RAY_HEAD_IP}" != *"["* ]]; then
    ray_host_for_url="[${RAY_HEAD_IP}]"
else
    ray_host_for_url="${RAY_HEAD_IP}"
fi

tmp_agent_config="examples/search_agent/agent_config_${experiment_datetime}.yaml"
sed -e "s|\${RAY_HEAD_IP}|${ray_host_for_url}|g" \
    "${AGENT_CONFIG_PATH}" > "${WORKING_DIR}/${tmp_agent_config}"

AGENT_CONFIG_PATH="${tmp_agent_config}"

# ================= algorithm =================
adv_estimator=grpo
loss_mode=gspo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

bypass_mode=False
rollout_is=token

clip_ratio_low=0.2
clip_ratio_high=0.28

actor_lr=1e-6
rollout_n_val=1
loss_agg_mode="token-mean"

save_freq=-1
test_freq=10
log_val_generations=30

# ================= performance =================
use_fused_kernels=False
disable_log_stats=False
enable_chunked_prefill=True

# ================= training =================
offload=False
TP_SIZE=4
CP_SIZE=4
PP_SIZE=2
VPP_SIZE=null
EP_SIZE=8
ETP_SIZE=1

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 126))
actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) / CP_SIZE))
log_prob_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) / CP_SIZE))

# ================= fully async specific =================
train_prompt_bsz=0
total_rollout_steps=200000
staleness_threshold=1.0
trigger_parameter_sync_step=8
require_batches=1
partial_rollout=True

# ================= inference =================
rollout_name=vllm
if [ "$rollout_name" = "vllm" ]; then
    export VLLM_USE_V1=1
fi
infer_tp=4
infer_dp=1
infer_ep=1
gpu_memory_utilization=0.8


# ================= Main command =================

ray job submit --no-wait \
    --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    hydra.searchpath=[pkg://verl.trainer.config] \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    algorithm.rollout_correction.bypass_mode=$bypass_mode \
    algorithm.rollout_correction.rollout_is=$rollout_is \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.prompt_key=prompt \
    data.train_batch_size=$train_prompt_bsz \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=$filter_overlong_prompts \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_fused_kernels=$use_fused_kernels \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_decay_steps=$total_rollout_steps \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP_SIZE \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$VPP_SIZE \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP_SIZE \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP_SIZE \
    actor_rollout_ref.actor.megatron.param_offload=$offload \
    actor_rollout_ref.actor.megatron.grad_offload=$offload \
    actor_rollout_ref.actor.megatron.optimizer_offload=$offload \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=2 \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.data_parallel_size=$infer_dp \
    actor_rollout_ref.rollout.expert_parallel_size=$infer_ep \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.enable_chunked_prefill=$enable_chunked_prefill \
    actor_rollout_ref.rollout.disable_log_stats=$disable_log_stats \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$rollout_n_val \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$AGENT_CONFIG_PATH \
    actor_rollout_ref.rollout.prometheus.enable=True \
    actor_rollout_ref.rollout.prometheus.port=9090 \
    actor_rollout_ref.rollout.prometheus.file=$PROMETHEUS_FILE \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.ref.megatron.param_offload=$offload \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.ref.megatron.context_parallel_size=$CP_SIZE \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP_SIZE \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP_SIZE \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES_TRAIN \
    trainer.val_before_train=$val_before_train \
    trainer.log_val_generations=$log_val_generations \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto \
    trainer.default_local_dir=$save_path \
    rollout.nnodes=$NNODES_ROLLOUT \
    rollout.n_gpus_per_node=$NGPUS_PER_NODE \
    rollout.total_rollout_steps=$total_rollout_steps \
    async_training.staleness_threshold=$staleness_threshold \
    async_training.trigger_parameter_sync_step=$trigger_parameter_sync_step \
    async_training.require_batches=$require_batches \
    async_training.partial_rollout=$partial_rollout

rm -f "${WORKING_DIR}/${tmp_agent_config}"
