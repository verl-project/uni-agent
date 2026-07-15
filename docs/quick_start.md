# Uni-Agent Quick Start

> Minimal end-to-end pipeline: host deployment + dummy data + dummy reward,
> using Qwen3-0.6B.

---

## 1. Prerequisites

- A working verl GRPO environment (NPU or GPU)
- Python ≥ 3.10

## 2. Clone & Install

```bash
git clone https://github.com/verl-project/uni-agent.git ~/uni-agent
cd ~/uni-agent

# verl submodule — provides the GRPO training engine
git submodule update --init --recursive
pip install --no-deps -e ./verl

# Other dependencies
pip install swe-rex loguru pydantic pydantic_settings aiohttp
```

## 3. Download Model & Replace Chat Template

### 3.1 Download the model

```bash
hf download Qwen/Qwen3-0.6B-Instruct --local-dir $HOME/models/Qwen3-0.6B-Instruct
```


## 4. Create Dummy Reward

Create `~/uni-agent/uni_agent/reward/dummy.py`:

```python
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec

@register_reward_spec("dummy")
class DummyRewardSpec(AbstractRewardSpec):
    def __init__(self, expected: str = "", **kwargs):
        self.expected = expected

    async def compute_reward(self, interaction_result: dict, **kwargs):
        trajectory = interaction_result.get("trajectory", [])
        submitted = any(
            step.exit_reason == "finished" for step in trajectory
        )
        return (1.0 if submitted else 0.0), {"submitted": submitted}
```

Edit `~/uni-agent/uni_agent/reward/registry.py` and add to the `REWARD_SPEC_MODULES` dict:

```python
"dummy": "uni_agent.reward.dummy",
```

## 5. Create Agent Config

Create `~/uni-agent/agent_config_host.yaml`:

```yaml
- name: swe_agent
  _target_: uni_agent.agent_loop.UniAgentLoop
  tool_parser: hermes   # "qwen3_coder" for XML, "hermes" for JSON tool calls
  concurrency: 16
  log_dir: ~/logs/agent
  interaction:
    action_timeout: 30
    max_turns: 10
  env:
    deployment:
      type: host
    tool_install_dir: ~/.local/bin
  tools:
    - name: execute_bash
    - name: submit
  reward:
    name: dummy
```

## 6. Create Dummy Training Data

Create `generate_dummy_data.py`:

```python
import os
import pandas as pd

DATA_DIR = os.path.expanduser("~/data/swe_agent")
os.makedirs(DATA_DIR, exist_ok=True)

samples = []
for i in range(8):
    samples.append({
        # prompt MUST be list[dict], NOT a plain string.
        # agent_loop.py:79 does list(kwargs["raw_prompt"]).
        "prompt": [
            {
                "role": "system",
                "content": "You are a coding assistant. Use bash to explore "
                           "and fix the bug. When done, call submit.",
            },
            {
                "role": "user",
                "content": "Fix the bug: the function add(a, b) should "
                           "return a + b, but it returns a - b instead.",
            },
        ],
        "agent_name": "swe_agent",
        "extra_info": {
            "index": i,
            "task_id": f"dummy-{i}",
            "tools_kwargs": {"dummy": "placeholder"},
        },
    })

df = pd.DataFrame(samples)
df.to_parquet(os.path.join(DATA_DIR, "dummy_agent_train.parquet"))
print(f"Generated {len(df)} samples")
```

```bash
python3 generate_dummy_data.py
```

## 7. Create Training Script

Create `~/uni-agent/train_dummy.sh`:

```bash
#!/usr/bin/env bash
set -xeuo pipefail

project_name='Uni-Agent-Dummy'
exp_name='GRPO-Dummy-Debug'

MODEL_PATH=${MODEL_PATH:-"$HOME/models/Qwen3-0.6B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/swe_agent/dummy_agent_train.parquet"}
AGENT_CONFIG_PATH=${AGENT_CONFIG_PATH:-"${HOME}/uni-agent/agent_config_host.yaml"}

cd ~/uni-agent
export PYTHONPATH=~/uni-agent:$PYTHONPATH
python3 -c "import uni_agent.reward.dummy"

gen_tp=1
train_tp=1
hybrid_engine=True

max_prompt_length=4096
max_response_length=2048

python3 -m verl.trainer.main_ppo \
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
```

## 8. Launch Training

```bash
ray start --head

cd ~/uni-agent
export NGPUS_PER_NODE=1
bash train_dummy.sh
```

## 9. Verify

```bash
ls ~/logs/agent/
grep "STEP 1" ~/logs/agent/*/run.log     # agent loop ran
grep "reward_score" ~/logs/agent/*/run.log
```



## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `response_mask must contain at least one valid token` | Check run.log for the actual crash; increase `max_response_length` |
| `string indices must be integers, not 'str'` | `prompt` must be `list[dict]`, not a plain string (see step 6) |
| `No function call found in the response` | Set `tool_parser: hermes` in agent config |
