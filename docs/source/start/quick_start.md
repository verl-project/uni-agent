# Quick Start

This is a **debug-only** end-to-end run of the agent-loop RL pipeline with the smallest possible setup: **host deployment + 8 dummy samples + a dummy reward**, on **Qwen3-0.6B**. It verifies that the full path works — data → agent loop → trajectory → reward → GRPO training — and nothing more. It produces **no meaningful accuracy or training gains**. For real training, use a real dataset, a real reward, and a larger model (see[Agent Reinforcement Learning](agent_train.md)). The runnable files live under `examples/quick_start/`.

---

## Prerequisites

- Uni-Agent installed (see [Installation](installation.md), "Single-Node Trial")
- A Qwen3-0.6B checkpoint available locally
- Python ≥ 3.10

## Components

Everything is already in the repo — no files to create:

| File | Purpose |
|---|---|
| `uni_agent/reward/dummy.py` | Dummy reward: 1.0 if the agent called `submit`, else 0.0 |
| `examples/quick_start/agent_config.yaml` | Host-deployment agent loop (bash + submit tools) |
| `examples/quick_start/generate_dummy_data.py` | Generates 8 dummy training samples |
| `examples/quick_start/runtime_env.yaml` | Ray runtime env (packages the repo, sets `PYTHONPATH`) |
| `examples/quick_start/train.sh` | Single-node GRPO launcher (`ray job submit`) |

The agent config uses `tool_parser: hermes` so it works with Qwen3's default
JSON tool-call format (no chat-template change needed).

## Generate data & train

```bash
cd ~/uni-agent

python examples/quick_start/generate_dummy_data.py --local-save-dir ~/data/swe_agent

ray start --head
export MODEL_PATH=$HOME/models/Qwen3-0.6B
export NGPUS_PER_NODE=1
bash examples/quick_start/train.sh
```

## Verify

```bash
grep "STEP 1" ~/logs/agent/*/run.log       # agent loop ran
grep "reward_score" ~/logs/agent/*/run.log  # reward computed
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `response_mask must contain at least one valid token` | Check run.log for the real crash; increase `max_response_length` |
| `string indices must be integers, not 'str'` | `prompt` must be `list[dict]`, not a plain string |
| `No function call found in the response` | Set `tool_parser: hermes` in the agent config |
| `Cannot write struct with no child field` | `extra_info.tools_kwargs` must be non-empty |
| Tool install permission denied | Set `tool_install_dir: ~/.local/bin` in the agent config |
