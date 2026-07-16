# Quickstart Example

**Debug-only** quick start of the full agent-loop RL pipeline: host deployment,8 dummy samples, dummy reward Qwen3-0.6B. Verifies the pipeline runs; produces no meaningful training.

Full walkthrough: [Quick Start](https://uni-agent.readthedocs.io/en/latest/start/quick_start.html)

## Files

- `agent_config.yaml` — host-deployment agent loop
- `generate_dummy_data.py` — generates 8 dummy samples
- `runtime_env.yaml` — Ray runtime env
- `train.sh` — single-node GRPO launcher

## Run

```bash
python examples/quick_start/generate_dummy_data.py --local-save-dir ~/data/swe_agent
ray start --head
export MODEL_PATH=$HOME/models/Qwen3-0.6B
bash examples/quick_start/train.sh
```
