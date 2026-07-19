# Agent Inference Examples

This directory contains the external API and verl-managed inference examples.

For the full setup guide, see the documentation:

[Run Agent Inference](https://uni-agent.readthedocs.io/en/latest/quickstart/agent-inference.html)

## Files

- `parallel_infer_api.py`: run tasks against an existing model API.
- `parallel_infer_verl.py`: let verl launch the rollout engine and use the training rollout path.
- `parallel_verify_swe.py`: verify generated SWE patches.
- `task_config.yaml`: ReAct SWE-Bench task config.
- `task_config_claude_code.yaml`: Claude Code SWE-Bench task config.
- `runtime_env.yaml`: Ray runtime env example.

Minimal external API example:

```bash
BASE_URL=http://localhost:8000/v1 MODEL=<served-model-name> \
python examples/inference/parallel_infer_api.py \
    --data-path ~/data/swe_agent/swe_bench_verified.parquet \
    --task-config examples/inference/task_config.yaml \
    --limit 8
```
