# Agent Inference Examples

This directory contains external API inference, verl-managed inference, and oracle verification examples.

For the full setup guides, see the documentation:

- [Run Agent Inference](https://uni-agent.readthedocs.io/en/latest/quickstart/agent-inference.html)
- [Verify Oracle Solutions](https://uni-agent.readthedocs.io/en/latest/quickstart/oracle-verification.html)

## Files

- `parallel_infer_api.py`: run tasks against an existing model API.
- `parallel_infer_verl.py`: let verl launch the rollout engine and use the training rollout path.
- `parallel_run_oracle.py`: run task-provided oracle solutions in parallel and verify task rewards.
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
