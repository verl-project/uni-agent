# SWE-bench Verified Inference Example

This directory contains the SWE-Bench Verified parallel interaction example.

For the full setup guide, see the documentation:

[Parallel Agent Interaction](https://uni-agent.readthedocs.io/en/latest/start/agent_interaction.html)

## Files

- `agent_config_modal.yaml`: Modal agent loop config.
- `agent_config_vefaas.yaml`: veFaaS agent loop config.
- `runtime_env.yaml`: Ray runtime env example.
- `parallel_infer.py`: parallel rollout script.
- `parallel_verify_swe.py`: optional verification script.

Minimal Modal example:

```bash
DEPLOYMENT=modal python examples/data_preprocess/swe_bench_verified.py \
    --local-save-dir ~/data/swe_agent

python examples/agent_interaction/parallel_infer.py \
    --data-path ~/data/swe_agent/swe_bench_verified_modal.parquet \
    --model-path /path/to/your/local/model \
    --agent-config-path examples/agent_interaction/agent_config_modal.yaml \
    --num-workers 8 \
    --max-turns 100 \
    --max-samples 4
```
