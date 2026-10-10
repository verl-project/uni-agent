# jiuwenswarm Blackbox Sidecar Integration

Integrates the [jiuwenswarm](https://gitcode.com/openJiuwen/jiuwenswarm) agent
(PyPI: `workswarm==0.2.6`) into uni-agent as a sidecar tool image for SWE-bench
code repair tasks.

## Architecture

- **Tool image**: `Dockerfile.jiuwenswarm-tool` builds a `FROM scratch` image
  with python-build-standalone + `workswarm` + `run_agent.sh` entrypoint.
- **Sandbox**: openyuanrong remote sandbox with the tool image mounted at
  `/opt/jiuwenswarm`.
- **Host agent**: `uni_agent/agents/jiuwenswarm/agent.py` (`JiuwenswarmAgent`)
  base64-pipes the task prompt into the bash entrypoint, passes the gateway
  config via env vars, and parses the result JSON.
- **Task config**: `task_config_jiuwenswarm.yaml` declares sandbox mounts and
  agent params for two tasks sharing one prompt template: `swe_rebench` (RL
  training) and `swe_bench` (evaluation).
- **Training**: `run_train.sh` launches Megatron async PPO training.

## Build

```bash
# Build the tool image
bash examples/jiuwenswarm_swe_task/build_tool.sh

# Build with a custom pip index (e.g., mirror)
bash examples/jiuwenswarm_swe_task/build_tool.sh --pip-index https://pypi.tuna.tsinghua.edu.cn/simple/

# Build, tag and push to a registry
bash examples/jiuwenswarm_swe_task/build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

## Run (inference / evaluation)

verl brings the engine up and routes rollouts through the agent-framework
gateway:

```bash
ray job submit --address http://127.0.0.1:8265 \
  -- python examples/inference/parallel_infer_verl.py \
  --task-config examples/jiuwenswarm_swe_task/task_config_jiuwenswarm.yaml \
  --data-path /path/to/swe_bench_verified.parquet \
  --model-path /path/to/Qwen3.5-4B \
  --tool-parser qwen3_coder --tensor-parallel-size 2 \
  --nnodes 1 --n-gpus-per-node 2 --limit 1 --concurrency 1
```

## Training

```bash
bash examples/jiuwenswarm_swe_task/run_train.sh
```

## Files

| File | Purpose |
|---|---|
| `Dockerfile.jiuwenswarm-tool` | Sidecar tool image build |
| `build_tool.sh` | Build/push helper |
| `run_agent.sh` | In-sandbox bash entrypoint |
| `task_config_jiuwenswarm.yaml` | Task + sandbox defaults |
| `run_train.sh` | Megatron RL training launch |
| `README.md` | This file |
