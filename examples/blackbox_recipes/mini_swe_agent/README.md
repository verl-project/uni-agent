# Mini-SWE-Agent Blackbox Recipe

This recipe runs mini-swe-agent through the unified runtime introduced by the
Task/Agent/Sandbox refactor. The recipe owns only AKernel-specific adaptation
and launch configuration; the episode lifecycle lives in the shared runtime.

## Runtime flow

```text
Agent Framework + Gateway session
  -> recipe task_runner.run_task
       - converts legacy OpenYuanRong rows to a Task Config
       - binds the session tunnel to the AKernel sandbox
  -> uni_agent.framework.task_runner.run_task
  -> SWEBenchTask
       -> AKernelSandbox
       -> MiniSweAgentAgent
       -> SWE-bench reward
  -> reward_info -> Gateway trajectory
```

`MiniSweAgentAgent` runs mini-swe-agent inside the task sandbox and points its
LiteLLM model at the current Gateway session. `SWEBenchTask` owns sandbox
lifecycle and reward evaluation, so the recipe no longer implements either in
a custom agent runner.

## Files

| File | Purpose |
|---|---|
| `task_config.yaml` | Default SWE task, AKernel sandbox, and mini-swe-agent settings |
| `task_runner.py` | Legacy-row normalization and per-session AKernel tunnel binding |
| `akernel_sandbox.py` | AKernel implementation of the unified `Sandbox` interface |
| `dataset.py` | Converts legacy `env`/`reward` rows to `tools_kwargs.task` |
| `parallel_infer.py` | Standalone vLLM + Gateway blackbox evaluation |
| `run_train.sh` | Megatron/V1 async training launcher |
| `Dockerfile.mini-swe-agent-tool` | Portable mini-swe-agent Python sidecar |

## AKernel setup

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="localhost,127.0.0.1,<gateway-host>"
export no_proxy="$NO_PROXY"

export AKERNEL_SERVER_ADDRESS="<host>:<port>"
export AKERNEL_TOKEN="<token>"
export TUNNEL_SSL_VERIFY=0
```

The recipe mounts the sidecar at `/opt/mini-swe-agent-venv`. The agent writes a
small driver and task JSON into `/tmp`, then launches them with the sidecar's
Python interpreter. The sidecar avoids installing dependencies for every
rollout while preserving the same `MiniSweAgentAgent` interface used by other
sandbox providers.

## Build the sidecar

```bash
bash examples/blackbox_recipes/mini_swe_agent/build_tool.sh

# Build and push to a registry reachable by AKernel.
bash examples/blackbox_recipes/mini_swe_agent/build_tool.sh \
  --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

## Standalone inference

```bash
MODEL_PATH=/path/to/Qwen3.5-9B \
DATA_PATH=/path/to/swe_bench_verified_openyuanrong.parquet \
MAX_SAMPLES=1 \
AGENT_MAX_TURNS=100 \
PROMPT_LENGTH=4096 \
RESPONSE_LENGTH=125952 \
TP=8 \
N_GPUS_PER_NODE=8 \
MAX_CONCURRENT_SESSIONS=1 \
bash examples/blackbox_recipes/mini_swe_agent/run_infer.sh
```

## Training

```bash
MODEL_PATH=/path/to/model \
TRAIN_DATA=/path/to/train.parquet \
VAL_DATA=/path/to/validation.parquet \
bash examples/blackbox_recipes/mini_swe_agent/run_train.sh
```

## Main settings

| Variable | Default | Description |
|---|---:|---|
| `AGENT_MAX_TURNS` | `100` | `MiniSweAgentConfig.step_limit` |
| `SWE_AGENT_RUN_TIMEOUT` | `7200` | Agent and sandbox wall-clock timeout |
| `SWE_AGENT_TOOL_IMAGE` | OpenYuanRong sidecar image | Portable mini-swe-agent runtime |
| `TASK_CONFIG` | `task_config.yaml` | Unified Task Config defaults |
| `MAX_CONCURRENT_SESSIONS` | inference: `8`, training: `128` | Concurrent task sessions |

Both the legacy OpenYuanRong schema (`tools_kwargs.env` +
`tools_kwargs.reward`) and the refactored schema (`tools_kwargs.task`) are
accepted. New datasets should use the refactored Task Config schema directly.
