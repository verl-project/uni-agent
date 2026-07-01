# Mini-SWE-Agent In-Sandbox Execution

## Overview

`mini-swe-agent` runs inside the SWE-bench sandbox through a sidecar tool image.
The external runner creates the sandbox, mounts the tool image at
`/opt/mini-swe-agent`, starts the agent process, and evaluates the reward in the
same sandbox.

The agent executes commands through `LocalEnvironment` (local bash) inside the
sandbox and calls the LLM through the gateway URL passed in via stdin. The
`mini_swe` tool image uses
[python-build-standalone](https://github.com/astral-sh/python-build-standalone)
to build an isolated Python environment, then copies the result into a minimal
`FROM scratch` final stage, so the sandbox base image does not need to provide
Python for the sidecar tool runtime.

**This recipe is self-contained.** It shares only
[`../sandbox_client.py`](../sandbox_client.py) with the claude-code recipe;
everything else (`dataset.py`, `reward.py`, `run_agent.py`, `build_tool.sh`,
`run_train.sh`, config) lives in this directory and does not depend on
`claude_code/`.

**Supported runners:**

| runner | Description |
|--------|-------------|
| `mini_swe` | mini-swe-agent sidecar runner |

**Supported sandbox types:**

| Type | Description |
|------|-------------|
| openyuanrong | Uses `akernel_sdk.Mount` and `sandbox.commands.run()` |

## Architecture

```text
[Rollouter Host: mini_swe_agent_runner]
  |
  |-- SandboxClient.create(image, sidecar_image, sidecar_target="/opt/mini-swe-agent")
  |     `-- akernel: Sandbox(mounts=[Mount(target="/opt/mini-swe-agent", ...)])
  |
  |-- sandbox.run("<tool entrypoint>")
  |     `-- [Inside Sandbox]
  |           /opt/mini-swe-agent/bin/python /opt/mini-swe-agent/bin/run_agent.py
  |           stdin <- task config JSON (task, gateway_url, agent)
  |           commands run inside the SWE-bench sandbox
  |           stdout -> agent execution result JSON
  |
  |-- parse agent result
  |-- SandboxEnvForReward(sandbox) -> evaluate_in_env()
  `-- POST session.reward_info_url
```

## Prerequisites

1. **AKernel** — set `AKERNEL_SERVER_ADDRESS` and `AKERNEL_TOKEN`.
2. **Tool image** — build the mini-swe-agent tool image and push it to a remote
   registry if the sandbox service cannot access local Docker images.

## 1. Build Tool Image

`mini_swe` is injected into the SWE-bench sandbox as a sidecar tool image. Use
`build_tool.sh` to build it.

| Default tool image | Dockerfile | Sandbox mount path | Image contents |
|--------------------|------------|--------------------|----------------|
| `mini-swe-agent-tool:latest` | `Dockerfile.mini-swe-agent-tool` | `/opt/mini-swe-agent` | Standalone Python 3.12, `mini-swe-agent`, `litellm`, and `run_agent.py` |

```bash
# Use the default PyPI source.
bash examples/blackbox_recipes/mini_swe_agent/build_tool.sh

# Use a custom PyPI mirror.
bash examples/blackbox_recipes/mini_swe_agent/build_tool.sh --pip-index https://pypi.tuna.tsinghua.edu.cn/simple/

# Build and push to a remote registry.
bash examples/blackbox_recipes/mini_swe_agent/build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

The `mini_swe` Python runtime is fully isolated from the sandbox container's
Python.

### Build Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TOOL_IMAGE` | `mini-swe-agent-tool` | Image name |
| `TOOL_TAG` | `latest` | Image tag |
| `PIP_INDEX_URL` | unset, use PyPI | pip index URL (`--pip-index`) |

After pushing, point training at it with `SWE_AGENT_TOOL_IMAGE`.

## 2. Training (Fully Async)

```bash
AKERNEL_SERVER_ADDRESS="6.2.179.37:8888" \
AKERNEL_TOKEN="<token>" \
SWE_AGENT_TOOL_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest \
MODEL_PATH=~/models/Qwen3.5-9B \
bash examples/blackbox_recipes/mini_swe_agent/run_train.sh
```

The training YAML keeps `mini_swe` as the only runner:

```yaml
agent_runner_fqn: examples.blackbox_recipes.mini_swe_agent.mini_swe_agent_runner.mini_swe_agent_runner
```

## 3. Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MAX_TURNS` | `100` | mini-swe-agent `step_limit` (the agent's turn budget); read by the runner from the `AGENT_MAX_TURNS` env var |
| `SWE_AGENT_EVAL_TIMEOUT` | `600` | Reward evaluation timeout (seconds) |
| `SWE_AGENT_RUN_TIMEOUT` | `7200` | Max wall time for the agent process in the sandbox |
| `SWE_AGENT_TOOL_IMAGE` | `swr.cn-east-3.myhuaweicloud.com/openyuanrong/mini-swe-agent-tool:latest` | Sidecar tool image |
| `CONDA_ENV` | `testbed` | Conda env activated inside the sandbox before running the agent |
