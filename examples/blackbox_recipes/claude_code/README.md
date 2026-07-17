# Claude Code In-Sandbox Execution

## Overview

Claude Code runs inside the SWE-bench sandbox through a sidecar tool image. The
external runner creates the sandbox, mounts the tool image at `/opt/claude-code`,
invokes the `claude` binary against the gateway URL, and evaluates the reward in
the same sandbox.

Unlike the mini-swe-agent recipe, there is no in-sandbox Python entrypoint
(`run_agent.py`): the runner builds a single `claude -p ...` command and executes
it directly. The agent reaches the LLM gateway through the sandbox-internal
tunnel (`ANTHROPIC_BASE_URL` rewritten to `http://127.0.0.1:<proxy_port>`).

The Claude Code tool image uses a Node builder to install the
`@anthropic-ai/claude-code` npm package, then copies the result into a minimal
`FROM scratch` final stage. The sandbox base image therefore does not need Node
or npm for the sidecar tool runtime.

**This recipe is self-contained.** It shares only
[`../sandbox_client.py`](../sandbox_client.py) with the mini-swe-agent recipe;
everything else (`dataset.py`, `reward.py`, `build_tool.sh`, `run_train.sh`,
config) lives in this directory and does not depend on `mini_swe_agent/`.

**Supported runners:**

| runner | Description |
|--------|-------------|
| `claude_code` | Claude Code sidecar runner |

**Supported sandbox types:**

| Type | Description |
|------|-------------|
| openyuanrong | Uses `akernel_sdk.Mount` and `sandbox.commands.run()` |

## Architecture

```text
[Rollouter Host: claude_code_runner]
  |
  |-- SandboxClient.create(image, sidecar_image, sidecar_target="/opt/claude-code")
  |     `-- akernel: Sandbox(mounts=[Mount(target="/opt/claude-code", ...)])
  |
  |-- sandbox.run("<env> /opt/claude-code/bin/claude -p <task> ...")
  |     `-- [Inside Sandbox]
  |           claude binary, ANTHROPIC_BASE_URL -> 127.0.0.1:<proxy_port>
  |           commands run inside the SWE-bench sandbox /testbed
  |
  |-- SandboxEnvForReward(sandbox) -> evaluate_in_env()
  `-- POST session.reward_info_url
```

## Prerequisites

1. **AKernel** — set `AKERNEL_SERVER_ADDRESS` and `AKERNEL_TOKEN`.
2. **Tool image** — build the claude-code tool image and push it to a remote
   registry if the sandbox service cannot access local Docker images.

## 1. Build Tool Image

`claude_code` is injected into the SWE-bench sandbox as a sidecar tool image.
Use `build_tool.sh` to build it.

| Default tool image | Dockerfile | Sandbox mount path | Image contents |
|--------------------|------------|--------------------|----------------|
| `claude-code-tool:latest` | `Dockerfile.claude-code-tool` | `/opt/claude-code` | Node-built `@anthropic-ai/claude-code` npm package |

```bash
# Use the default npm registry.
bash examples/blackbox_recipes/claude_code/build_tool.sh

# Use a custom npm mirror.
bash examples/blackbox_recipes/claude_code/build_tool.sh --npm-registry https://registry.npmmirror.com

# Pin a specific claude-code version.
bash examples/blackbox_recipes/claude_code/build_tool.sh --tool-version latest

# Build and push to a remote registry.
bash examples/blackbox_recipes/claude_code/build_tool.sh --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

### Build Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TOOL_IMAGE` | `claude-code-tool` | Image name |
| `TOOL_TAG` | `latest` | Image tag |
| `TOOL_VERSION` | `latest` | `@anthropic-ai/claude-code` package version (`--tool-version`) |
| `NPM_REGISTRY` | unset, use npm default | npm registry URL (`--npm-registry`) |

After pushing, point training at it with `CLAUDE_CODE_TOOL_IMAGE`.

## 2. Training (Fully Async)

```bash
AKERNEL_SERVER_ADDRESS="6.2.179.37:8888" \
AKERNEL_TOKEN="<token>" \
CLAUDE_CODE_TOOL_IMAGE=swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest \
MODEL_PATH=~/models/Qwen3.5-9B \
bash examples/blackbox_recipes/claude_code/run_train.sh
```

The training YAML keeps `claude_code` as the only runner:

```yaml
agent_runner_fqn: examples.blackbox_recipes.claude_code.claude_code_runner.claude_code_runner
```

## 3. Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MAX_TURNS` | `100` | `claude --max-turns` (the agent's turn budget); read by the runner from the `AGENT_MAX_TURNS` env var |
| `SWE_AGENT_EVAL_TIMEOUT` | `600` | Reward evaluation timeout (seconds) |
| `SWE_AGENT_RUN_TIMEOUT` | `7200` | Max wall time for the claude process in the sandbox |
| `CLAUDE_CODE_TOOL_IMAGE` | `swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest` | Sidecar tool image |
| `CONDA_ENV` | `testbed` | Conda env activated inside the sandbox before running claude |

`AGENT_MAX_TURNS` is the only knob that bounds the agent. The trainer's
`multi_turn.max_assistant_turns` is not enforced on the blackbox rollout path
(`AgentFrameworkRolloutAdapter`) — claude runs to its own `--max-turns` inside
the sandbox and the gateway counts the turns afterward — so it is not exposed as
a separate knob. A value of `1` would cripple the agent, hence the default `100`.
