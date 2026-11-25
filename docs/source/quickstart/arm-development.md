# Develop on ARM Hosts

This guide covers developing Uni-Agent on **aarch64 (ARM64)** Linux hosts —
building agents, tools, and sandbox integrations, and running the CPU test
suites — without a GPU. Verified on openEuler 24.03 (aarch64) with Docker
25.0.3 and Python 3.11.

For RL training, use a GPU or NPU host instead; see
[Run Agent RL Training](rl-training.md) and
[Run Training on NPU](npu-training.md).

## Requirements

- An aarch64 Linux host (openEuler 24.03 verified; other ARM64 distributions
  work similarly)
- Python 3.10–3.12 (`requires-python`)
- Docker with a running daemon for sandbox workflows
- No GPU is needed for development, sandbox workflows, or CPU tests

!!! note "Model inference on ARM hosts"
    vLLM does not provide aarch64 CPU wheels, so do not expect to serve models
    locally. Point agents at an external OpenAI-compatible endpoint or a remote
    Gateway; all development and test workflows below work without a local
    inference service.

## Configure Mirrors (Recommended on Slow or CN Networks)

### pip index

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Or set `PIP_INDEX_URL` per command/CI job instead of changing global config.

### Docker registry mirrors

```json
// /etc/docker/daemon.json
{"registry-mirrors": ["https://docker.m.daocloud.io"]}
```

```bash
systemctl restart docker
docker pull python:3.12
```

### In-container apt mirrors

Sandbox tools may install packages inside task images (for example,
`stateful_shell` installs `tmux` on first use). Debian-based images default to
`deb.debian.org`, which can be slow; switch to a nearby mirror first:

```bash
# deb822 format (Debian trixie and newer)
sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources
# legacy format (bookworm and older)
sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
```

## Install Uni-Agent

```bash
git clone https://github.com/verl-project/uni-agent.git
cd uni-agent
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/pip install -e .
```

Install the lightweight packages used by the sandbox and tool flows:

```bash
.venv/bin/pip install pydantic jsonschema
```

If you need the bundled `verl` source (for example for the framework tests),
install it without its heavy dependencies:

```bash
.venv/bin/pip install --no-deps -e ./verl
```

Do **not** run `pip install -r requirements-test.txt` on an ARM CPU host: it
pins `vllm==0.23.0`, which is unavailable there. Install the test dependencies
selectively instead (next section).

## Run CPU Tests

Start with the core suites, which need only a small dependency set:

```bash
.venv/bin/pip install pytest pytest-asyncio
.venv/bin/python -m pytest tests/uni_agent/sandbox tests/uni_agent/tools -q
```

The wider `cpu and level0` suite additionally uses these packages (versions
aligned with the bundled `verl`):

```bash
.venv/bin/pip install \
  fastapi uvicorn cachetools loguru pyzmq aiohttp orjson omegaconf \
  'tensordict>=0.8.0,<=0.10.0' 'transformers>=5.5.3,<5.13' \
  'ray==2.54.1' 'TransferQueue==0.1.11' 'swebench==4.1.0'
.venv/bin/python -m pytest tests/uni_agent/ -m 'cpu and level0' -q
```

Tests that exercise vLLM parser integrations (under `tests/uni_agent/gateway/`
and `tests/uni_agent/agent_aware_router/`) require vLLM and are expected to
fail with `ModuleNotFoundError: No module named 'vllm'` on ARM CPU hosts. Skip
them explicitly when running the full directory:

```bash
.venv/bin/python -m pytest tests/uni_agent/ -m 'cpu and level0' -q \
  --ignore=tests/uni_agent/gateway \
  --ignore=tests/uni_agent/agent_aware_router
```

Before opening a pull request, run the same checks as CI:

```bash
.venv/bin/pre-commit run --all-files --show-diff-on-failure
```

pre-commit downloads its hook environments from GitHub; configure a network
proxy first if your host cannot reach GitHub reliably.

## Run the Sandbox Quickstart

```bash
DEBUG_MODE=1 SANDBOX_PROVIDER=docker .venv/bin/python examples/quickstart/sandbox/demo.py
```

The demo finishes with `Demo done`. On a fresh `python:3.12` image,
`stateful_shell` installs `tmux` through apt on first use. On slow networks,
pre-install it to keep tool startup fast and deterministic:

```dockerfile
FROM python:3.12
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux \
    && rm -rf /var/lib/apt/lists/*
```

```bash
docker build -t uni-agent-arm-base:latest -f Dockerfile .
DEBUG_MODE=1 SANDBOX_PROVIDER=docker IMAGE=uni-agent-arm-base:latest \
  .venv/bin/python examples/quickstart/sandbox/demo.py
```

!!! note "Docker < 26: image mounts"
    `image_mounts` (mounting sidecar tool images) requires **Docker >= 26**
    with the containerd image store enabled. On older Docker versions, build a
    combined task image instead and reference it as the sandbox `image`:

    ```dockerfile
    FROM python:3.12
    COPY --from=<tool-image>:<tag> / /opt/<tool>
    ```

    Keep the same `executable_paths` as the mount-based configuration.

## Verify Your Setup

- `.venv/bin/python -c 'import uni_agent'` succeeds
- `pytest tests/uni_agent/sandbox tests/uni_agent/tools -q` passes
- The sandbox demo prints `Demo done`

You are ready to develop on ARM. Changes to agents, tools, sandboxes, and
tasks can all be validated locally; see
[Launch a Sandbox](launch-sandbox.md) and
[Run Agent Inference](agent-inference.md) for the next steps.
