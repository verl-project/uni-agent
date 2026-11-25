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
    The CPU-only workflows in this guide do not require a local inference
    service; point agents at an external OpenAI-compatible endpoint or a remote
    Gateway. On ARM hosts with a supported NVIDIA GPU, you can build vLLM from
    source and serve models locally. Follow vLLM's source-build instructions
    for your ARM64 platform, GPU, and CUDA environment.

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

Do **not** run `pip install -r requirements-test.txt` for this CPU-only setup:
it pins `vllm==0.23.0`, which is unnecessary for the vLLM-independent tests
covered here and may require a source build on ARM. Install the test
dependencies selectively instead (next section).

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
and `tests/uni_agent/agent_aware_router/`) require vLLM. If it is not installed,
they fail with `ModuleNotFoundError: No module named 'vllm'`; skip them
explicitly when running the full directory:

```bash
.venv/bin/python -m pytest tests/uni_agent/ -m 'cpu and level0' -q \
  --ignore=tests/uni_agent/gateway \
  --ignore=tests/uni_agent/agent_aware_router
```

Before opening a pull request, run the same checks as CI:

```bash
.venv/bin/pre-commit run --all-files --show-diff-on-failure
```

## Run the Sandbox Quickstart

```bash
DEBUG_MODE=1 SANDBOX_PROVIDER=docker .venv/bin/python examples/quickstart/sandbox/demo.py
```

The demo finishes with `Demo done`. On a fresh `python:3.12` image,
`stateful_shell` installs `tmux` through apt on first use; see the
troubleshooting notes below if your network makes that slow.

## Troubleshooting

The default flows above use the official package sources. Use the options
below only when you hit the corresponding problem.

### PyPI is unreachable or slow

Switch to a nearby mirror such as Tsinghua's:

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Or set `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` per command
instead of changing global config.

### Docker Hub is unreachable or slow

Configure a registry mirror in `/etc/docker/daemon.json` and restart Docker:

```json
{"registry-mirrors": ["https://docker.m.daocloud.io"]}
```

```bash
systemctl restart docker
docker pull python:3.12
```

### GitHub clone/fetch or pre-commit hooks fail

pre-commit downloads its hook environments from GitHub on first use. On hosts
with unstable GitHub access, retry the operation or route git/pip traffic
through an HTTP proxy:

```bash
export http_proxy=http://<proxy-host>:<port>
export https_proxy=http://<proxy-host>:<port>
.venv/bin/pre-commit run --all-files --show-diff-on-failure
```

Once the hook environments are cached, pre-commit works offline.

### apt is slow inside task images

Sandbox tools may install packages inside task images (for example,
`stateful_shell` installs `tmux` on first use). Debian-based images default to
`deb.debian.org`; switch to a nearby mirror first:

```bash
# deb822 format (Debian trixie and newer)
sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources
# legacy format (bookworm and older)
sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
```

### stateful_shell startup is slow or times out on first use

The first start installs `tmux` in the container. If an attempt times out,
later retries wait for the in-flight installer and converge instead of
failing on the apt lock. To avoid the install entirely, pre-install tmux into
a base image and select it with `IMAGE=`:

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

### `image_mounts` fails on Docker < 26

`image_mounts` (mounting sidecar tool images) requires **Docker >= 26** with
the containerd image store enabled. On older Docker versions, build a combined
task image instead and reference it as the sandbox `image`:

```dockerfile
FROM python:3.12
COPY --from=<tool-image>:<tag> / /opt/<tool>
```

Keep the same `executable_paths` as the mount-based configuration.

### Tests skip or behave differently when run as root

Some sandbox tests exercise permission errors that do not occur for root
(for example, writing to a read-only file). They skip with a message such as
`root can write read-only files`; this is expected and not a failure. For
closer parity with CI, run the suite as an unprivileged user.

### `make` is missing for the docs build

Minimal ARM images may not ship `make`. Either install it
(`dnf install -y make`) or invoke MkDocs directly from the repository root:

```bash
.venv/bin/pip install -r docs/requirements.txt
.venv/bin/mkdocs build
```

### Sidecar image builds need a proxy or pip mirror

Tool images based on `python-build-standalone` download a Python tarball from
GitHub releases and packages from PyPI. Use the recipe's pip-index build
argument and, when needed, Docker's built-in proxy build arguments:

```bash
docker build --network host \
  --build-arg http_proxy=http://127.0.0.1:<port> \
  --build-arg https_proxy=http://127.0.0.1:<port> \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -f Dockerfile.<tool>-tool -t <tool-image>:<tag> .
```

`--network host` lets build steps reach a proxy listening on the host's
loopback address.

## Verify Your Setup

- `.venv/bin/python -c 'import uni_agent'` succeeds
- `pytest tests/uni_agent/sandbox tests/uni_agent/tools -q` passes
- The sandbox demo prints `Demo done`

You are ready to develop on ARM. Changes to agents, tools, sandboxes, and
tasks can all be validated locally; see
[Launch a Sandbox](launch-sandbox.md) and
[Run Agent Inference](agent-inference.md) for the next steps.
