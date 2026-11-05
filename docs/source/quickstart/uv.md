# Manage the environment with uv

Uni-Agent and its pinned `verl` submodule share one uv workspace and one root
`uv.lock`. The commands below target Python 3.12 on Linux x86_64/aarch64,
following the pinned verl's UV backend policy. Uni-Agent's existing Python
package requirement (`>=3.10,<3.13`) is unchanged. SGLang 0.5.20 wheels require
glibc >=2.34. The lock resolves all backends, but one environment installs only
the selected, compatible combination.

```bash
git submodule update --init --recursive
uv sync --locked --all-packages --extra vllm --extra fsdp
uv run --locked --all-packages --extra vllm --extra fsdp \
  python -c 'import uni_agent, verl, ray; print(uni_agent.__file__, verl.__file__)'
```

## Available extras

Select extras with repeated `--extra NAME` flags. Keep `--all-packages` on
sync/run commands so extras declared by the `verl` workspace member are selected
there as well. The tables below match the pinned submodule, not verl's latest
upstream revision.

### Backend selection (declared by verl)

| Extra | Purpose |
| --- | --- |
| `vllm` | GPU rollout/inference with vLLM, including the core verl dependencies. |
| `sglang` | GPU rollout/inference with SGLang, including the core verl dependencies. |
| `fsdp` | GPU training with FSDP, including native training kernels. |
| `megatron` | GPU training with Megatron-Core/Bridge, Transformer Engine and native kernels. |
| `cpu` | CPU test/development environment; does not provide a GPU training environment. |

For GPU training, normally select one inference backend (`vllm` or `sglang`)
and one training backend (`fsdp` or `megatron`). `vllm` and `sglang` conflict;
`cpu` conflicts with both inference backends and both GPU training extras.
Do not use `--all-extras`.

### Uni-Agent features

| Extra | Purpose |
| --- | --- |
| `vefaas` | veFaaS sandbox client: `volcengine-python-sdk` and `swe-rex`. |
| `modal` | Modal sandbox client. |
| `e2b` | E2B sandbox client. |
| `swebench` | SWE-bench dataset/evaluation utilities. |
| `insight` | RL-Insight client. Does not start a server or enable telemetry automatically. |

Select the sandbox extra matching your task configuration; installing a client
does not configure credentials or create the sandbox service. For example, the
GPU ReAct training quickstarts use `vefaas` by default. For RL-Insight setup,
see the [integration guide](../concepts/rl-insight-integration.md).

```bash
# vLLM + Megatron training, veFaaS sandbox, SWE-bench and RL-Insight client.
uv sync --locked --all-packages --extra vllm --extra megatron \
  --extra vefaas --extra swebench --extra insight
# Use the same extras on subsequent uv run commands.
```

### Other extras declared by the pinned verl

| Extra | Purpose |
| --- | --- |
| `test` | Upstream verl test/debug tools. |
| `math` | Math verification utility (`math-verify`). |
| `ci` | Upstream CI helpers, including `hf-transfer` and `sglang-router`. |
| `verl-core` | Shared verl runtime dependencies; included by the main backend extras. |
| `flash-attn-cu130torch213` | FlashAttention for the pinned CUDA/PyTorch stack; included by `fsdp`/`megatron`. |
| `cupy-cu130` | CUDA 13 CuPy; included by `fsdp`/`megatron`. |
| `veomni-sft` | Upstream VeOmni SFT integration. |
| `torchtitan` | Upstream TorchTitan integration. |

These names are available through the workspace; they are not all validated
Uni-Agent training combinations. GPU CI covers `vllm` + `megatron` + `swebench`;
`veomni-sft` and `torchtitan` do not have Uni-Agent GPU validation in this PR.
CPU level0 tests remain in `tests-cpu`. GPU CI runs both the GPU level0 UV
checks and the original Router E2E in
`tests/uni_agent/agent_aware_router/e2e/test_router_e2e.py`. The latter retains
its GPU level1 marker and original assertions; a dedicated CI step selects it
explicitly. CI prepares its Qwen3-4B-Instruct-2507 model and four SWE-bench
Verified rows using the existing dataset preprocessor. The original test uses
real vLLM/router/gateway/TQ components with its simulated sandbox runner;
it does not validate an external sandbox deployment or model training.
Verbose logs show case names and outcomes, and both steps are required.

`dev` is a **dependency group**, not an extra. It contains Uni-Agent's test/lint
tools and is installed by default; pass `--no-dev` for runtime-only environments.

## GPU training examples

The GPU quickstarts `examples/quickstart/training/train_qwen3_moe.sh` and
`train_qwen3p5_dense.sh` use the plain Ray CLI to submit jobs and a locked UV
command for the remote Python entrypoint. The submitting machine only needs a
compatible Ray CLI, not the training GPU dependencies. Ray supports propagating
UV through [`py_executable`](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#using-uv-for-package-management),
but this pinned verl disables Ray's automatic UV hook in `verl/__init__.py`.
The scripts therefore explicitly set
`ray_kwargs.ray_init.runtime_env.py_executable` to the same locked UV command
plus `python`, ensuring the worker environment matches the remote driver.
The scripts select `megatron`, `swebench`, the `ROLLOUT_NAME` backend (default
`vllm`), and `UV_SANDBOX_EXTRA` (default `vefaas`, matching the default task).

Run from the repository root after preparing the model/data and Ray cluster:

```bash
cp examples/quickstart/training/runtime_env_uv.yaml ~/runtime_env_uv.yaml
# Add your sandbox credentials to the private copy.
export DATA_DIR=/path/to/data RUNTIME_DIR=/path/to/output
export RUNTIME_ENV=~/runtime_env_uv.yaml
bash examples/quickstart/training/train_qwen3_moe.sh
# Or: bash examples/quickstart/training/train_qwen3p5_dense.sh
```

Every cluster node needs `uv`, CUDA support and access to the selected wheels.
Start Ray using the same locked backend/sandbox extras and Python 3.12, for example
`uv run --locked --all-packages --python 3.12 --extra vllm --extra megatron --extra swebench --extra vefaas ray start --head`;
workers must use matching Ray/Python versions. A custom `RUNTIME_ENV` may supply
sandbox credentials and tuning variables, but must not define `pip`, `conda`,
`uv` or `py_executable`: dependencies come from `uv.lock`. Select another sandbox
with `UV_SANDBOX_EXTRA=modal` or `e2b` alongside its corresponding task config.
Other recipes retain their existing launch environments.

## Updating the verl dependency

Backend dependencies, native-wheel sources and CPU/GPU conflicts stay in
`verl/pyproject.toml`. uv natively references the submodule through:

```toml
[tool.uv.workspace]
members = ["verl"]

[tool.uv.sources]
verl = { workspace = true }
```

This references the dependency; it does not inherit all of its UV configuration.
For example, uv reads [`override-dependencies`](https://docs.astral.sh/uv/reference/settings/#override-dependencies)
only from the workspace root and ignores declarations in members.

`scripts/refresh_uv.py` is a maintenance helper, not a UV requirement for using
workspace dependencies. The root settings could be maintained by hand; this
repository generates them from the pinned verl configuration to avoid drift.
Normal `uv sync --locked` installs use the checked-in files and do not run the
script. After changing the verl submodule pin, regenerate the root policy:

```bash
python3.12 scripts/refresh_uv.py
uv lock --python 3.12
python3.12 scripts/refresh_uv.py --check
uv lock --check --python 3.12
```

The generated section of `pyproject.toml` copies the pinned verl's environment,
override, dependency-metadata and index settings. It does not duplicate backend
requirements or alter files in the submodule. Commit the submodule pin,
generated policy and root lockfile together.
