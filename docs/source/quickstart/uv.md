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

Use `sglang` instead of `vllm`, and/or `megatron` instead of `fsdp`. Use
`--extra cpu` on its own for verl's CPU test/development slice. The backend
extras belong to the verl workspace member: keep `--all-packages` on sync/run
commands so they are selected there. Do not use `--all-extras`, because CPU/GPU
and inference-engine choices conflict.

Optional Uni-Agent features are additive: `--extra modal`, `--extra e2b`,
`--extra vefaas`, `--extra swebench`, and `--extra insight`. The latter installs
the RL-Insight client; it does not start its server or enable telemetry. Follow
the [RL-Insight integration guide](../concepts/rl-insight-integration.md) for
service setup and environment variables. Development tools are in the default
`dev` group; pass `--no-dev` for runtime-only environments.

## GPU training examples

The GPU quickstarts `examples/quickstart/training/train_qwen3_moe.sh` and
`train_qwen3p5_dense.sh` use locked UV commands for both `ray job submit` and
its remote Python entrypoint. Ray propagates the driver's UV command to its
workers, as described in [Ray's UV integration](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#using-uv-for-package-management).
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
