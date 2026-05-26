# Installation

Uni-Agent can run directly on top of the standard `verl` training environment. In practice, this means you can start from an existing `verl` setup or an official `verl` Docker image, and then install a small set of additional dependencies required by Uni-Agent.

This is the recommended setup for both large-scale inference and agent RL training, because Uni-Agent reuses `verl` for the training/runtime stack rather than replacing it.

---

## Base Image

Start from one of the following:

- an existing `verl` training environment that is already working
- an official `verl` Docker image that matches your rollout backend, such as vLLM or SGLang

Uni-Agent is currently based on the `verl 0.7.1` release. We recommend using the corresponding `verl` dependencies or Docker images for that release, together with the matching rollout backend stack:

- `vLLM 0.17.0` for vLLM-based rollouts
- `SGLang 0.5.9` for SGLang-based rollouts

---

## Install `verl`

Install `verl 0.7.1` first. Uni-Agent uses `verl` as a normal Python package dependency. We recommend using an editable install when working from this repo, because it makes it easier to keep the `verl` package environment synchronized across multi-node training setups.

Recommended: use the bundled `verl` checkout/submodule in this repo:

```bash
git submodule update --init --recursive
pip install --no-deps -e ./verl
```

Alternatively, you can install from PyPI:

```bash
pip install verl==0.7.1
```

The editable install is especially useful when you want Uni-Agent and `verl` to live in the same workspace during development or keep `verl` aligned across distributed training environments.

---

## Extra Dependencies

On top of the base `verl` environment, Uni-Agent typically needs the following Python packages:

```bash
pip install --no-cache-dir swe-rex loguru pydantic pydantic_settings
pip install --no-cache-dir --upgrade aiohttp
```

These packages are used for:

- `swe-rex`: persistent sandbox runtime used by Uni-Agent environments
- `loguru`: structured logging used by Uni-Agent
- `pydantic` and `pydantic_settings`: config models and settings management
- `aiohttp`: upgraded for compatibility with the runtime stack

---

## Optional Dependencies By Task

Different tasks need different extra packages. The simplest way to think about it is by example:

If you want to use VEFAAS as the remote environment backend, install the Volcengine Python SDK:

```bash
pip install --no-cache-dir volcengine-python-sdk
```

If you want to run SWE-Bench interaction, verification, or reward evaluation, install:

```bash
pip install --no-cache-dir swebench
```

If you want to train or evaluate on R2E-Gym, install `R2E-Gym` from source:

```bash
git clone https://github.com/R2E-Gym/R2E-Gym.git /home/R2E-Gym
cd /home/R2E-Gym
pip install --no-cache-dir --no-deps -e .
```

In some containerized setups, Git may complain about repository ownership. If that happens, mark the repo as safe:

```bash
git config --system --add safe.directory /home/R2E-Gym
```

---

## Example: Derived Docker Image

Below is the logical diff of a Uni-Agent-ready image on top of a `verl` base image:

```dockerfile
FROM <your-verl-base-image>

RUN pip install --no-cache-dir swe-rex loguru pydantic pydantic_settings
RUN pip install --no-cache-dir --upgrade aiohttp

# Optional: VEFAAS
RUN pip install --no-cache-dir volcengine-python-sdk

# Optional: SWE-Bench
RUN pip install --no-cache-dir swebench

# Optional: R2E-Gym
RUN git clone https://github.com/R2E-Gym/R2E-Gym.git /home/R2E-Gym
WORKDIR /home/R2E-Gym
RUN pip install --no-cache-dir --no-deps -e .
RUN git config --system --add safe.directory /home/R2E-Gym
```
