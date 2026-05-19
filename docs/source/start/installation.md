# Installation

Uni-Agent can run directly on top of the standard `verl` training environment. You can start from an existing `verl` setup or an official `verl` Docker image, and then install a small set of additional dependencies required by Uni-Agent.

---

## Base Image

Start from one of the following:

- an existing `verl` training environment that is already working
- an official `verl` Docker image that matches your rollout backend

---

## Install veRL

Uni-Agent depends on `verl` as its training engine and is regularly updated to track the latest `verl` branch.

Choose the setup path that matches how you plan to run Uni-Agent:

### Single-Node Trial

For a local single-node debug trial, install `verl` directly in the current Python environment:

```bash
git submodule update --init --recursive
pip install --no-deps -e ./verl
```

Then install any task-specific optional dependencies you need. For example:

```bash
pip install swe-rex loguru pydantic pydantic_settings
```

### Ray Submit Jobs

For jobs submitted to a Ray cluster, keep the base image aligned with the `verl` stack and use Ray Runtime Env for task-specific Python packages and environment variables:

```yaml
working_dir: ./
excludes:
  - "/.git/"
pip:
  - swe-rex
  - loguru
  - pydantic
  - pydantic_settings
env_vars:
  PYTHONPATH: "verl"
  TORCH_NCCL_AVOID_RECORD_STREAMS: "1"
  CUDA_DEVICE_MAX_CONNECTIONS: "1"
  VLLM_DISABLE_COMPILE_CACHE: "1"

  # If you use veFaaS sandbox deployment
  VEFAAS_FUNCTION_ID: "xxx"
  VEFAAS_FUNCTION_ROUTE: "xxx"
  VOLCE_ACCESS_KEY: "xxx"
  VOLCE_SECRET_KEY: "xxx"

  # If you use Modal sandbox deployment
  MODAL_TOKEN_ID: "xxx"
  MODAL_TOKEN_SECRET: "xxx"
```

Save this file as a runtime environment YAML, for example `examples/agent_interaction/runtime_env.yaml`. Then submit your job with `ray job submit`:

```bash
ray job submit --runtime-env runtime_env.yaml -- python3 xxx.py
```

---

## Extra Dependencies

Uni-Agent keeps the base setup minimal. Install additional packages only for the sandbox backend, dataset, or evaluation workflow you plan to use.

**Sandbox Backends:**

```bash
# If you use Modal as the sandbox backend:
pip install modal

# If you use veFaaS as the sandbox backend:
pip install volcengine-python-sdk
```

**Datasets and Evaluation:**

```bash
# If you use swebench
pip install --no-cache-dir swebench

# If you use R2E-GYM
git clone https://github.com/R2E-Gym/R2E-Gym.git /home/R2E-Gym
cd /home/R2E-Gym
pip install --no-cache-dir --no-deps -e .
```
