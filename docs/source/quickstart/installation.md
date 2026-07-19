# Installation

Start with the environment that matches your workflow, then add the task and sandbox dependencies you need.

- **Non-training workflows:** a standard Python 3.10+ environment.
- **RL training:** a compatible `verl` Docker image with the training dependencies.

## Install Uni-Agent

Clone the repository and enter its directory:

```bash
git clone https://github.com/verl-project/uni-agent.git
cd uni-agent
```

For RL training, install the bundled `verl` source into your training environment:

```bash
git submodule update --init --recursive
pip install --no-deps -e ./verl
```

## Optional Dependencies

The additional dependencies introduced by Uni-Agent mainly fall into two categories: task dependencies and sandbox dependencies. Most are lightweight and can be installed on demand.

### Task Dependencies

Task dependencies provide task-specific datasets, verifiers, and reward implementations. Each task guide lists its own requirements; packages for tasks like SWE-Bench are not required by unrelated tasks.

For example, install the SWE-Bench package only when running a SWE-Bench task:


=== "SWE-Bench"

    ```bash
    pip install swebench
    ```


### Sandbox Dependencies

Install the client package for the sandbox backend you plan to use:

=== "Local"

    No additional sandbox SDK is required.

=== "Modal"

    ```bash
    pip install modal
    ```

=== "veFaaS"

    ```bash
    pip install volcengine-python-sdk swe-rex
    ```

## Ray Runtime Environments

For workloads running on Ray worker nodes, use a Ray Runtime Environment to ship the repository, expose the bundled `verl` source, and install additional task or sandbox dependencies on every node.

```yaml
working_dir: ./
excludes: ["/.git/"]
pip:
  packages:
    - "volcengine-python-sdk"
    - "swe-rex"
    - "swebench"
env_vars:
  PYTHONPATH: "verl"
  # ......
```

Pass the file when submitting the Ray job:

```bash
ray job submit --runtime-env runtime_env.yaml -- python entrypoint.py
```

Next, [launch a sandbox and run some code](launch-sandbox.md).
