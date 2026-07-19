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

The additional dependencies introduced by Uni-Agent mainly include task dependencies and sandbox dependencies. Most are lightweight and can be installed on demand.

### Task Dependencies

Task dependencies provide task-specific datasets, verifiers, and reward implementations. For example, install the `swebench` package only when running a SWE-Bench task:


=== "SWE-Bench"

    ```bash
    pip install swebench
    ```


### Sandbox Dependencies

Install the client package for the sandbox backend you plan to use, for example:

=== "Modal"

    ```bash
    pip install modal
    ```

=== "veFaaS"

    ```bash
    pip install volcengine-python-sdk swe-rex
    ```

## Ray Runtime Environments

For distributed Ray workloads, you can use a Runtime Environment to inject the required task, sandbox, and `verl` dependencies into every worker node.

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

Next, you can [launch a sandbox and run some code](launch-sandbox.md).
