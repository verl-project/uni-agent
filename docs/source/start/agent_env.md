# Launch an Agent Environment

Long-horizon agent tasks, such as software engineering, need a **persistent sandbox**: a place where the agent can run commands, modify the environment, and preserve state across many steps. This document shows how to start a sandbox, install tools, and run a simple script.

The runnable code referenced in this document lives under `examples/agent_env`. You can run the example with:

```bash
DEPLOYMENT=<local|vefaas> DEBUG_MODE=1 python examples/agent_env/demo.py
```

Setting `DEBUG_MODE=1` prints all intermediate startup and runtime output to the current terminal.

---

## Start a Sandbox

The first step is to start a sandbox. You can do this either locally or on a remote FaaS platform.

### Local deployment

Local deployment starts a sandbox on the current machine, then connects to the `swerex` server inside that sandbox. This is the easiest way to debug environment behavior before moving to a remote platform.

The local backend prefers Apptainer or Singularity when available. Docker and Podman are also supported and can be selected explicitly with `container_runtime` or discovered from `PATH` when Apptainer/Singularity are not installed.

**Dependencies.** Install the runtime package and make sure a supported runtime is available. For Apptainer:

```bash
pip install swe-rex
apptainer --version
```

If your runtime is not on `PATH`, set `LOCAL_CONTAINER_RUNTIME` or `UNI_AGENT_CONTAINER_RUNTIME` to the binary path. When no explicit `container_runtime` is provided, the local backend checks these environment variables, then discovers `apptainer`, `singularity`, `docker`, or `podman` from `PATH`.

**Config and start.** Build the config, create the environment, and start it:

```python
import os
import uuid
from uni_agent.interaction import AgentEnv, AgentEnvConfig

run_id = str(uuid.uuid4())
env_config = {
    "deployment": {
        "type": "local",
        "image": os.getenv("LOCAL_DEPLOYMENT_IMAGE", "python:3.12"),
        "command": (
            "python3 -m pip install -q swe-rex && "
            "python3 -m swerex.server --host 0.0.0.0 --port {port} --auth-token {token}"
        ),
        "timeout": 300.0,
        "startup_timeout": 180.0,
    },
    "env_variables": {
        "PIP_PROGRESS_BAR": "off",
    }
}
env_config = AgentEnvConfig(**env_config)
env = AgentEnv(run_id=run_id, env_config=env_config)
env.start()
```

- **`type`** must be `"local"`.
- **`image`** is the sandbox image. For Apptainer, plain image names such as `python:3.12` are treated as Docker/OCI images and run as `docker://python:3.12`.
- **`command`** runs inside the sandbox and should start `swerex.server`. It can use `{token}` and `{port}` placeholders.
- **`container_runtime`** can be set to an Apptainer/Singularity binary path, `docker`, or `podman`.
- **`published_port`** optionally pins the localhost port used by the `swerex` server.
- **`extra_run_args`** can pass additional runtime flags. For example, Apptainer bind mounts or GPU flags must appear before the image argument.
- **`network`** is Docker/Podman-specific and useful when the current process is itself running inside Docker.

Apptainer launches the server with host networking, so the selected port is passed directly to `swerex.server`. Docker and Podman keep using port publishing.

You can run the full demo from the repo root with:

```bash
DEPLOYMENT=local DEBUG_MODE=1 python examples/agent_env/demo.py
```

Useful local overrides:

```bash
export LOCAL_CONTAINER_RUNTIME=/opt/apptainer/bin/apptainer
export LOCAL_DEPLOYMENT_IMAGE=python:3.12
export LOCAL_DEPLOYMENT_EXTRA_ARGS="--bind /data:/data"
```

### Remote deployment (VEFAAS)

VEFAAS is a Volcengine FaaS platform. For workloads with many concurrent runs, it is often more stable and scales better than self-hosted local instances.

You can directly refer to [this tutorial](https://www.volcengine.com/docs/6662/2278468?lang=zh) to obtain the required configuration parameters for the integration with the veFaaS cloud sandbox, complete the environment setup, and verify connectivity.

**Dependencies.** Install the required packages:

```bash
pip install volcengine-python-sdk swe-rex
```

**Environment variables.** Set your credentials and optional function settings in the environment.

```bash
export VOLCE_ACCESS_KEY=xxxxxxxxxx
export VOLCE_SECRET_KEY=xxxxxxxxxx
export VEFAAS_FUNCTION_ID=xxxxxxxxxx
export VEFAAS_FUNCTION_ROUTE=xxxxxxxxxx
```

| Variable | Description |
|----------|-------------|
| `VOLCE_ACCESS_KEY` or `VOLCENGINE_ACCESS_KEY` | Volcengine access key |
| `VOLCE_SECRET_KEY` or `VOLCENGINE_SECRET_KEY` | Volcengine secret key |
| `VEFAAS_REGION` | Region (optional, default `cn-beijing`) |

**Config and start.** Build the config, create the environment, and start it:

```python
import os
import uuid
from uni_agent.interaction import AgentEnv, AgentEnvConfig

run_id = str(uuid.uuid4())
env_config = {
    "deployment": {
        "type": "vefaas",
        "image": "enterprise-public-2-cn-beijing.cr.volces.com/vefaas-public/python:3.12",
        "command": "curl -fsSL https://vefaas-swe.tos-cn-beijing.ivolces.com/swe-rex/install_1.4.0.sh | bash -s -- {token}",
        "timeout": 300.0,
        "startup_timeout": 180.0,
        "function_id": os.getenv("VEFAAS_FUNCTION_ID"),
        "function_route": os.getenv("VEFAAS_FUNCTION_ROUTE"),
    },
    "env_variables": {
        "PIP_PROGRESS_BAR": "off",
    }
}
env_config = AgentEnvConfig(**env_config)
env = AgentEnv(run_id=run_id, env_config=env_config)
env.start()
```

- **`run_id`**: A unique identifier for the run, used in logging and deployment.
- **`deployment`**: `type` must be `"vefaas"`.
  - `image` is the sandbox Docker image.
  - `command` is the startup command.
  - `timeout` and `startup_timeout` are specified in seconds.
  - `function_id` and `function_route` identify your VEFAAS function.
- **`env_variables`**: Environment variables exported into the sandbox shell after startup.

---

## Install Tools

Once the sandbox is running, install tools so the agent can execute bash commands and edit files:

```python
from uni_agent.tools import ToolConfig

tools_config = [
    {"name": "execute_bash"},
    {"name": "str_replace_editor"},
]
tools = [ToolConfig(**tool_config).get_tool() for tool_config in tools_config]
env.install_tools(tools)
```

We provide common tool implementations for tasks such as running bash commands and editing files. You can also customize and integrate your own tools.

You can verify that the tools were installed successfully by running:

```python
print(env.communicate("which str_replace_editor"))
```

This command returns:
```python
/usr/local/bin/str_replace_editor
```
This indicates that the installation succeeded.

---

## Run the Demo

The demo runs a few simple steps to show sandbox persistence: install a dependency, create a script, execute it, and read the output.

**1. Install numpy**

```python
env.communicate("pip install numpy -q")
```

The dependency is installed in the current sandbox and persists for the rest of the run.

**2. Create a script and write its output to a file**

Create a small Python script with `str_replace_editor`, then run it and redirect its stdout to a file:

```python
import shlex
_script = "import numpy as np; print(np.array([1,2,3]).sum())"
env.communicate(f"str_replace_editor create --path /tmp/demo.py --file_text {shlex.quote(_script)}")
env.communicate("execute_bash 'python3 /tmp/demo.py > /tmp/demo_out.txt'")
```

**3. View the result**

```python
print(env.communicate("cat /tmp/demo_out.txt"))  # -> 6
```

**4. Close the environment**

```python
env.close()
```

After setting your credentials, you can run the full demo from the repo root with:

```bash
python examples/agent_env/demo.py
```

## Agent Environment Quick Reference

| Step | Code |
|------|------|
| Config & start | `AgentEnvConfig(**env_config)` → `AgentEnv(...)` → `env.start()` |
| Tools | `env.install_tools(tools)` |
| Run command | `env.communicate("...")` |
| Close | `env.close()` |
