# Launch an Agent Environment

Long-horizon agent tasks, such as software engineering, need a **persistent sandbox** where the agent can run commands, install packages, edit files, and preserve state across many steps. This document shows how to start an agent environment, install tools inside it, and run a simple persistence demo.

The runnable example lives under `examples/agent_env`; the final section shows how to run it end to end.

---

## Start a Sandbox

The first step is to start a sandbox. Uni-Agent supports local sandboxes and remote cloud sandbox backends.

- Use **local deployment** for quick debugging on a machine with container runtime permissions.
- Use **veFaaS deployment** when you run sandboxes on Volcengine FaaS (best for CN clusters).
- Use **Modal deployment** when you run sandboxes on Modal (available globally).

> **Note:** Some environments do not grant permission to start local containers or virtualized sandboxes. If these runtimes are restricted, use a cloud sandbox backend such as veFaaS or Modal.

Each subsection below focuses only on the deployment config needed to start the sandbox. Tool installation and the end-to-end demo are covered later.

Each deployment config is passed through `AgentEnvConfig` and then used to start an `AgentEnv`:

```python
import uuid
from uni_agent.interaction import AgentEnv, AgentEnvConfig

run_id = str(uuid.uuid4())
env_config = AgentEnvConfig(**{
    "deployment": {
        "type": "<local|vefaas|modal>",
        # backend-specific fields go here
    },
    "env_variables": {
        "PIP_PROGRESS_BAR": "off",
    },
})
env = AgentEnv(run_id=run_id, env_config=env_config)
env.start()
```

### Local deployment

Local deployment starts a sandbox on the current machine, then connects to the `swerex` server inside that sandbox. This is the easiest way to debug environment behavior before moving to a remote backend.

The local backend prefers Apptainer or Singularity when available. Docker and Podman are also supported and can be selected explicitly with `container_runtime` or discovered from `PATH` when Apptainer/Singularity are not installed.

**Dependencies.** Install the runtime package and make sure a supported runtime is available. For Apptainer:

```bash
pip install swe-rex
apptainer --version
```

If your runtime is not on `PATH`, set `LOCAL_CONTAINER_RUNTIME` or `UNI_AGENT_CONTAINER_RUNTIME` to the binary path. When no explicit `container_runtime` is provided, the local backend checks these environment variables, then discovers `apptainer`, `singularity`, `docker`, or `podman` from `PATH`.

**Config.** Use `type: "local"` and provide an image plus a command that starts `swerex.server` inside the sandbox:

```python
import os

env_config = AgentEnvConfig(**{
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
})
```

- **`type`** must be `"local"`.
- **`image`** is the sandbox image. For Apptainer, plain image names such as `python:3.12` are treated as Docker/OCI images and run as `docker://python:3.12`.
- **`command`** runs inside the sandbox and should start `swerex.server`. It can use `{token}` and `{port}` placeholders.
- **`container_runtime`** can be set to an Apptainer/Singularity binary path, `docker`, or `podman`.
- **`published_port`** optionally pins the localhost port used by the `swerex` server.
- **`extra_run_args`** can pass additional runtime flags. For example, Apptainer bind mounts or GPU flags must appear before the image argument.
- **`network`** is Docker/Podman-specific and useful when the current process is itself running inside Docker.

Apptainer launches the server with host networking, so the selected port is passed directly to `swerex.server`. Docker and Podman keep using port publishing.

Useful local overrides:

```bash
export LOCAL_CONTAINER_RUNTIME=/opt/apptainer/bin/apptainer
export LOCAL_DEPLOYMENT_IMAGE=python:3.12
export LOCAL_DEPLOYMENT_EXTRA_ARGS="--bind /data:/data"
```

### veFaaS deployment

veFaaS is a Volcengine FaaS platform. For workloads with many concurrent runs, it is often more stable and scales better than self-hosted local instances.

Follow the [Volcengine tutorial](https://www.volcengine.com/docs/6662/2278468?lang=zh) to obtain the required veFaaS configuration parameters, complete the cloud sandbox setup, and verify connectivity.

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

- `VOLCE_ACCESS_KEY` or `VOLCENGINE_ACCESS_KEY`: Volcengine access key.
- `VOLCE_SECRET_KEY` or `VOLCENGINE_SECRET_KEY`: Volcengine secret key.
- `VEFAAS_FUNCTION_ID`: veFaaS function ID.
- `VEFAAS_FUNCTION_ROUTE`: veFaaS function route.
- `VEFAAS_REGION`: optional region, defaulting to `cn-beijing`.

**Config.** Use `type: "vefaas"` and pass the veFaaS function settings:

```python
import os

env_config = AgentEnvConfig(**{
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
})
```

- **`image`** is the sandbox Docker image.
- **`command`** is the startup command. It must start `swerex.server` or install and start it through the veFaaS bootstrap script.
- **`function_id`** and **`function_route`** identify your veFaaS function.
- **`timeout`** and **`startup_timeout`** are specified in seconds.

### Modal deployment

Modal deployment starts the sandbox on Modal and exposes the `swerex` runtime through a Modal encrypted port. It is useful when local container permissions are unavailable or when you want a managed remote sandbox without maintaining your own cluster.

**Dependencies.** Install the Modal client and the runtime package:

```bash
pip install modal swe-rex boto3
```

**Environment variables.** Set your Modal credentials before starting the sandbox:

```bash
export MODAL_TOKEN_ID=xxxxxxxxxx
export MODAL_TOKEN_SECRET=xxxxxxxxxx
```

**Config.** Use `type: "modal"`. Modal starts `swerex` automatically, so you only need to provide the image and timeout settings:

```python
env_config = AgentEnvConfig(**{
    "deployment": {
        "type": "modal",
        "image": "python:3.12",
        "startup_timeout": 600.0,
        "runtime_timeout": 300.0,
        "deployment_timeout": 3600.0,
    },
    "env_variables": {
        "PIP_PROGRESS_BAR": "off",
    }
})
```

- **`image`** can be a public registry image such as `python:3.12` or a local Dockerfile path.
- **`startup_timeout`** controls how long Uni-Agent waits for the `swerex` runtime to become reachable.
- **`runtime_timeout`** controls per-operation runtime requests.
- **`deployment_timeout`** controls the Modal sandbox lifetime.
- **`modal_sandbox_kwargs`** can pass additional keyword arguments to `modal.Sandbox.create`.
- **`install_pipx`** defaults to `true`, so Modal can start `swerex` with `pipx` if it is not already installed in the image.

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

Uni-Agent provides common tool implementations for tasks such as running bash commands and editing files. You can also customize and integrate your own tools.

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

You can run the full demo from the repo root with the deployment backend you want to test:

```bash
DEPLOYMENT=<local|vefaas|modal> DEBUG_MODE=1 python examples/agent_env/demo.py
```

Set `DEBUG_MODE=1` when you want to print sandbox startup and runtime output in the current terminal.
