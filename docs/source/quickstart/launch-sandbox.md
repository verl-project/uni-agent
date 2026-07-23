# Launch a Sandbox

Uni-Agent provides an isolated and persistent execution environment for agents. Files, installed packages, and runtime state remain available across interactions within the same sandbox session.

This guide focuses on the ReAct Agent Tool workflow. It uses [`examples/quickstart/sandbox/demo.py`](https://github.com/verl-project/uni-agent/blob/main/examples/quickstart/sandbox/demo.py) to install a package, create and edit a Python script, execute it, and verify that state persists across tool calls.

```bash
DEBUG_MODE=1 SANDBOX_PROVIDER=xxx python examples/quickstart/sandbox/demo.py
```

!!! note "About `demo.py` and `DEBUG_MODE`"
    You can use this `demo.py` to run an end-to-end sandbox check.

    Set `DEBUG_MODE=1` to display the per-session `INFO` walkthrough in the console.

## Start a Sandbox

### Choose a Sandbox Backend

Uni-Agent supports multiple sandbox backends. Choose the backend that matches your workflow. Providers are loaded lazily, so only the selected backend SDK needs to be installed.

=== "Local"

    **Local, non-isolated.** Commands run directly on the host.

    !!! warning "Local execution can modify your machine"
        Commands can read, modify, or delete local files and change the active Python environment. Use this backend only for small-scale validation with trusted commands.

    ```python
    from uni_agent.sandbox import SandboxConfig

    config = SandboxConfig(provider="local")
    ```

=== "Docker"

    **Local, isolated.** [Docker](https://www.docker.com/) runs agents in containers on your machine.

    Install Docker and make sure its daemon is running. By default, Docker pulls the image when it is not already available locally.

    ```python
    from uni_agent.sandbox import SandboxConfig

    config = SandboxConfig(
        provider="docker",
        image="python:3.12",
        sandbox_kwargs={
            # "missing" (default), "always", or "never".
            "pull_policy": "missing",
            # Optional arguments inserted before the image in `docker run`.
            "run_args": ["--network", "none"],
        },
    )
    ```

    The provider starts an ephemeral container, executes commands with `docker exec`,
    transfers files with `docker cp`, and removes the container when the sandbox exits.
    The default container command is `sleep infinity`; images without `sleep` can override
    `entrypoint` and `command` in `sandbox_kwargs`. Run `docker login <registry>` first when
    pulling from a private registry.

=== "veFaaS"

    **Remote cloud service.** [veFaaS](https://www.volcengine.com/product/vefaas) provides elastic, isolated sandboxes on Volcengine.

    Install its client dependencies:

    ```bash
    pip install volcengine-python-sdk swe-rex
    ```

    Configure the veFaaS endpoint and Volcengine credentials. You can follow the [Volcengine tutorial](https://www.volcengine.com/docs/6662/2278468?lang=zh) to obtain the required configuration parameters.

    ```bash
    export VEFAAS_FUNCTION_ID="<function-id>"
    export VEFAAS_FUNCTION_ROUTE="<function-route>"
    export VOLCE_ACCESS_KEY="<access-key>"
    export VOLCE_SECRET_KEY="<secret-key>"
    ```

    Create the sandbox configuration:

    ```python
    from uni_agent.sandbox import SandboxConfig

    config = SandboxConfig(
        provider="vefaas",
        image="python:3.12",
        runtime_timeout=3600,
        sandbox_kwargs={"startup_timeout": 120},
    )
    ```

=== "Modal"

    **Remote cloud service.** [Modal](https://modal.com/) provides on-demand isolated sandboxes without requiring you to manage a cluster.

    Install and authenticate the Modal client:

    ```bash
    pip install modal
    modal token set
    ```

    Alternatively, you can provide the Modal credentials through environment variables:

    ```bash
    export MODAL_TOKEN_ID="<token-id>"
    export MODAL_TOKEN_SECRET="<token-secret>"
    ```

    Configure the image, lifecycle timeout, and optional Modal app name:

    ```python
    from uni_agent.sandbox import SandboxConfig

    config = SandboxConfig(
        provider="modal",
        image="python:3.12",
        runtime_timeout=3600,
        sandbox_kwargs={"app_name": "agent-sandbox"},
    )
    ```

=== "YuanRong"

    **Remote or self-hosted service.** [YuanRong](https://docs.openyuanrong.org/zh-cn/latest/index.html) provides elastic sandbox management for distributed agent workloads.

    Integration and configuration instructions are to be filled.

### Start and Stop the Sandbox

Build the selected provider from its configuration:

```python
from uni_agent.sandbox import build_sandbox

sandbox = build_sandbox(config)
```

Entering the async context starts the selected backend. Exiting it releases remote resources:

```python
async with sandbox:
    result = await sandbox.exec_shell("...")
    ...
```

## Build a ReAct Toolbox

A ReAct agent interacts with the sandbox through tools. The demo binds a stateful shell and a file editor:

```python
from uni_agent.tools import Toolbox

tool_specs = [
    {"name": "stateful_shell", "command_timeout": 120},
    {"name": "str_replace_editor"},
]

toolbox = Toolbox.from_specs(tool_specs, sandbox=sandbox)
```

Enter the toolbox and export the model-facing tool schemas:

```python
async with toolbox.entered(retry=3, timeout=60):
    schemas = toolbox.schemas()
```

`stateful_shell` is exposed to the model as `shell`. Tools route their operations through the sandbox backend.

## Run Tool Calls

Use the shell tool to install dependencies and execute commands:

```python
result = await toolbox.call(
    "shell",
    {"command": "pip install -q numpy && echo installed"},
)
```

Use the editor to create a Python script inside the sandbox:

```python
script = "import numpy as np\nprint(int(np.array([1, 2, 4]).sum()))\n"

await toolbox.call(
    "str_replace_editor",
    {
        "command": "create",
        "path": "/tmp/demo.py",
        "file_text": script,
    },
)
```

The shell can immediately run the file created by the editor:

```python
result = await toolbox.call(
    "shell",
    {"command": "python3 /tmp/demo.py"},
)
print(result)
```

## Verify Persistence

All tools share the sandbox lifecycle and filesystem:

- Packages installed by one call remain available to later calls.
- Files written by the editor can be executed and read by the shell.
- The stateful shell preserves environment variables and its working directory.
- The editor is stateless; its files persist because they live in the shared sandbox.

For example, a later shell call still sees a previous `cd`:

```python
await toolbox.call("shell", {"command": "cd /tmp"})
result = await toolbox.call("shell", {"command": "pwd; python3 demo.py"})
```

Both `Sandbox` and `Toolbox` use async context managers, so tools are closed and remote resources are released even if a call fails.

## Run the Complete Demo

After configuring a supported backend above, run the complete connectivity and persistence check:

=== "Local"

    ```bash
    DEBUG_MODE=1 SANDBOX_PROVIDER=local python examples/quickstart/sandbox/demo.py
    ```

=== "Docker"

    ```bash
    DEBUG_MODE=1 SANDBOX_PROVIDER=docker python examples/quickstart/sandbox/demo.py
    ```

=== "veFaaS"

    ```bash
    DEBUG_MODE=1 SANDBOX_PROVIDER=vefaas python examples/quickstart/sandbox/demo.py
    ```

=== "Modal"

    ```bash
    DEBUG_MODE=1 SANDBOX_PROVIDER=modal python examples/quickstart/sandbox/demo.py
    ```

Next, you can [run agent inference](agent-inference.md) against a sandbox-backed task.
