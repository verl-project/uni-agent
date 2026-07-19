# Launch a Sandbox

Uni-Agent sandboxes provide a persistent execution environment for tools. The agent or driver stays on the host, while shell commands and file operations are routed through the selected sandbox backend.

This guide uses [`examples/quickstart/sandbox/demo.py`](https://github.com/verl-project/uni-agent/blob/main/examples/quickstart/sandbox/demo.py) to install a package, create and edit a Python script, execute it, and verify that state persists across tool calls.

## What the Demo Does

The demo walks through one sandbox lifecycle:

1. Build a sandbox from `SandboxConfig`.
2. Bind a stateful shell and file editor through `Toolbox`.
3. Install NumPy inside the sandbox.
4. Create and run `/tmp/demo.py`.
5. Edit the script from `sum` to `product` and run it again.
6. Change the shell working directory and verify that a later call keeps it.

## Run in an Isolated Sandbox

Modal is the default provider in the demo. Install its client, authenticate, and run:

```bash
pip install pydantic modal
modal token set
DEBUG_MODE=1 python examples/quickstart/sandbox/demo.py
```

The demo launches a `python:3.12` sandbox by default. Use another Python image with `IMAGE`:

```bash
DEBUG_MODE=1 IMAGE=python:3.11 python examples/quickstart/sandbox/demo.py
```

The driver remains on your host. NumPy, the script, and its output are created inside the remote sandbox.

## Run Locally

Use the local provider to inspect the API without remote credentials:

```bash
pip install pydantic
DEBUG_MODE=1 SANDBOX_PROVIDER=local python examples/quickstart/sandbox/demo.py
```

!!! warning "The local provider is not isolated"
    Commands run directly on your host, and the demo installs NumPy into the active Python environment. Use Modal or another remote backend when you need isolation.

## Build the Sandbox

The provider and image are selected through `SandboxConfig`:

```python
config = SandboxConfig(
    provider="modal",
    image="python:3.12",
    runtime_timeout=3600,
)
sandbox = build_sandbox(config)
```

Sandbox providers are loaded lazily, so you only need to install the SDK for the backend you use.

## Attach Tools

The demo binds two tools to the same sandbox:

```python
tool_specs = [
    {"name": "stateful_shell", "command_timeout": 120},
    {"name": "str_replace_editor"},
]

toolbox = Toolbox.from_specs(tool_specs, sandbox=sandbox)
```

The model-facing tool names are exposed through `toolbox.schemas()`. Calls are routed to the sandbox:

```python
result = await toolbox.call("shell", {"command": "python3 /tmp/demo.py"})
print(result)
```

## Understand Persistence

All tools share the same sandbox lifecycle and filesystem:

- Packages installed by one shell call are available to later calls.
- Files created by the editor can be executed and read by the shell.
- The stateful shell preserves its working directory between calls.
- The editor itself is stateless; persistence comes from the shared sandbox filesystem.

Both `Sandbox` and `Toolbox` use async context managers. When the context exits, tools are closed and remote sandbox resources are released automatically.

Next, [run agent inference](agent-inference.md) against a sandbox-backed task.
