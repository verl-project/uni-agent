# Sandbox

A Sandbox is the execution boundary of an agent episode. It owns the runtime lifecycle and provides a common data plane for commands, files, uploads, downloads, and optional port exposure.

Tools depend on the narrow `SandboxBackend` protocol. Tasks and Agents receive the full `Sandbox` object, but the Task remains responsible for starting and stopping it.

## Configuration

`SandboxConfig` selects the provider and carries common lifecycle settings:

```python
from uni_agent.sandbox import SandboxConfig, build_sandbox

config = SandboxConfig(
    provider="modal",
    image="python:3.12",
    runtime_timeout=3600,
    sandbox_kwargs={"app_name": "agent-sandbox"},
)

sandbox = build_sandbox(config)
```

The standard fields are:

- `provider`: registered backend name.
- `image`: container image used by image-backed providers such as Docker and Modal.
- `runtime_timeout`: maximum remote sandbox lifetime.
- `sandbox_kwargs`: provider-specific constructor arguments.

Unknown fields are rejected. Put provider-specific options inside `sandbox_kwargs`.

## Lifecycle

Use a Sandbox as an async context manager:

```python
async with sandbox:
    result = await sandbox.exec_shell("python --version")
```

Entering the context starts the backend with retry, timeout, and global startup-concurrency controls. Exiting the context calls `stop()` even when the episode raises.

The shared lifecycle uses:

- `SANDBOX_STARTUP_TIMEOUT`: startup timeout, 600 seconds by default.
- `SANDBOX_STARTUP_CONCURRENCY`: process-wide startup limit, 64 by default.

Remote providers should implement `is_alive()` as a non-throwing health probe.

## Data Plane

`SandboxBackend` exposes:

```python
await sandbox.exec(argv, timeout=..., workdir=..., env=...)
await sandbox.exec_shell(script, timeout=..., workdir=..., env=...)
await sandbox.read_file(path)
await sandbox.write_file(path, content)
await sandbox.upload(local_path, remote_path)
await sandbox.download(remote_path, local_path)
await sandbox.expose_port(port)
```

The base class provides portable command, file, and directory-transfer implementations. Providers may override file methods when they have a native data plane. For example, Local uses the host filesystem directly, Docker uses `docker cp`, and Modal uses its filesystem API for absolute paths.

## Error Policy

All providers share the public `Sandbox.exec()` error policy:

- A timeout becomes `ExecResult(exit_code=-1)`.
- A normal command/provider error while the sandbox is alive becomes `exit_code=127`.
- An infrastructure error after the sandbox has died is re-raised.

This policy prevents one bad agent command from crashing an otherwise healthy episode while still surfacing backend failures.

!!! warning "Do not override `exec()`"
    Custom providers implement `_exec()`. The public `exec()` wrapper owns the cross-provider error policy and is enforced by tests.

## Custom Backend

A minimal backend implements `start()`, `stop()`, and `_exec()`:

```python
from uni_agent.sandbox.base import ExecResult, Sandbox
from uni_agent.sandbox.registry import register_sandbox


@register_sandbox("my_backend")
class MySandbox(Sandbox):
    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        ...
```

If the backend needs custom configuration mapping, override `from_config()`:

```python
@classmethod
def from_config(cls, config: SandboxConfig):
    return cls(
        image=config.image,
        runtime_timeout=config.runtime_timeout,
        **config.sandbox_kwargs,
    )
```

Register the lazy module in `SANDBOX_MODULES`:

```python
SANDBOX_MODULES["my_backend"] = "my_package.sandbox"
```

The module is imported only when the provider is selected, so optional SDKs do not affect other backends.

## Implementation Rules

- Implement `_exec()`, not `exec()`.
- Make `start()` and `stop()` safe to call during partial failures.
- Override `is_alive()` for remote providers.
- Extend `_is_timeout_error()` when the provider SDK uses custom timeout exceptions.
- Preserve `stdout`, `stderr`, and exit codes in `ExecResult`.
- Prefer native file APIs when available; otherwise inherit the base transfer implementation.
- Add tests for lifecycle cleanup, timeout classification, dead-backend errors, environment forwarding, and binary file transfer.
