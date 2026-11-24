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
- `image_map`: optional rewrite from the dataset image name to the image your cluster can pull (see below).
- `image_mounts`: optional runtime images mounted into the task sandbox.
- `executable_paths`: command names mapped to executables supplied by the task or mounted images.
- `runtime_timeout`: maximum sandbox lifetime.
- `sandbox_kwargs`: provider-specific constructor arguments.

Unknown fields are rejected. Put provider-specific options inside `sandbox_kwargs`.

### `image_map`

SWE-Bench samples ship with public image names such as `swebench/sweb.eval.x86_64.astropy_1776_astropy-13033`. If your cluster cannot pull those names (private registry, mirror, or a local tag), add `image_map` under `sandbox` in Task Config. Do not edit the parquet.

```yaml
sandbox:
  provider: vefaas
  image_map:
    - from: "swebench/**:latest"
      to: "<your-registry>/swe-bench-verified/**:v2"
    - from: "swerebench/**:latest"
      to: "<your-registry>/swe-rebench/**:latest"
```

`**` copies the instance-specific path, so `swebench/sweb.eval.x86_64.astropy_1776_astropy-13033` becomes `<your-registry>/swe-bench-verified/sweb.eval.x86_64.astropy_1776_astropy-13033:v2`.

List as many rules as you need; the first matching `from` is used. Images that match none of the rules are left unchanged.

### Runtime images

Black-box agent harnesses can be packaged separately from task images and attached with:

```yaml
sandbox:
  provider: modal
  image: swebench/sweb.eval.x86_64.example
  image_mounts:
    - image: docker.io/example/claude-code-runtime:1.0
      mount_path: /opt/agent-runtime
  executable_paths:
    claude: /opt/agent-runtime/bin/claude
```

Docker, Modal, OpenJiuwen agent-dx and OpenYuanRong support `image_mounts`. `image_map` rewrites only the task `image`, so mounted image references must already be complete and pullable. `executable_paths` exposes selected mounted commands after startup; Local and veFaaS reject it.

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
- `SANDBOX_STOP_TIMEOUT`: timeout for the cleanup `stop()` after a failed or cancelled start, 120 seconds by default. Cancellation waits for this cleanup, so the bound keeps a hung provider from blocking it forever.

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

The base class provides portable command, file, and directory-transfer implementations. Providers may override file methods when they have a native data plane. For example, Local uses the host filesystem directly, Docker uses `docker cp` for file upload/download and stdin for `write_file`, and Modal uses its filesystem API for absolute paths.

Docker `write_file` sends raw bytes through `docker exec -i`, avoiding host command-argument size limits. The task image needs `bash`, `mkdir`, `dirname`, and `cat`. Writes overwrite in place (including through symlinks), preserve existing file permissions, and are not atomic. If a write is cancelled, its Docker client is killed and reaped; the task owner must still call `stop()` to guarantee container-side cleanup. A cancelled or failed write may leave a partial file.

To run the focused real-Docker regressions on Linux with 4 KiB pages:

```bash
docker pull ubuntu:24.04
UNI_AGENT_DOCKER_WRITE_IMAGE=ubuntu:24.04 PYTHONPATH=. python -m pytest -q \
  tests/uni_agent/sandbox/test_docker_write_integration.py
```

These checks require the normal test dependencies, including `pytest`, `pydantic`, and `jsonschema`; they run containers with networking disabled and remove them on teardown.

## Error Policy

All providers share the public `Sandbox.exec()` error policy:

- A timeout becomes `ExecResult(exit_code=-1, timed_out=True)`; check `timed_out`, since a command can legitimately exit with a negative code.
- A normal command/provider error while the sandbox is alive becomes `exit_code=127`.
- An infrastructure error after the sandbox has died is re-raised.
- A provider's `_is_fatal_exec_error()` hook can preserve infrastructure errors even while the runtime is alive. Agent DX uses this for unknown command outcomes, lost results, and truncated output.

This policy prevents one bad agent command from crashing an otherwise healthy episode while still surfacing backend failures.

!!! warning "Do not override `exec()`"
    Custom providers implement `_exec()`. The public `exec()` wrapper owns the cross-provider error policy and is enforced by tests.

## Gateway Reverse Tunnel

Some sandbox providers support a Gateway reverse tunnel and declare this with `supports_reverse_tunnel = True`. With `sandbox_kwargs.proxy_port`, the task runner supplies the Gateway Session's upstream and rewrites the model URL to the in-sandbox HTTP proxy, preserving the Session path. Providers without this capability reject the configuration. A provider class can be inspected without installing its SDK or creating a remote environment.

Each supporting provider owns its upstream format through the `Sandbox.tunnel_upstream(base_url)` classmethod. The default requires an explicit `host:port`; a provider can override it to preserve its required URL scheme, supply default ports, or handle IPv6 addresses. The task runner delegates this formatting to the selected provider.

## GKE Agent Sandbox (`gke_agent_sandbox`)

`gke_agent_sandbox` runs each sandbox as a pod on Google Kubernetes Engine, or on any Kubernetes cluster with the [`kubernetes-sigs/agent-sandbox`](https://github.com/kubernetes-sigs/agent-sandbox) controller and CRDs installed. Commands run through the Kubernetes `pods/exec` API, so the image needs `sh` and `timeout`.

```yaml
sandbox:
  provider: gke_agent_sandbox
  image: python:3.12
  runtime_timeout: 7200
  sandbox_kwargs:
    namespace: default
    node_selector:
      cloud.google.com/gke-nodepool: sandbox-cpu-pool
```

This provider needs Python 3.11 or 3.12, because both client packages below require Python 3.11 or newer.

Install the client dependencies (since `agent-sandbox-rl` is not on PyPI yet, install both client packages from `kubernetes-sigs/agent-sandbox`):

```bash
pip install -e '.[gke]'  # from a uni-agent checkout
pip install 'git+https://github.com/kubernetes-sigs/agent-sandbox.git@9afee7039342f18fa9ad402929cabf8b66eebaa7#subdirectory=clients/python/agentic-sandbox-client'
pip install 'git+https://github.com/kubernetes-sigs/agent-sandbox.git@9afee7039342f18fa9ad402929cabf8b66eebaa7#subdirectory=examples/agent-sandbox-rl'
```

The process needs Kubernetes credentials (in-cluster service account or kubeconfig) that may create `SandboxTemplate`, `SandboxWarmPool` and `SandboxClaim` objects and use `pods/exec` in the namespace.

Options under `sandbox_kwargs`:

| Option | Default | Meaning |
| --- | --- | --- |
| `namespace` | `SANDBOX_NAMESPACE`, then the process's own namespace, then `default` | Namespace the sandbox pods run in. |
| `run_id` | `SANDBOX_RUN_ID`, else a random ID per process | Sandboxes with the same run ID share warm pools across processes. |
| `node_selector` | `NODE_SELECTOR_KEY` / `NODE_SELECTOR_VAL` | Node labels the pods require. A matching `NoSchedule` toleration is added for each label, so a node pool tainted with its own label is usable. |
| `network_policy_management` | `Managed` | `Managed`: the controller creates a NetworkPolicy per image. `Unmanaged`: it creates none. |
| `network_policy` | deny all ingress and egress | The NetworkPolicy rules for `Managed` mode. |

The three environment fallbacks exist because those values change per run, which a Task Config file cannot express.

**Network isolation.** Sandboxes have no network access by default; only the pod's loopback and `pods/exec` work. To allow traffic, pass `network_policy` rules. `Unmanaged` mode is for clusters where one namespace-wide policy is preferable to one per image (large GKE Dataplane V2 clusters): the provider then refuses to start unless a NetworkPolicy in the namespace restricts egress for the sandbox pods (label `app=agent-sandbox-rl`).

**Lifetime.** A sandbox is stopped when `runtime_timeout` elapses, and the controller is asked to delete its claim at the same deadline in case the process has died.

**Shell.** The `stateful_shell` tool uses a built-in shell session that needs only `bash`, because an isolated sandbox cannot download tmux.

### Warm pools

A sandbox starts without any preparation: the first one for an image creates the pod on demand and waits for the image pull. To avoid that wait, keep sandboxes started ahead of time:

```python
from uni_agent.sandbox.gke import GKEAgentSandbox

# image -> number of ready sandboxes to keep; returns image -> ready
ready = GKEAgentSandbox.prewarm({"python:3.12": 16}, run_id="my-run", timeout=900)
...
GKEAgentSandbox.unwarm(["python:3.12"], run_id="my-run")
```

Both calls take the same `sandbox_kwargs` as the sandboxes that will use the pools, and both block. Sandboxes in other processes use the pools when they share the `run_id`.

### Cleanup

When a process exits it releases its sandboxes and deletes the pools it created. A process that is killed cannot do that; remove what it left with the reaper from `agent-sandbox-rl`, using `--in-cluster` when running inside the cluster:

```bash
python -m agent_sandbox_rl.reaper --namespace <namespace> --run-id <run_id>
```

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
        image_mounts=config.image_mounts,
        executable_paths=config.executable_paths,
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
