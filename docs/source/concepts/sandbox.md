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

- A timeout becomes `ExecResult(exit_code=-1)`.
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

`gke_agent_sandbox` runs isolated container episodes on Google Kubernetes Engine (or any Kubernetes cluster with the [`kubernetes-sigs/agent-sandbox`](https://github.com/kubernetes-sigs/agent-sandbox) CRDs installed) using `agent_sandbox_rl` warm pools and the Kubernetes `pods/exec` API.

```yaml
sandbox:
  provider: gke_agent_sandbox
  runtime_timeout: 7200
  image_map:
    - from: "swebench/**:latest"
      to: "<region>-docker.pkg.dev/<project>/<repo>/**:latest"
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

Key behaviors and options:

- **Network isolation (loopback-only by default)**: Templates default to `networkPolicyManagement: Managed` with `networkPolicy: {ingress: [], egress: []}` (deny-all ingress and egress; only pod loopback and Kubernetes API `pods/exec` remain usable). To use a single operator-managed namespace-wide `NetworkPolicy` instead (for large GKE Dataplane V2 / Cilium clusters), set `sandbox_kwargs.network_policy_management: "Unmanaged"` (or `SANDBOX_NETWORK_POLICY_MANAGEMENT=Unmanaged`); startup fails closed unless an egress-restricting `NetworkPolicy` selecting `app=agent-sandbox-rl` exists in the namespace.
- **Overall lifetime limit (`runtime_timeout`)**: Enforced both in-process (expiring the sandbox, clamping per-command timeouts, and releasing the `SandboxClaim`) and on-cluster via a best-effort `SandboxClaim.spec.lifecycle.shutdownTime` (`shutdownPolicy: Delete`).
- **Image rewriting**: Configure `sandbox.image_map` in Task Config (or set `SANDBOX_IMAGE_MAP`, e.g. `org/**=mirror.example.com/repo/**`).
- **Optional pod annotations**: Pass cluster-specific pod annotations via `sandbox_kwargs.pod_annotations` or `SANDBOX_POD_ANNOTATIONS` (JSON object). No cluster-specific annotations are added by default.

See `examples/quickstart/training/task_config_react_gke.yaml` for a full ReAct training config.

To [verify oracle solutions](../quickstart/oracle-verification.md) on this provider, select it with `SANDBOX_PROVIDER` and give the run its own ID:

```bash
export SANDBOX_PROVIDER=gke_agent_sandbox
export SANDBOX_RUN_ID="oracle-$(date +%m%d-%H%M%S)"
export SANDBOX_MAX_WARMPOOL_SIZE=1  # each oracle sample claims one sandbox for its image
python examples/inference/parallel_run_oracle.py \
    --data-path ~/data/uni_agent/r2e_gym_subset.parquet \
    --task-config examples/quickstart/oracle/task_config_oracle.yaml \
    --num-workers 8 \
    --concurrency 64
```

The oracle runner does not tear the warm pools down when it exits. Remove them afterwards with the reaper from `agent-sandbox-rl`:

```bash
python -m agent_sandbox_rl.reaper --namespace "${SANDBOX_NAMESPACE:-default}" --run-id "${SANDBOX_RUN_ID}"
```

Sandboxes have no network access by default, so oracle samples whose tests need DNS or outside hosts report as unsolved.

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
