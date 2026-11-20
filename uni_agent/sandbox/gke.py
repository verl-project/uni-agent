"""GKE sandbox provider: one sandbox is one pod claimed through the Kubernetes Agent Sandbox controller.

The cluster needs the ``kubernetes-sigs/agent-sandbox`` controller and CRDs. Pods come from per-image warm pools
managed by ``agent_sandbox_rl``: :meth:`GKEAgentSandbox.prewarm` fills a pool ahead of time, and a sandbox whose
image has no pool is cold-started on demand.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import math
import os
import shlex
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar

from .base import ExecResult, Sandbox
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import SandboxConfig

logger = logging.getLogger(__name__)

_NETWORK_POLICY_MODES = ("Managed", "Unmanaged")
_DENY_ALL_NETWORK_POLICY: dict[str, list[Any]] = {"ingress": [], "egress": []}
_DEFAULT_EXEC_TIMEOUT = 120.0
# How long after a command's own timeout the exec stream may stay open before the client gives up on it.
_EXEC_STREAM_GRACE = 10.0
# Tries to open the exec connection; only a connection that was never established is retried.
_EXEC_CONNECT_ATTEMPTS = 3
# A claim can wait minutes for a pod, so claims get their own threads and never starve command execution.
_CLAIM_WORKERS = 128
_EXEC_WORKERS = 256
# Threads the fleet uses to create or delete several warm pools at once.
_FLEET_WORKERS = 64


@functools.cache
def _load_sdk() -> SimpleNamespace:
    """Import the Kubernetes clients lazily so this provider stays importable without them."""
    try:
        import agent_sandbox_rl
        from agent_sandbox_rl.preflight import pod_selector_covers
        from kubernetes import client
        from kubernetes.stream import stream
    except ImportError as exc:
        raise ImportError(
            "the gke_agent_sandbox provider requires kubernetes, k8s-agent-sandbox and agent-sandbox-rl; "
            "see the GKE section of docs/source/concepts/sandbox.md for the install commands"
        ) from exc
    return SimpleNamespace(rl=agent_sandbox_rl, client=client, stream=stream, pod_selector_covers=pod_selector_covers)


@functools.cache
def _executor(name: str, workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"gke-sandbox-{name}")


def _in_cluster_namespace() -> str | None:
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


@dataclass(frozen=True)
class _FleetSettings:
    """Everything sandboxes must agree on to share warm pools; also the cache key of their fleet."""

    namespace: str
    run_id: str | None
    node_selector: tuple[tuple[str, str], ...]
    network_policy_management: str
    network_policy: str  # JSON, so the settings stay hashable

    @classmethod
    def resolve(
        cls,
        *,
        namespace: str | None = None,
        run_id: str | None = None,
        node_selector: Mapping[str, str] | None = None,
        network_policy_management: str = "Managed",
        network_policy: Mapping[str, Any] | None = None,
    ) -> _FleetSettings:
        """Fill unset values from the environment: these three change per run, which a Task Config file cannot."""
        if network_policy_management not in _NETWORK_POLICY_MODES:
            raise ValueError(
                f"network_policy_management must be one of {_NETWORK_POLICY_MODES}, got {network_policy_management!r}"
            )
        if network_policy is not None and network_policy_management != "Managed":
            raise ValueError("network_policy only applies when network_policy_management is 'Managed'")
        if node_selector is None:
            key, value = os.getenv("NODE_SELECTOR_KEY"), os.getenv("NODE_SELECTOR_VAL")
            node_selector = {key: value} if key and value else {}
        return cls(
            namespace=namespace or os.getenv("SANDBOX_NAMESPACE") or _in_cluster_namespace() or "default",
            run_id=run_id or os.getenv("SANDBOX_RUN_ID") or None,
            node_selector=tuple(sorted((str(k), str(v)) for k, v in node_selector.items())),
            network_policy_management=network_policy_management,
            network_policy=json.dumps(
                _DENY_ALL_NETWORK_POLICY if network_policy is None else dict(network_policy), sort_keys=True
            ),
        )


_FLEETS: dict[_FleetSettings, Any] = {}
_FLEETS_LOCK = threading.Lock()
# Images this process started a sandbox for without prewarming them, per fleet.
_COLD_STARTED: dict[_FleetSettings, set[str]] = {}


def _get_fleet(settings: _FleetSettings) -> Any:
    """Return the process-wide ``SandboxFleet`` for ``settings``, building it on first use."""
    with _FLEETS_LOCK:
        if settings not in _FLEETS:
            fleet = _build_fleet(settings)
            if not _FLEETS:
                # Not atexit: teardown() uses worker threads, which can no longer be started by the time atexit
                # handlers run. This hook runs earlier, alongside concurrent.futures' own.
                threading._register_atexit(_teardown_fleets)
            _FLEETS[settings] = fleet
        return _FLEETS[settings]


def _build_fleet(settings: _FleetSettings) -> Any:
    sdk = _load_sdk()
    node_selector = dict(settings.node_selector)
    # Start a pool's replicas on the same node where possible, so the image is pulled once.
    template: dict[str, Any] = {
        "network_policy_management": settings.network_policy_management,
        "colocate_replicas": True,
    }
    if settings.network_policy_management == "Managed":
        template["extra_template_spec"] = {"networkPolicy": json.loads(settings.network_policy)}
    if node_selector:
        # A dedicated sandbox node pool is usually tainted with its own label; tolerate that taint.
        template["extra_pod_spec"] = {
            "tolerations": [
                {"key": key, "operator": "Equal", "value": value, "effect": "NoSchedule"}
                for key, value in node_selector.items()
            ]
        }
    fleet = sdk.rl.SandboxFleet(
        sdk.rl.FleetConfig(
            clusters=[
                sdk.rl.ClusterConfig(
                    name="default",
                    namespace=settings.namespace,
                    node_selector=node_selector or None,
                    in_cluster="KUBERNETES_SERVICE_HOST" in os.environ,
                )
            ],
            max_concurrent=_FLEET_WORKERS,
            run_id=settings.run_id,
            # Put the run ID in pool and template names, so runs sharing a namespace never share or delete
            # each other's pools.
            run_isolation="names",
            template_name_prefix="uni-agent-",
            template=sdk.rl.TemplateSpec(**template),
        )
    )
    if settings.network_policy_management == "Unmanaged":
        _require_egress_policy(sdk, fleet.registry.get("default"))
    return fleet


def _require_egress_policy(sdk: SimpleNamespace, cluster: Any) -> None:
    """Refuse Unmanaged mode unless the namespace restricts egress for every sandbox pod.

    In this mode the controller creates no NetworkPolicy, so without one of the operator's the sandboxes would have
    full network access.
    """
    labels = cluster.resources.fleet_pod_labels()
    policies = sdk.client.NetworkingV1Api(cluster.api_client).list_namespaced_network_policy(cluster.namespace).items
    restricting = [
        policy.spec
        for policy in policies
        if sdk.pod_selector_covers(policy.spec.pod_selector, labels) and "Egress" in (policy.spec.policy_types or [])
    ]
    # Policies are additive: a single rule with neither ``to`` nor ``ports`` re-opens all egress.
    allows_all = any(not (rule.to or rule.ports) for spec in restricting for rule in spec.egress or [])
    if not restricting or allows_all:
        raise RuntimeError(
            f"network_policy_management='Unmanaged' needs a NetworkPolicy in namespace {cluster.namespace!r} that "
            f"restricts egress for pods labelled {labels}, and none that allows all egress for them. Apply one, or "
            "use the default 'Managed' mode."
        )


def _teardown_fleets() -> None:
    """At interpreter exit, release this process's sandboxes and delete the warm pools it created."""
    for settings, fleet in list(_FLEETS.items()):
        try:
            fleet.teardown()
            if settings.run_id:
                _delete_cold_start_pools(fleet, _COLD_STARTED.get(settings, ()))
        except Exception:
            logger.warning("GKE sandbox teardown failed; leftovers carry the run ID label", exc_info=True)


def _delete_cold_start_pools(fleet: Any, images: Iterable[str]) -> None:
    """Delete the pools that cold starts of ``images`` created under a shared run ID.

    With a shared run ID the fleet's own teardown only deletes pools this process prewarmed, since any other pool
    may belong to another process of the run. A pool still marked on-demand was never prewarmed by anyone, so it is
    safe to delete; one that a ``prewarm`` has taken over since is left to its owner.
    """
    constants = _load_sdk().rl.constants
    cluster = fleet.registry.get("default")
    for image in images:
        pool = cluster.resources.get_warmpool(fleet.config.pool_name(image)) or {}
        meta = pool.get("metadata") or {}
        if (meta.get("labels") or {}).get(constants.RUN_ID_LABEL) != fleet.run_id:
            continue
        if (meta.get("annotations") or {}).get(constants.ON_DEMAND_ANNOTATION) != "true":
            continue
        # Conditional on the uid just read, so an object re-created under the same name is left alone.
        cluster.resources.delete_warmpool(meta["name"], uid=meta["uid"])
        template = (cluster.resources.get_template(fleet.config.template_name(image)) or {}).get("metadata") or {}
        if (template.get("labels") or {}).get(constants.RUN_ID_LABEL) == fleet.run_id:
            cluster.resources.delete_template(template["name"], uid=template["uid"])


def _plan(fleet: Any, images: Iterable[str]) -> None:
    """Register images with the fleet, which only warms images it has planned a pool for."""
    tasks = [_load_sdk().rl.Task(id=image, image=image) for image in images]
    with _FLEETS_LOCK:
        if fleet.plan_ is None:
            fleet.load_tasks(tasks)
            fleet.plan()
        else:
            fleet.add_tasks(tasks)


def _claim(settings: _FleetSettings, image: str, runtime_timeout: float) -> tuple[Any, Any]:
    """Claim one sandbox pod for ``image`` and return ``(fleet, handle)``. Blocks until the pod is ready."""
    sdk = _load_sdk()
    fleet = _get_fleet(settings)
    if fleet.plan_ is not None and fleet.plan_.for_image(image) is not None:
        # This process prewarmed the image, so the claim addresses its pool by name. Recreate the pool if it has
        # been unwarmed since; a no-op while it exists. An unplanned image gets an on-demand pool from acquire().
        with _FLEETS_LOCK:
            fleet.warm_images([image], replicas_override=1, wait=False)
    else:
        with _FLEETS_LOCK:
            _COLD_STARTED.setdefault(settings, set()).add(image)
    handle = fleet.acquire(sdk.rl.Task(id=image, image=image))
    try:
        # Best effort: have the controller delete the claim at the deadline even if this process has died.
        cluster = fleet.registry.get(handle.cluster_name)
        deadline = datetime.now(timezone.utc) + timedelta(seconds=math.ceil(runtime_timeout))
        cluster.custom_api.patch_namespaced_custom_object(
            group=sdk.rl.constants.GROUP,
            version=sdk.rl.constants.VERSION,
            namespace=cluster.namespace,
            plural=sdk.rl.constants.CLAIMS_PLURAL,
            name=handle.claim_name,
            body={
                "spec": {
                    "lifecycle": {"shutdownTime": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"), "shutdownPolicy": "Delete"}
                }
            },
        )
    except Exception:
        logger.debug("could not set a shutdown time on claim %s", handle.claim_name, exc_info=True)
    return fleet, handle


def _release_abandoned_claim(claim: asyncio.Future[tuple[Any, Any]]) -> None:
    """Release a sandbox whose ``start()`` was cancelled while the claim was still in flight."""
    if not claim.cancelled() and claim.exception() is None:
        fleet, handle = claim.result()
        _executor("claim", _CLAIM_WORKERS).submit(fleet.release, handle)


def _exec_in_pod(
    fleet: Any,
    handle: Any,
    argv: list[str],
    timeout: float,
    workdir: str | None,
    env: dict[str, str] | None,
) -> ExecResult:
    """Run ``argv`` in the sandbox pod through the Kubernetes exec API. Needs ``sh`` and ``timeout`` in the image."""
    sdk = _load_sdk()
    cluster = fleet.registry.get(handle.cluster_name)
    steps = [f"export {key}={shlex.quote(str(value))}" for key, value in (env or {}).items()]
    if workdir:
        steps.append(f"cd {shlex.quote(workdir)}")
    steps.append("exec " + " ".join(shlex.quote(arg) for arg in argv))
    seconds = max(1, math.ceil(timeout))
    command = ["/bin/sh", "-c", f"timeout {seconds}s /bin/sh -c {shlex.quote(' && '.join(steps))}"]
    for attempt in range(_EXEC_CONNECT_ATTEMPTS):
        try:
            resp = sdk.stream(
                cluster.exec_core_api().connect_get_namespaced_pod_exec,
                handle.pod_name,
                cluster.namespace,
                command=command,
                container=handle.runtime_container,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _preload_content=False,
            )
            break
        except sdk.client.ApiException as exc:
            # Status 0 means the connection to the API server was never established (for example a connect
            # timeout under load), so the command has not started and it is safe to try again.
            if exc.status != 0 or attempt == _EXEC_CONNECT_ATTEMPTS - 1:
                raise
            time.sleep(attempt + 1)
    started = time.monotonic()
    deadline = started + seconds + _EXEC_STREAM_GRACE
    stdout: list[str] = []
    stderr: list[str] = []
    # The deadline cannot be left to the loop below alone: update() may not return while the stream keeps reporting
    # data (a command that never stops printing, or a connection that was dropped without a close frame), and a
    # thread spinning there slows every other sandbox of the process down. Closing the stream from a timer makes
    # update() fail, which ends the loop.
    watchdog = threading.Timer(deadline - started, resp.close)
    watchdog.daemon = True
    watchdog.start()
    try:
        while resp.is_open():
            if time.monotonic() > deadline:
                raise TimeoutError
            resp.update(timeout=1.0)
            if resp.peek_stdout():
                stdout.append(resp.read_stdout())
            if resp.peek_stderr():
                stderr.append(resp.read_stderr())
        if time.monotonic() > deadline:
            raise TimeoutError
        # Output that arrives together with the close frame stays buffered and never surfaces through the loop
        # above. Drain it, or large outputs (such as read_file's base64) come back truncated.
        channels = getattr(resp, "_channels", {})
        stdout.append(channels.pop(1, ""))
        stderr.append(channels.pop(2, ""))
        exit_code = int(resp.returncode or 0)
    except Exception as exc:
        if time.monotonic() > deadline:
            raise TimeoutError(f"exec exceeded {seconds}s in pod {handle.pod_name}") from exc
        raise
    finally:
        watchdog.cancel()
        with contextlib.suppress(Exception):
            resp.close()
    out, err = "".join(stdout), "".join(stderr)
    # ``timeout`` exits 124 when it kills the command, but a command may also exit 124 by itself. Only report a
    # timeout once the limit has actually passed, so an early 124 keeps its status and output.
    if exit_code == 124 and time.monotonic() - started >= seconds:
        raise TimeoutError(err or out or f"command timed out after {seconds}s")
    return ExecResult(exit_code=exit_code, stdout=out, stderr=err)


class _Shell:
    """A shell session that keeps its working directory and exported variables between commands.

    Sandboxes have no network access by default, so the generic tmux shell cannot install tmux in an image that
    lacks it. This one needs only ``bash``: each command restores the session from a state file and saves it back.
    """

    def __init__(self, sandbox: GKEAgentSandbox, *, cwd: str | None, env: dict[str, str] | None) -> None:
        self._sandbox = sandbox
        self._state = f"/tmp/.uni-agent-shell-{uuid.uuid4().hex[:8]}"
        lines = [f"export {key}={shlex.quote(str(value))}" for key, value in (env or {}).items()]
        if cwd:
            lines.append(f"cd -- {shlex.quote(cwd)}")
        self._initial_state: str | None = "\n".join(lines) + "\n"

    async def run(self, command: str, *, timeout: float | None = _DEFAULT_EXEC_TIMEOUT) -> ExecResult:
        if self._initial_state is not None:
            await self._sandbox.write_file(self._state, self._initial_state)
            self._initial_state = None
        script = (
            f". {self._state} 2>/dev/null\n"
            f"{command}\n"
            "__uni_agent_exit=$?\n"
            f"{{ export -p; printf 'cd -- %q\\n' \"$PWD\"; }} > {self._state} 2>/dev/null\n"
            "exit $__uni_agent_exit"
        )
        return await self._sandbox.exec(["bash", "-c", script], timeout=timeout)

    async def close(self) -> None:
        if self._initial_state is None:
            await self._sandbox.exec(["rm", "-f", self._state], timeout=10.0)


@register_sandbox("gke_agent_sandbox")
class GKEAgentSandbox(Sandbox):
    """Run each sandbox as a pod on GKE, or on any cluster with the Kubernetes Agent Sandbox controller.

    ``namespace``, ``run_id`` and ``node_selector`` fall back to the ``SANDBOX_NAMESPACE``, ``SANDBOX_RUN_ID`` and
    ``NODE_SELECTOR_KEY`` / ``NODE_SELECTOR_VAL`` environment variables. Sandboxes that share a ``run_id`` share
    warm pools across processes; without one, each process uses and cleans up its own.

    Pods have no network access by default: ``network_policy_management='Managed'`` gives every image a
    NetworkPolicy with the rules in ``network_policy`` (deny all when unset). ``'Unmanaged'`` creates none and
    requires a namespace policy that restricts the pods' egress.
    """

    supports_shell: ClassVar[bool] = True

    def __init__(
        self,
        *,
        image: str = "python:3.12",
        runtime_timeout: float = 3600.0,
        namespace: str | None = None,
        run_id: str | None = None,
        node_selector: Mapping[str, str] | None = None,
        network_policy_management: str = "Managed",
        network_policy: Mapping[str, Any] | None = None,
        executable_paths: dict[str, str] | None = None,
    ) -> None:
        if not math.isfinite(runtime_timeout) or runtime_timeout <= 0:
            raise ValueError("runtime_timeout must be finite and positive")
        self.image = image
        self.runtime_timeout = float(runtime_timeout)
        self.executable_paths = dict(executable_paths or {})
        self._settings = _FleetSettings.resolve(
            namespace=namespace,
            run_id=run_id,
            node_selector=node_selector,
            network_policy_management=network_policy_management,
            network_policy=network_policy,
        )
        self._fleet: Any = None
        self._handle: Any = None
        self._deadline = 0.0
        self._expired = False
        self._expiry: asyncio.Task[None] | None = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> GKEAgentSandbox:
        if config.image_mounts:
            raise ValueError("GKEAgentSandbox does not support image_mounts")
        return cls(
            image=config.image or "python:3.12",
            runtime_timeout=config.runtime_timeout,
            executable_paths=config.executable_paths,
            **config.sandbox_kwargs,
        )

    # ----- warm pools: optional, a sandbox also starts without one -----
    @classmethod
    def prewarm(
        cls,
        images: Mapping[str, int],
        *,
        wait: bool = True,
        timeout: float | None = None,
        **fleet_kwargs: Any,
    ) -> dict[str, bool]:
        """Keep sandboxes started ahead of time, so :meth:`start` does not wait for a pod.

        ``images`` maps each image to the number of ready sandboxes to keep; a pool that is already larger is left
        as it is. ``fleet_kwargs`` are the constructor's ``namespace``, ``run_id``, ``node_selector``,
        ``network_policy_management`` and ``network_policy``, and must match the sandboxes that will use the pools.

        With ``wait``, blocks for up to ``timeout`` seconds and returns ``image -> ready``. This call blocks: use a
        worker thread from async code.
        """
        if not images:
            return {}
        fleet = _get_fleet(_FleetSettings.resolve(**fleet_kwargs))
        _plan(fleet, images)
        by_replicas: dict[int, list[str]] = {}
        for image, replicas in images.items():
            by_replicas.setdefault(int(replicas), []).append(image)
        for replicas, group in by_replicas.items():
            fleet.warm_images(group, replicas_override=replicas, wait=False)
        return fleet.wait_pools_ready(list(images), timeout=timeout, replicas=dict(images)) if wait else {}

    @classmethod
    def unwarm(cls, images: Iterable[str], **fleet_kwargs: Any) -> None:
        """Delete the warm pools this process created with :meth:`prewarm` for ``images``.

        Call it once the sandboxes using those pools have stopped. Pools still present at interpreter exit are
        deleted then. This call blocks: use a worker thread from async code.
        """
        fleet = _get_fleet(_FleetSettings.resolve(**fleet_kwargs))
        if fleet.plan_ is not None:
            fleet.unwarm_images(list(images))

    # ----- control plane -----
    async def start(self) -> None:
        if self._handle is not None:
            return
        self._expired = False
        loop = asyncio.get_running_loop()
        claim = loop.run_in_executor(
            _executor("claim", _CLAIM_WORKERS), _claim, self._settings, self.image, self.runtime_timeout
        )
        try:
            self._fleet, self._handle = await asyncio.shield(claim)
        except asyncio.CancelledError:
            # The claim keeps running in its thread and may still return a sandbox nobody is waiting for.
            claim.add_done_callback(_release_abandoned_claim)
            raise
        self._deadline = time.monotonic() + self.runtime_timeout
        self._expiry = asyncio.create_task(self._expire())
        await self._setup_executable_paths(self.executable_paths)

    async def stop(self) -> None:
        expiry, self._expiry = self._expiry, None
        if expiry is not None and expiry is not asyncio.current_task():
            expiry.cancel()
        handle, self._handle = self._handle, None
        if handle is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_executor("claim", _CLAIM_WORKERS), self._fleet.release, handle)

    async def _expire(self) -> None:
        await asyncio.sleep(self.runtime_timeout)
        self._expired = True
        logger.warning("GKEAgentSandbox reached runtime_timeout=%gs; stopping it", self.runtime_timeout)
        try:
            await self.stop()
        except Exception:
            logger.warning("failed to stop an expired GKEAgentSandbox", exc_info=True)

    async def is_alive(self) -> bool:
        return self._handle is not None and not self._expired

    async def open_shell(self, *, cwd: str | None = None, env: dict[str, str] | None = None) -> _Shell:
        return _Shell(self, cwd=cwd, env=env)

    # ----- data plane -----
    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        if self._expired:
            raise RuntimeError(f"GKEAgentSandbox was stopped after runtime_timeout={self.runtime_timeout:g}s")
        if self._handle is None:
            raise RuntimeError("GKEAgentSandbox not started; call start() first")
        # A command may not outlive the sandbox.
        limit = min(float(timeout or _DEFAULT_EXEC_TIMEOUT), max(self._deadline - time.monotonic(), 1.0))
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor("exec", _EXEC_WORKERS), _exec_in_pod, self._fleet, self._handle, argv, limit, workdir, env
        )
