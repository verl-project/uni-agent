"""Traffic-split sandbox provider: weighted routing with start-time failover.

``RoutesSandbox`` is a *composite* provider: it owns no backend of its own, it
picks one of several child :class:`~uni_agent.sandbox.base.SandboxConfig`
entries and then delegates the whole data plane to that child. Routing is
deterministic so a stable key (by default the gateway ``session_id`` injected by
:mod:`uni_agent.framework.task_runner`) always lands on the same route for a
given ``salt``; the ``weighted`` strategy is the grey-release knob and
``primary`` pins traffic to the first route (canary stays fallback-only).

Start-time failover is deliberately *inside* the provider: if a weighted route
cannot start (transient infra error, timeout) the next route is tried in
declaration order (wrapping around). Configuration/SDK/permission errors are
never masked by failover -- see ``_NON_RETRYABLE_START_ERRORS`` -- because
silently switching routes on a misconfiguration would hide the root cause.

Example (``sandbox_kwargs`` of a task config)::

    sandbox:
      provider: routes
      sandbox_kwargs:
        strategy: weighted
        salt: rollout-2026-11
        routes:
          - name: e2b-prod
            weight: 90
            config: {provider: e2b, image: ubuntu:24.04, sandbox_kwargs: {template: uni-agent-swe}}
          - name: agentenv-canary
            weight: 10
            config: {provider: agentenv, sandbox_kwargs: {template: uni-agent-swe}}
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import uuid
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from .base import (
    ExecResult,
    Sandbox,
    SandboxConfig,
    _startup_slot,
    _stop_after_failed_start_shielded,
)
from .registry import get_sandbox_cls, register_sandbox

logger = logging.getLogger(__name__)

#: A failing route is retried/failed-over only for transient errors. These are
#: configuration/SDK/permission problems: fall over instead of hiding them.
_NON_RETRYABLE_START_ERRORS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    ImportError,
    NotImplementedError,
    PermissionError,
)

RouteStrategy = Literal["weighted", "primary"]


def _bucket_position(salt: str, route_key: str) -> float:
    """Map ``(salt, route_key)`` to a stable position in ``[0, 1]``.

    ``sha256`` (not the per-process-randomised builtin ``hash``) keeps the
    mapping reproducible across runs and hosts, so a rollout's traffic split
    stays stable while the weights are tuned. The first 8 digest bytes give a
    64-bit bucket; dividing by ``2**64`` yields a float.
    """
    digest = hashlib.sha256(f"{salt}\x00{route_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class RouteSpec(BaseModel):
    """One traffic route: a named child sandbox config plus its routing weight."""

    name: str = Field(
        default="",
        description="Route name; empty means auto 'route-{index}'. Non-empty names must be unique per sandbox.",
    )
    weight: float = Field(
        default=1.0,
        gt=0,
        description="Relative traffic weight; the 'weighted' strategy splits proportionally to it.",
    )
    config: SandboxConfig = Field(
        description=(
            "Full child sandbox config; standard fields (image, image_mounts, executable_paths, image_map) belong here."
        ),
    )
    start_timeout: float | None = Field(
        default=None,
        gt=0,
        description="Per-route start timeout in seconds; falls back to the routes-level start_timeout.",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("weight", "start_timeout")
    @classmethod
    def _validate_finite(cls, value: float | None) -> float | None:
        # NaN/Inf would silently corrupt bucket selection and timeout arithmetic.
        if value is not None and not math.isfinite(value):
            raise ValueError("route weight/start_timeout must be finite")
        return value


_ROUTE_LIST_ADAPTER: TypeAdapter[list[RouteSpec]] = TypeAdapter(list[RouteSpec])


@register_sandbox("routes")
class RoutesSandbox(Sandbox):
    """Weighted traffic split across sandbox providers with start-time failover.

    The data-plane surface (exec, file transfer, port exposure) is delegated to
    the active child; the class-level capability flags are stamped per instance
    after ``start()`` because the active route is only known then.
    """

    # Runtime reverse-tunnel injection (task_runner._inject_gateway_tunnel) keys
    # off the provider *class*, so routes must advertise the capability up front.
    # ``start()`` enforces that every configured child provider really supports
    # it whenever a tunnel is configured.
    supports_reverse_tunnel: ClassVar[bool] = True

    def __init__(
        self,
        *,
        routes: list[RouteSpec],
        strategy: RouteStrategy = "weighted",
        salt: str = "",
        fallback: bool = True,
        route_key: str | None = None,
        start_timeout: float | None = None,
        upstream: str | None = None,
        proxy_port: int | None = None,
    ) -> None:
        if not isinstance(routes, list) or not routes:
            raise ValueError("routes must be a non-empty list of RouteSpec")
        # Accept raw mappings too (YAML/from_config): pydantic validates each
        # entry, including the nested SandboxConfig and its extra='forbid' rule.
        validated_routes = _ROUTE_LIST_ADAPTER.validate_python(routes)
        names: list[str] = []
        for index, route in enumerate(validated_routes):
            name = route.name or f"route-{index}"
            if name in names:
                raise ValueError(f"route name {name!r} is not unique; names must be unique per RoutesSandbox")
            names.append(name)
        # Nesting a routes sandbox inside another one would re-enter the startup
        # slot and can self-lock once SANDBOX_STARTUP_CONCURRENCY is 1; one
        # routes sandbox is enough to express any split, so reject the nesting.
        nested = [
            name for name, route in zip(names, validated_routes, strict=True) if route.config.provider == "routes"
        ]
        if nested:
            raise ValueError(
                f"nested routes sandboxes are not supported: {', '.join(nested)}; "
                "declare every backend as a route of a single routes sandbox instead"
            )
        # Copy with resolved names instead of mutating the caller's specs.
        self.routes: list[RouteSpec] = [
            route.model_copy(update={"name": name}) for route, name in zip(validated_routes, names, strict=True)
        ]

        if strategy not in ("weighted", "primary"):
            raise ValueError(f"unknown routes strategy {strategy!r}; expected 'weighted' or 'primary'")
        self.strategy: RouteStrategy = strategy
        if not isinstance(salt, str):
            raise TypeError("salt must be a string")
        self.salt = salt
        if not isinstance(fallback, bool):
            raise TypeError("fallback must be a boolean")
        self.fallback = fallback
        if route_key is not None and not isinstance(route_key, str):
            raise TypeError("route_key must be a string or None")
        # Generated once per instance: retries of the same sandbox stay on the
        # same route. The framework injects the gateway session id when available
        # (a random key would silently route every session by chance only).
        self.route_key: str = route_key if route_key is not None else uuid.uuid4().hex
        self.start_timeout = _positive_or_none(start_timeout, "start_timeout")
        if upstream is not None and (not isinstance(upstream, str) or not upstream.strip()):
            raise ValueError("upstream must be a non-empty 'host:port' string when set")
        self.upstream = upstream
        if proxy_port is not None and (
            isinstance(proxy_port, bool) or not isinstance(proxy_port, int) or not 2 <= proxy_port <= 65535
        ):
            raise ValueError("proxy_port must be an integer between 2 and 65535 when set")
        self.proxy_port = proxy_port

        self.attempts: list[tuple[str, str]] = []
        self.active_route_name: str | None = None
        self.active_route_index: int | None = None
        self._sandbox: Sandbox | None = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> RoutesSandbox:
        """Build from ``sandbox_kwargs``; standard fields must live on each route."""
        misconfigured = [
            field
            for field, value in (
                ("image", config.image),
                ("image_map", config.image_map),
                ("image_mounts", config.image_mounts),
                ("executable_paths", config.executable_paths),
            )
            if value
        ]
        if misconfigured:
            raise ValueError(
                f"RoutesSandbox does not accept top-level {', '.join(misconfigured)}; "
                "configure image/image_map/image_mounts/executable_paths on each routes[].config instead"
            )
        return cls(**config.sandbox_kwargs)

    # ----- control plane -----
    async def start(self) -> None:
        """Start exactly one child: pick the primary route, fail over on transient errors."""
        if self._sandbox is not None:
            return
        self._validate_reverse_tunnel_support()
        primary = self._primary_index()
        order = self._failover_order(primary)
        last_exc: BaseException | None = None
        for index in order:
            spec = self.routes[index]
            child: Sandbox | None = None
            try:
                child = self._build_child(spec)
                timeout = spec.start_timeout if spec.start_timeout is not None else self.start_timeout
                async with _startup_slot():
                    if timeout is None:
                        await child._run_start()
                    else:
                        await asyncio.wait_for(child._run_start(), timeout=timeout)
            except _NON_RETRYABLE_START_ERRORS as exc:
                # Misconfiguration/SDK/permission errors are not transient:
                # failing over would mask the real problem.
                self.attempts.append((spec.name, "failed"))
                if child is not None:
                    await _stop_after_failed_start_shielded(child)
                logger.error("sandbox route %s failed with a non-retryable error: %r", spec.name, exc)
                raise
            except Exception as exc:
                self.attempts.append((spec.name, "failed"))
                if child is not None:
                    await _stop_after_failed_start_shielded(child)
                last_exc = exc
                logger.warning("sandbox route %s failed to start, trying next route: %r", spec.name, exc)
                continue
            except BaseException:
                # Cancelled mid-start: release the partial child, then propagate.
                self.attempts.append((spec.name, "failed"))
                if child is not None:
                    await _stop_after_failed_start_shielded(child)
                raise

            self._sandbox = child
            self.active_route_name = spec.name
            self.active_route_index = index
            self.attempts.append((spec.name, "ok"))
            # Base declares these as ClassVars; a routed sandbox only learns its
            # active child's capabilities after start(), so shadow them per
            # instance (tools/task_runner read the instance attributes).
            self.supports_shell = child.supports_shell
            self.supports_reverse_tunnel = child.supports_reverse_tunnel
            logger.info(
                "sandbox route selected: %s (key=%s strategy=%s)",
                spec.name,
                self.route_key,
                self.strategy,
            )
            return
        assert last_exc is not None
        logger.error(
            "all sandbox routes failed to start (key=%s strategy=%s attempts=%s)",
            self.route_key,
            self.strategy,
            self.attempts,
        )
        raise last_exc

    async def stop(self) -> None:
        """Stop the active child once; safe to call repeatedly or when never started."""
        child, name = self._sandbox, self.active_route_name
        self._sandbox = None
        self.active_route_name = None
        self.active_route_index = None
        # Drop the per-instance capability overrides so class defaults apply again.
        self.__dict__.pop("supports_shell", None)
        self.__dict__.pop("supports_reverse_tunnel", None)
        if child is None:
            return
        try:
            await child.stop()
        except Exception:
            logger.warning("failed to stop routed sandbox %s", name, exc_info=True)

    async def __aenter__(self, retry: int = 2) -> RoutesSandbox:
        """Start with at most ``retry`` passes (each pass tries every route once).

        The base implementation retries ``start()`` three times; multiplying that
        by the number of routes would amplify a broken endpoint into
        ``3 * len(routes)`` startup attempts. Two passes keep one failover sweep
        per retry. No ``_run_start`` wrapper (and thus no outer startup slot) is
        used: each route already acquires a slot and is bounded by its own
        timeout / ``SANDBOX_STARTUP_TIMEOUT``, and an outer slot would deadlock
        under ``SANDBOX_STARTUP_CONCURRENCY=1``.
        """
        retry = max(1, retry)
        last_exc: BaseException | None = None
        for attempt in range(1, retry + 1):
            try:
                await self.start()
                return self
            except _NON_RETRYABLE_START_ERRORS:
                raise
            except Exception as exc:
                last_exc = exc
                await _stop_after_failed_start_shielded(self)
            except BaseException:
                await _stop_after_failed_start_shielded(self)
                raise
            logger.warning("sandbox routes failed to start (pass %d/%d): %r", attempt, retry, last_exc)
            if attempt < retry:
                await asyncio.sleep(2 * attempt)
        assert last_exc is not None
        logger.error("sandbox routes failed to start after %d passes: %r", retry, last_exc)
        raise last_exc

    # ----- routing -----
    def _primary_index(self) -> int:
        """Deterministically select the primary route for ``salt`` + ``route_key``."""
        if self.strategy == "primary":
            return 0
        total = sum(route.weight for route in self.routes)
        position = _bucket_position(self.salt, self.route_key) * total
        cumulative = 0.0
        for index, route in enumerate(self.routes):
            cumulative += route.weight
            if position < cumulative:
                return index
        # bucket == 1.0 is possible after float rounding for extreme keys.
        return len(self.routes) - 1

    def _failover_order(self, primary: int) -> list[int]:
        """Primary first, then declaration order wrapping around (empty without fallback)."""
        if not self.fallback:
            return [primary]
        count = len(self.routes)
        return [(primary + offset) % count for offset in range(count)]

    # ----- children -----
    def _build_child(self, spec: RouteSpec) -> Sandbox:
        """Instantiate the child provider for ``spec``, merging runtime tunnel kwargs."""
        child_config = spec.config
        runtime_kwargs: dict[str, Any] = {}
        if self.upstream is not None:
            runtime_kwargs["upstream"] = self.upstream
        if self.proxy_port is not None:
            runtime_kwargs["proxy_port"] = self.proxy_port
        if runtime_kwargs:
            # Instantiate from a private deep copy so runtime injection never
            # leaks into the shared spec (the same spec is reused on retries and
            # by other sandbox instances built from the same task config).
            child_config = child_config.model_copy(deep=True)
            child_config = child_config.model_copy(
                update={"sandbox_kwargs": {**child_config.sandbox_kwargs, **runtime_kwargs}}
            )
        return get_sandbox_cls(child_config.provider).from_config(child_config)

    def _validate_reverse_tunnel_support(self) -> None:
        """Reject a tunnel config when any route's provider cannot reverse-tunnel."""
        if self.upstream is None and self.proxy_port is None:
            return
        unsupported: list[str] = []
        for spec in self.routes:
            provider = spec.config.provider
            if not get_sandbox_cls(provider).supports_reverse_tunnel:
                unsupported.append(f"{spec.name}({provider})")
        if unsupported:
            raise ValueError(
                "the gateway reverse tunnel (upstream/proxy_port) requires every route provider to declare "
                f"supports_reverse_tunnel=True; unsupported routes: {', '.join(unsupported)}"
            )

    def _active_child(self) -> Sandbox:
        if self._sandbox is None:
            raise RuntimeError("RoutesSandbox not started; call start() first")
        return self._sandbox

    # ----- data plane: delegate to the active child -----
    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        return await self._active_child()._exec(argv, timeout=timeout, workdir=workdir, env=env)

    def _is_timeout_error(self, exc: BaseException) -> bool:
        return self._active_child()._is_timeout_error(exc)

    def _is_fatal_exec_error(self, exc: BaseException) -> bool:
        # Child providers may classify their own infrastructure errors as fatal
        # (e.g. agent_dx); keep that classification or routes would downgrade them
        # to a regular command failure.
        return self._active_child()._is_fatal_exec_error(exc)

    async def is_alive(self) -> bool:
        return await self._active_child().is_alive()

    async def open_shell(self, *, cwd: str | None = None, env: dict[str, str] | None = None) -> Any:
        return await self._active_child().open_shell(cwd=cwd, env=env)

    async def read_file(self, path: str) -> bytes:
        return await self._active_child().read_file(path)

    async def write_file(self, path: str, content: bytes | str) -> None:
        await self._active_child().write_file(path, content)

    async def upload_file(self, local_file: Path | str, remote_file: str) -> None:
        await self._active_child().upload_file(local_file, remote_file)

    async def download_file(self, remote_file: str, local_file: Path | str) -> None:
        await self._active_child().download_file(remote_file, local_file)

    async def expose_port(self, port: int) -> str:
        return await self._active_child().expose_port(port)


def _positive_or_none(value: float | None, name: str) -> float | None:
    """Validate an optional positive finite float (routes-level ``start_timeout``)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number or None")
    return float(value)
