"""Unit tests for the ``routes`` traffic-split sandbox provider.

Everything here is in-memory: fake providers registered under ``test_routes_*``
names record lifecycle/data-plane calls so the tests can prove deterministic
weighted routing, start-time failover, non-retryable error propagation, the
retry-amplification cap and reverse-tunnel runtime injection without any SDK,
network or credentials.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest

from uni_agent.sandbox import routes as routes_module
from uni_agent.sandbox.base import ExecResult, Sandbox, SandboxConfig
from uni_agent.sandbox.registry import get_sandbox_cls, register_sandbox
from uni_agent.sandbox.routes import RouteSpec, RoutesSandbox, _bucket_position

_START_EXCEPTIONS: dict[str, type[Exception]] = {
    "runtime": RuntimeError,
    "value": ValueError,
    "type": TypeError,
    "import": ImportError,
    "notimpl": NotImplementedError,
    "permission": PermissionError,
}

# Every fake child constructed by a RoutesSandbox, in construction order.
_CREATED: list[_FakeRouteSandbox] = []


class _FakeTimeout(Exception):
    """Provider-specific timeout type, so delegation of ``_is_timeout_error`` is observable."""


class _FakeRouteSandbox(Sandbox):
    """Scriptable child provider: behaviour comes from ``sandbox_kwargs``."""

    supports_shell: ClassVar[bool] = True
    supports_reverse_tunnel: ClassVar[bool] = True

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.started = 0
        self.stopped = 0
        self.calls: list[tuple[str, Any]] = []
        self._alive = False
        _CREATED.append(self)

    @classmethod
    def from_config(cls, config: SandboxConfig) -> _FakeRouteSandbox:
        return cls(**config.sandbox_kwargs)

    async def start(self) -> None:
        self.started += 1
        behavior = self.kwargs.get("start_behavior", "ok")
        if behavior == "ok":
            self._alive = True
            return
        if behavior == "sleep":
            await asyncio.sleep(self.kwargs.get("sleep_seconds", 10))
            self._alive = True
            return
        if behavior == "chain":
            raise RuntimeError(f"{self.provider} failed") from KeyError(f"{self.provider} root cause")
        raise _START_EXCEPTIONS[behavior](f"{self.provider} failed: {behavior}")

    async def stop(self) -> None:
        self.stopped += 1
        self._alive = False
        if self.kwargs.get("stop_behavior") == "raise":
            raise RuntimeError("fake stop failed")

    async def is_alive(self) -> bool:
        return self._alive

    def _is_timeout_error(self, exc: BaseException) -> bool:
        return isinstance(exc, _FakeTimeout)

    def _is_fatal_exec_error(self, exc: BaseException) -> bool:
        return isinstance(exc, LookupError)

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        self.calls.append(("_exec", list(argv)))
        return ExecResult(exit_code=0, stdout="fake", stderr="")

    async def read_file(self, path: str) -> bytes:
        self.calls.append(("read_file", path))
        return b"fake-bytes"

    async def write_file(self, path: str, content: bytes | str) -> None:
        self.calls.append(("write_file", (path, content)))

    async def upload_file(self, local_file: Any, remote_file: str) -> None:
        self.calls.append(("upload_file", (local_file, remote_file)))

    async def download_file(self, remote_file: str, local_file: Any) -> None:
        self.calls.append(("download_file", (remote_file, local_file)))

    async def expose_port(self, port: int) -> str:
        self.calls.append(("expose_port", port))
        return f"http://fake:{port}"

    async def open_shell(self, *, cwd: str | None = None, env: dict[str, str] | None = None) -> Any:
        self.calls.append(("open_shell", (cwd, env)))
        return SimpleNamespace(run=AsyncMock(), close=AsyncMock())


@register_sandbox("test_routes_ok")
class _OkRouteSandbox(_FakeRouteSandbox):
    """Tunnel-capable fake route provider."""


@register_sandbox("test_routes_plain")
class _PlainRouteSandbox(_FakeRouteSandbox):
    """Fake route provider without reverse-tunnel (or native shell) support."""

    supports_shell = False
    supports_reverse_tunnel = False


@pytest.fixture(autouse=True)
def _reset_created_children() -> Iterator[None]:
    _CREATED.clear()
    yield
    _CREATED.clear()


def _route(
    name: str = "",
    *,
    provider: str = "test_routes_ok",
    weight: float = 1.0,
    start_timeout: float | None = None,
    **sandbox_kwargs: Any,
) -> RouteSpec:
    return RouteSpec(
        name=name,
        weight=weight,
        start_timeout=start_timeout,
        config=SandboxConfig(provider=provider, sandbox_kwargs=dict(sandbox_kwargs)),
    )


def _sandbox(*routes: RouteSpec, **kwargs: Any) -> RoutesSandbox:
    return RoutesSandbox(routes=list(routes), **kwargs)


def _keys(count: int) -> list[str]:
    """Deterministic pseudo-session ids; stable across runs (no random per call)."""
    return [f"session-{index:05d}" for index in range(count)]


async def _no_sleep(*_: Any, **__: Any) -> None:
    return None


# --------------------------- deterministic routing ---------------------------


@pytest.mark.cpu
@pytest.mark.level0
def test_bucket_position_matches_sha256_definition():
    salt, key = "rollout-2026-11", "session-42"
    expected = int.from_bytes(hashlib.sha256(f"{salt}\x00{key}".encode()).digest()[:8], "big") / 2**64
    assert _bucket_position(salt, key) == expected
    # Different salts must not collapse to the same bucket stream.
    assert _bucket_position("other-salt", key) != expected


@pytest.mark.cpu
@pytest.mark.level0
def test_weighted_routing_is_deterministic_for_salt_and_key():
    routes = [_route("a", weight=1), _route("b", weight=1)]
    sandbox = _sandbox(*routes, salt="s", route_key="session-1")
    selected = [sandbox._primary_index() for _ in range(20)]
    assert len(set(selected)) == 1
    # A fresh instance with the same key/salt/weights owns the same route.
    assert _sandbox(*routes, salt="s", route_key="session-1")._primary_index() == selected[0]


@pytest.mark.cpu
@pytest.mark.level0
def test_primary_strategy_always_selects_first_declared_route():
    routes = [_route("canary", weight=1), _route("prod", weight=99)]
    primary = _sandbox(*routes, strategy="primary", salt="s")
    assert [primary._primary_index() for _ in _keys(50)] == [0] * 50
    # Same keys on the weighted strategy do use the second route, proving the
    # primary strategy is actually pinning traffic rather than weights deciding.
    weighted_selection = {_sandbox(*routes, salt="s", route_key=key)._primary_index() for key in _keys(1000)}
    assert weighted_selection == {0, 1}


@pytest.mark.cpu
@pytest.mark.level0
def test_weighted_distribution_is_within_five_percentage_points():
    routes = [_route("prod", weight=90), _route("canary", weight=10)]
    counts = {0: 0, 1: 0}
    for key in _keys(1000):
        counts[_sandbox(*routes, salt="rollout", route_key=key)._primary_index()] += 1
    assert abs(counts[0] / 1000 - 0.9) <= 0.05
    assert abs(counts[1] / 1000 - 0.1) <= 0.05

    three_routes = [_route("a", weight=50), _route("b", weight=30), _route("c", weight=20)]
    counts = {0: 0, 1: 0, 2: 0}
    for key in _keys(1000):
        counts[_sandbox(*three_routes, salt="rollout", route_key=key)._primary_index()] += 1
    for index, expected in enumerate((0.5, 0.3, 0.2)):
        assert abs(counts[index] / 1000 - expected) <= 0.05


@pytest.mark.cpu
@pytest.mark.level0
def test_different_salt_changes_the_selected_route():
    routes = [_route("a", weight=1), _route("b", weight=1)]
    first = [_sandbox(*routes, salt="run-1", route_key=key)._primary_index() for key in _keys(100)]
    second = [_sandbox(*routes, salt="run-2", route_key=key)._primary_index() for key in _keys(100)]
    assert first != second


@pytest.mark.cpu
@pytest.mark.level0
def test_route_key_generated_once_and_kept_for_explicit_value():
    generated = _sandbox(_route("a"), _route("b"))
    assert isinstance(generated.route_key, str) and generated.route_key
    assert len(generated.route_key) == len(uuid.uuid4().hex)
    assert generated._primary_index() == generated._primary_index()

    explicit = _sandbox(_route("a"), route_key="gateway-session")
    assert explicit.route_key == "gateway-session"


# --------------------------- configuration validation ---------------------------


@pytest.mark.cpu
@pytest.mark.level0
def test_empty_routes_are_rejected():
    with pytest.raises(ValueError, match="non-empty list"):
        RoutesSandbox(routes=[])


@pytest.mark.cpu
@pytest.mark.level0
def test_nested_routes_are_rejected():
    with pytest.raises(ValueError, match="nested routes"):
        RoutesSandbox(routes=[{"name": "outer", "config": {"provider": "routes"}}])


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf")])
def test_route_weight_must_be_positive_and_finite(weight: float):
    with pytest.raises(ValueError):
        _route("broken", weight=weight)


@pytest.mark.cpu
@pytest.mark.level0
def test_duplicate_route_names_are_rejected():
    with pytest.raises(ValueError, match="not unique"):
        _sandbox(_route("same"), _route("same"))
    # An explicit name colliding with an auto-generated one is still a duplicate.
    with pytest.raises(ValueError, match="not unique"):
        _sandbox(_route("route-1"), _route(""))


@pytest.mark.cpu
@pytest.mark.level0
def test_empty_route_names_are_auto_assigned_in_order():
    sandbox = _sandbox(_route(""), _route(""), strategy="primary")
    assert [route.name for route in sandbox.routes] == ["route-0", "route-1"]


@pytest.mark.cpu
@pytest.mark.level0
def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown routes strategy"):
        RoutesSandbox(routes=[_route("a")], strategy="random")  # type: ignore[arg-type]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "overrides",
    [
        {"image": "ubuntu:24.04"},
        # image_map needs an image for its own validation before routes can reject it.
        {"image": "ubuntu:24.04", "image_map": [{"from": "a", "to": "b"}]},
        {"image_mounts": [{"image": "sidecar:v1", "mount_path": "/mnt/sidecar"}]},
        {"executable_paths": {"python3": "/usr/bin/python3"}},
    ],
)
def test_from_config_rejects_top_level_standard_fields(overrides: dict[str, Any]):
    config = SandboxConfig(provider="routes", **overrides)
    with pytest.raises(ValueError, match="configure .* on each routes"):
        RoutesSandbox.from_config(config)


@pytest.mark.cpu
@pytest.mark.level0
def test_from_config_accepts_raw_routes_and_optional_kwargs():
    config = SandboxConfig(
        provider="routes",
        sandbox_kwargs={
            "strategy": "weighted",
            "salt": "rollout-2026-11",
            "fallback": False,
            "start_timeout": 5,
            "routes": [
                {"name": "e2b-prod", "weight": 90, "config": {"provider": "test_routes_ok"}},
                {"weight": 10, "config": {"provider": "test_routes_plain", "sandbox_kwargs": {"start_behavior": "ok"}}},
            ],
        },
    )
    sandbox = RoutesSandbox.from_config(config)
    assert [route.name for route in sandbox.routes] == ["e2b-prod", "route-1"]
    assert [route.weight for route in sandbox.routes] == [90, 10]
    assert isinstance(sandbox.routes[0].config, SandboxConfig)
    assert sandbox.strategy == "weighted"
    assert sandbox.salt == "rollout-2026-11"
    assert sandbox.fallback is False
    assert sandbox.start_timeout == 5.0


@pytest.mark.cpu
@pytest.mark.level0
def test_route_spec_forbids_unknown_fields():
    with pytest.raises(ValueError):
        RouteSpec(name="a", config=SandboxConfig(provider="test_routes_ok"), unknown="oops")  # type: ignore[call-arg]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("start_timeout", [0, -1])
def test_invalid_route_start_timeout_is_rejected(start_timeout: float):
    with pytest.raises(ValueError):
        _route("a", start_timeout=start_timeout)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("start_timeout", [0, -1, float("inf")])
def test_invalid_routes_level_start_timeout_is_rejected(start_timeout: float):
    with pytest.raises(ValueError):
        _sandbox(_route("a"), start_timeout=start_timeout)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("proxy_port", [True, 0, 1, 65536, "38197"])
def test_invalid_proxy_port_is_rejected(proxy_port: Any):
    with pytest.raises(ValueError, match="proxy_port"):
        _sandbox(_route("a"), proxy_port=proxy_port)


@pytest.mark.cpu
@pytest.mark.level0
def test_invalid_upstream_and_route_key_are_rejected():
    with pytest.raises(ValueError, match="upstream"):
        _sandbox(_route("a"), upstream="  ")
    with pytest.raises(TypeError, match="route_key"):
        _sandbox(_route("a"), route_key=123)  # type: ignore[arg-type]


# --------------------------- start-time failover ---------------------------


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_transient_start_failure_fails_over_to_next_route():
    sandbox = _sandbox(
        _route("broken", start_behavior="runtime"),
        _route("healthy"),
        strategy="primary",
    )
    await sandbox.start()

    assert sandbox.active_route_name == "healthy"
    assert sandbox.active_route_index == 1
    assert sandbox.attempts == [("broken", "failed"), ("healthy", "ok")]
    broken, healthy = _CREATED
    assert broken.started == 1 and broken.stopped == 1
    assert healthy.started == 1 and healthy.stopped == 0
    assert sandbox._sandbox is healthy


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_failover_wraps_around_from_a_non_primary_primary():
    routes = [_route("a"), _route("b", start_behavior="runtime")]
    sandbox = _sandbox(*routes, salt="s")
    # Pick a key whose weighted primary is the *second* route, then prove the
    # failover order wraps back to the first declaration.
    for key in _keys(200):
        sandbox.route_key = key
        if sandbox._primary_index() == 1:
            break
    else:  # pragma: no cover - 50/50 weights make this unreachable in practice
        pytest.fail("could not find a key that routes to the second route")

    await sandbox.start()
    assert sandbox.attempts == [("b", "failed"), ("a", "ok")]
    assert sandbox.active_route_name == "a"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_no_fallback_raises_without_trying_other_routes():
    sandbox = _sandbox(
        _route("broken", start_behavior="runtime"),
        _route("healthy"),
        strategy="primary",
        fallback=False,
    )
    with pytest.raises(RuntimeError, match="failed: runtime"):
        await sandbox.start()
    assert sandbox.attempts == [("broken", "failed")]
    assert len(_CREATED) == 1
    assert _CREATED[0].stopped == 1
    assert sandbox._sandbox is None


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_per_route_start_timeout_fails_over():
    sandbox = _sandbox(
        _route("slow", start_behavior="sleep", sleep_seconds=30, start_timeout=0.01),
        _route("healthy"),
        strategy="primary",
    )
    await asyncio.wait_for(sandbox.start(), timeout=5)
    assert sandbox.active_route_name == "healthy"
    assert [status for _, status in sandbox.attempts] == ["failed", "ok"]
    slow = _CREATED[0]
    assert slow.stopped == 1


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_start_is_idempotent():
    sandbox = _sandbox(_route("only"))
    await sandbox.start()
    await sandbox.start()
    assert len(_CREATED) == 1
    assert _CREATED[0].started == 1


# --------------------------- non-retryable errors ---------------------------


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["value", "type", "import", "notimpl", "permission"])
async def test_configuration_and_sdk_errors_are_not_failed_over(behavior: str):
    sandbox = _sandbox(
        _route("fatal", start_behavior=behavior),
        _route("healthy"),
        strategy="primary",
    )
    expected = _START_EXCEPTIONS[behavior]
    with pytest.raises(expected):
        await sandbox.start()
    # Only the primary route was tried; the healthy fallback was never built.
    assert sandbox.attempts == [("fatal", "failed")]
    assert len(_CREATED) == 1
    assert _CREATED[0].started == 1 and _CREATED[0].stopped == 1
    assert sandbox._sandbox is None


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_non_retryable_error_is_not_retried_by_aenter():
    sandbox = _sandbox(_route("fatal", start_behavior="value"))
    with pytest.raises(ValueError):
        await sandbox.__aenter__()
    # A single pass, not retry(2) x routes: config errors are never transient.
    assert len(_CREATED) == 1
    assert _CREATED[0].started == 1


# --------------------------- all routes fail / retry cap ---------------------------


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_all_routes_failed_raises_last_error_with_chain():
    sandbox = _sandbox(
        _route("first", start_behavior="chain"), _route("last", start_behavior="chain"), strategy="primary"
    )
    with pytest.raises(RuntimeError) as caught:
        await sandbox.start()
    assert sandbox.attempts == [("first", "failed"), ("last", "failed")]
    assert isinstance(caught.value.__cause__, KeyError)
    assert "test_routes_ok root cause" in str(caught.value.__cause__)
    assert sandbox._sandbox is None


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_aenter_retry_caps_amplification_at_routes_times_retry(monkeypatch):
    import inspect

    assert inspect.signature(RoutesSandbox.__aenter__).parameters["retry"].default == 2
    monkeypatch.setattr(routes_module.asyncio, "sleep", _no_sleep)
    sandbox = _sandbox(_route("a", start_behavior="runtime"), _route("b", start_behavior="runtime"))
    with pytest.raises(RuntimeError):
        await sandbox.__aenter__(retry=2)

    # 2 passes x 2 routes = 4 child starts and 4 constructions, never 3 x 2.
    assert sum(child.started for child in _CREATED) == 2 * 2
    assert len(_CREATED) == 4
    # Every failed child attempt releases its partial sandbox exactly once.
    assert all(child.stopped == 1 for child in _CREATED)


# --------------------------- data plane delegation ---------------------------


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_data_plane_delegates_to_the_active_child():
    sandbox = _sandbox(_route("only"))
    await sandbox.start()
    child = _CREATED[0]

    assert (await sandbox.exec(["echo", "hi"])).stdout == "fake"
    assert (await sandbox.exec_shell("true")).exit_code == 0
    assert await sandbox.read_file("/tmp/x") == b"fake-bytes"
    await sandbox.write_file("/tmp/y", b"data")
    await sandbox.upload_file("/local", "/remote")
    await sandbox.download_file("/remote", "/local")
    assert await sandbox.expose_port(8080) == "http://fake:8080"
    handle = await sandbox.open_shell(cwd="/work", env={"A": "1"})
    await handle.close()
    assert await sandbox.is_alive() is True
    assert sandbox._is_timeout_error(_FakeTimeout())
    assert sandbox._is_fatal_exec_error(KeyError("infra"))

    names = [name for name, _ in child.calls]
    assert names == [
        "_exec",
        "_exec",
        "read_file",
        "write_file",
        "upload_file",
        "download_file",
        "expose_port",
        "open_shell",
    ]
    assert child.calls[0] == ("_exec", ["echo", "hi"])
    assert child.calls[1] == ("_exec", ["bash", "-c", "true"])


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_data_plane_requires_start():
    sandbox = _sandbox(_route("only"))

    with pytest.raises(RuntimeError, match="not started"):
        await sandbox.exec(["true"])
    with pytest.raises(RuntimeError, match="not started"):
        await sandbox.read_file("/tmp/x")
    with pytest.raises(RuntimeError, match="not started"):
        await sandbox.write_file("/tmp/x", b"data")
    with pytest.raises(RuntimeError, match="not started"):
        await sandbox.upload_file("/local", "/remote")
    with pytest.raises(RuntimeError, match="not started"):
        await sandbox.download_file("/remote", "/local")
    with pytest.raises(RuntimeError, match="not started"):
        await sandbox.expose_port(8080)
    with pytest.raises(RuntimeError, match="not started"):
        await sandbox.open_shell()
    with pytest.raises(RuntimeError, match="not started"):
        await sandbox.is_alive()
    with pytest.raises(RuntimeError, match="not started"):
        sandbox._is_timeout_error(TimeoutError())
    with pytest.raises(RuntimeError, match="not started"):
        sandbox._is_fatal_exec_error(KeyError("infra"))


# --------------------------- stop semantics ---------------------------


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_stop_delegates_once_and_is_idempotent():
    sandbox = _sandbox(_route("only"))
    await sandbox.start()
    await sandbox.stop()
    child = _CREATED[0]
    assert child.stopped == 1
    assert sandbox._sandbox is None
    assert sandbox.active_route_name is None and sandbox.active_route_index is None
    # Capability shadows are dropped so class defaults apply to a stopped instance.
    assert "supports_shell" not in sandbox.__dict__
    assert "supports_reverse_tunnel" not in sandbox.__dict__

    await sandbox.stop()
    assert child.stopped == 1


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_stop_without_start_is_a_noop():
    sandbox = _sandbox(_route("only"))
    await sandbox.stop()
    assert _CREATED == []
    assert sandbox.attempts == []


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_stop_swallows_child_stop_failure():
    sandbox = _sandbox(_route("only", stop_behavior="raise"))
    await sandbox.start()
    await sandbox.stop()
    assert sandbox._sandbox is None


# --------------------------- reverse tunnel ---------------------------


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_reverse_tunnel_requires_every_route_provider_to_support_it():
    sandbox = _sandbox(
        _route("ok"),
        _route("plain", provider="test_routes_plain"),
        strategy="primary",
        upstream="gateway.example:40169",
        proxy_port=38197,
    )
    with pytest.raises(ValueError, match="supports_reverse_tunnel=True"):
        await sandbox.start()
    assert _CREATED == []
    assert sandbox.attempts == []


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_reverse_tunnel_runtime_kwargs_are_merged_into_the_selected_child():
    sandbox = _sandbox(
        _route("only"),
        strategy="primary",
        upstream="gateway.example:40169",
        proxy_port=38197,
    )
    original_kwargs = dict(sandbox.routes[0].config.sandbox_kwargs)
    await sandbox.start()

    child = _CREATED[0]
    assert child.kwargs["upstream"] == "gateway.example:40169"
    assert child.kwargs["proxy_port"] == 38197
    # Runtime injection must not pollute the shared route spec.
    assert sandbox.routes[0].config.sandbox_kwargs == original_kwargs


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_reverse_tunnel_without_config_does_not_inject_kwargs():
    sandbox = _sandbox(_route("only"))
    await sandbox.start()
    assert "upstream" not in _CREATED[0].kwargs
    assert "proxy_port" not in _CREATED[0].kwargs


# --------------------------- capabilities / registration ---------------------------


@pytest.mark.cpu
@pytest.mark.level0
def test_routes_provider_is_registered_and_advertises_reverse_tunnel():
    assert get_sandbox_cls("routes") is RoutesSandbox
    assert RoutesSandbox.supports_reverse_tunnel is True
    # Before start the instance falls back to the class flags.
    assert _sandbox(_route("only")).supports_shell is False


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_capability_flags_are_shadowed_from_the_active_child():
    sandbox = _sandbox(_route("only"))
    assert sandbox.supports_shell is False  # class default before start
    await sandbox.start()
    # The fake child supports a native shell: the routed instance must reflect it.
    assert sandbox.supports_shell is True
    assert sandbox.supports_reverse_tunnel is True

    await sandbox.stop()
    assert sandbox.supports_shell is False
