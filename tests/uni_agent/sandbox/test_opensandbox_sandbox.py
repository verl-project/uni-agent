"""Unit tests for the OpenSandbox sandbox provider.

The provider imports the ``opensandbox`` SDK lazily inside :meth:`start` and the
data-plane methods, so every SDK-facing behavior is exercised against a fake
package injected into ``sys.modules`` -- no real SDK, network, or credentials.

Covered here:

* registry wiring and ``from_config`` mapping;
* constructor validation (image/snapshot_id exclusivity, budgets, image mounts);
* ``start``/``stop`` lifecycle against a fake ``Sandbox.create``/``destroy``;
* ``_exec`` result conversion: exit codes preserved, deadline kills mapped to
  ``TimeoutError``, lost results surfaced as errors for the shared policy;
* native file operations and port exposure.
"""

from __future__ import annotations

import sys
import types
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from uni_agent.sandbox.base import ExecResult, Sandbox, SandboxConfig
from uni_agent.sandbox.opensandbox import OpenSandboxSandbox
from uni_agent.sandbox.registry import SANDBOX_MODULES, build_sandbox

# --------------------------- fake opensandbox package ---------------------------


class _FakeRunCommandOpts(BaseModel):
    working_directory: str | None = None
    timeout: object | None = None
    envs: dict[str, str] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class _FakeNetworkRule(BaseModel):
    action: str
    target: str | None = None
    ports: list[int] | None = None


class _FakeNetworkPolicy(BaseModel):
    default_action: str | None = None
    egress: list[_FakeNetworkRule] | None = None

    model_config = ConfigDict(populate_by_name=True)


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = types.SimpleNamespace(name=name)
    return module


class _FakeSDKSandbox:
    """Stand-in for an ``opensandbox.Sandbox`` instance (what the provider calls)."""

    def __init__(self, *, sandbox_id: str = "sbx-fake-1", protocol: str = "http"):
        self.id = sandbox_id
        self.runs: list[dict] = []
        self.run_error: Exception | None = None
        self.executions: list[object] = []
        self.files = SimpleNamespace(read_bytes=None, write_file=None)
        self.connection_config = SimpleNamespace(protocol=protocol)
        self.endpoint_answers: dict[int, object] = {}
        self.destroy_calls = 0
        self.destroy_error: Exception | None = None
        self.health_answers: list[bool | Exception] = []

    async def destroy(self):
        self.destroy_calls += 1
        if self.destroy_error is not None:
            raise self.destroy_error

    async def is_healthy(self):
        answer = self.health_answers.pop(0) if self.health_answers else True
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def get_endpoint(self, port: int):
        return self.endpoint_answers[port]

    def attach_commands(self, executions: list[object] | Exception) -> None:
        """Script the next ``commands.run`` results (or a single failure)."""

        async def run(command, *, opts=None, handlers=None):
            self.runs.append({"command": command, "opts": opts})
            if isinstance(executions, Exception):
                raise executions
            return executions.pop(0)

        self.commands = SimpleNamespace(run=run)


class _FakeOpensandbox:
    """The module-level ``opensandbox`` surface: ``Sandbox.create``."""

    def __init__(self):
        self.create_calls: list[dict] = []
        self.create_results: list[object] = []
        self.Sandbox = self._build_sandbox_class()

    def _build_sandbox_class(self):
        fake = self

        class _SandboxClass:
            @classmethod
            async def create(cls, **kwargs):
                fake.create_calls.append(kwargs)
                if not fake.create_results:
                    return _FakeSDKSandbox()
                result = fake.create_results.pop(0)
                if isinstance(result, Exception):
                    raise result
                return result

        return _SandboxClass


def _install_fake_package(monkeypatch: pytest.MonkeyPatch, fake: _FakeOpensandbox) -> None:
    root = _module("opensandbox")
    root.Sandbox = fake.Sandbox
    models = _module("opensandbox.models")
    execd = _module("opensandbox.models.execd")
    sandboxes = _module("opensandbox.models.sandboxes")
    execd.RunCommandOpts = _FakeRunCommandOpts
    sandboxes.NetworkPolicy = _FakeNetworkPolicy
    sandboxes.NetworkRule = _FakeNetworkRule
    models.execd = execd
    models.sandboxes = sandboxes
    root.models = models
    monkeypatch.setitem(sys.modules, "opensandbox", root)
    monkeypatch.setitem(sys.modules, "opensandbox.models", models)
    monkeypatch.setitem(sys.modules, "opensandbox.models.execd", execd)
    monkeypatch.setitem(sys.modules, "opensandbox.models.sandboxes", sandboxes)


@pytest.fixture(autouse=True)
def fake_opensandbox(monkeypatch: pytest.MonkeyPatch) -> _FakeOpensandbox:
    """Install the fake ``opensandbox`` package for every test in this module.

    The provider imports the SDK lazily, so tests that never call ``start`` still
    need the module present (the real SDK is an optional dependency and must not
    be required by the unit suite).
    """
    fake = _FakeOpensandbox()
    _install_fake_package(monkeypatch, fake)
    return fake


def _execution(exit_code, *, stdout="", stderr="", error=None) -> SimpleNamespace:
    """Build an SDK-shaped execution: streamed logs plus an inferred exit code."""
    stdout_chunks = stdout if isinstance(stdout, list) else ([stdout] if stdout else [])
    logs = SimpleNamespace(
        stdout=[SimpleNamespace(text=chunk) for chunk in stdout_chunks],
        stderr=[SimpleNamespace(text=stderr)] if stderr else [],
    )
    return SimpleNamespace(exit_code=exit_code, logs=logs, error=error, complete=None, result=[])


def _attach(sandbox: OpenSandboxSandbox, sdk_sandbox: _FakeSDKSandbox) -> _FakeSDKSandbox:
    sandbox._sandbox = sdk_sandbox
    sandbox.sandbox_id = str(sdk_sandbox.id)
    return sdk_sandbox


# --------------------------- registry wiring ---------------------------


@pytest.mark.cpu
@pytest.mark.level0
def test_registry_builds_opensandbox_sandbox_from_config():
    config = SandboxConfig(
        provider="opensandbox",
        image="python:3.12",
        runtime_timeout=123,
        sandbox_kwargs={"cpu": 2, "memory_gb": 4, "env": {"FOO": "bar"}},
    )

    sandbox = build_sandbox(config)

    assert SANDBOX_MODULES["opensandbox"] == "uni_agent.sandbox.opensandbox"
    assert isinstance(sandbox, OpenSandboxSandbox)
    assert sandbox.image == "python:3.12"
    assert sandbox.snapshot_id is None
    assert sandbox.runtime_timeout == 123
    assert sandbox.cpu == 2
    assert sandbox.memory_gb == 4
    assert sandbox.env == {"FOO": "bar"}
    assert "_exec" in OpenSandboxSandbox.__dict__
    assert "exec" not in OpenSandboxSandbox.__dict__
    assert OpenSandboxSandbox.exec is Sandbox.exec


@pytest.mark.cpu
@pytest.mark.level0
def test_from_config_accepts_snapshot_id_and_resource_override():
    config = SandboxConfig(
        provider="opensandbox",
        sandbox_kwargs={"snapshot_id": "snap-7", "resource": {"cpu": "8", "memory": "32Gi"}},
    )

    sandbox = build_sandbox(config)

    assert sandbox.image is None
    assert sandbox.snapshot_id == "snap-7"
    assert sandbox.resource == {"cpu": "8", "memory": "32Gi"}


# --------------------------- constructor validation ---------------------------


@pytest.mark.cpu
@pytest.mark.level0
def test_rejects_image_and_snapshot_id_together():
    with pytest.raises(ValueError, match="exactly one of image or snapshot_id"):
        OpenSandboxSandbox(image="python:3.12", snapshot_id="snap-1")


@pytest.mark.cpu
@pytest.mark.level0
def test_requires_image_or_snapshot_id():
    with pytest.raises(ValueError, match="exactly one of image or snapshot_id"):
        OpenSandboxSandbox()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image": "python:3.12", "runtime_timeout": 0},
        {"image": "python:3.12", "runtime_timeout": -5},
        {"image": "python:3.12", "ready_timeout": 0},
        {"image": "python:3.12", "cpu": 0},
        {"image": "python:3.12", "cpu": True},
        {"image": "python:3.12", "memory_gb": 0},
    ],
)
@pytest.mark.cpu
@pytest.mark.level0
def test_rejects_invalid_budgets(kwargs):
    with pytest.raises(ValueError):
        OpenSandboxSandbox(**kwargs)


@pytest.mark.cpu
@pytest.mark.level0
def test_rejects_image_mounts_explicitly():
    with pytest.raises(NotImplementedError, match="image mounts"):
        OpenSandboxSandbox(image="python:3.12", image_mounts=[{"image": "tools:1", "mount_path": "/opt/tool"}])


# --------------------------- start lifecycle ---------------------------


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_start_creates_sandbox_from_image_with_budgets(fake_opensandbox: _FakeOpensandbox):
    sandbox = OpenSandboxSandbox(
        image="python:3.12",
        runtime_timeout=1800,
        ready_timeout=45,
        cpu=2,
        memory_gb=4,
        env={"FOO": "bar"},
        metadata={"job": "swe"},
        entrypoint=["sleep", "infinity"],
        connection_config="connection",
    )

    await sandbox.start()

    call = fake_opensandbox.create_calls[0]
    assert call["image"] == "python:3.12"
    assert "snapshot_id" not in call
    assert call["timeout"] == timedelta(seconds=1800)
    assert call["ready_timeout"] == timedelta(seconds=45)
    assert call["resource"] == {"cpu": "2", "memory": "4Gi"}
    assert call["env"] == {"FOO": "bar"}
    assert call["metadata"] == {"job": "swe"}
    assert call["entrypoint"] == ["sleep", "infinity"]
    assert call["network_policy"] is None
    assert call["volumes"] == []
    assert call["connection_config"] == "connection"
    assert sandbox.sandbox_id == "sbx-fake-1"


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_start_creates_sandbox_from_snapshot_id(fake_opensandbox: _FakeOpensandbox):
    sandbox = OpenSandboxSandbox(snapshot_id="snap-7")

    await sandbox.start()

    call = fake_opensandbox.create_calls[0]
    assert call["snapshot_id"] == "snap-7"
    assert "image" not in call


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_start_resource_override_wins_over_cpu_memory(fake_opensandbox: _FakeOpensandbox):
    sandbox = OpenSandboxSandbox(image="python:3.12", cpu=2, memory_gb=4, resource={"cpu": "8", "memory": "32Gi"})

    await sandbox.start()

    assert fake_opensandbox.create_calls[0]["resource"] == {"cpu": "8", "memory": "32Gi"}


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_start_converts_network_policy_dict(fake_opensandbox: _FakeOpensandbox):
    sandbox = OpenSandboxSandbox(
        image="python:3.12",
        network_policy={"default_action": "deny", "egress": [{"action": "allow", "target": "pypi.org"}]},
    )

    await sandbox.start()

    policy = fake_opensandbox.create_calls[0]["network_policy"]
    assert isinstance(policy, _FakeNetworkPolicy)
    assert policy.default_action == "deny"
    assert [rule.target for rule in policy.egress] == ["pypi.org"]


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_start_passes_network_policy_object_through(fake_opensandbox: _FakeOpensandbox):
    policy = _FakeNetworkPolicy(default_action="allow")
    sandbox = OpenSandboxSandbox(image="python:3.12", network_policy=policy)

    await sandbox.start()

    assert fake_opensandbox.create_calls[0]["network_policy"] is policy


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_start_is_idempotent(fake_opensandbox: _FakeOpensandbox):
    sandbox = OpenSandboxSandbox(image="python:3.12")

    await sandbox.start()
    await sandbox.start()

    assert len(fake_opensandbox.create_calls) == 1


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_start_propagates_create_failure(fake_opensandbox: _FakeOpensandbox):
    fake_opensandbox.create_results.append(RuntimeError("quota exhausted"))
    sandbox = OpenSandboxSandbox(image="python:3.12")

    with pytest.raises(RuntimeError, match="quota exhausted"):
        await sandbox.start()
    assert sandbox._sandbox is None


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_start_failure_after_create_destroys_partial_sandbox(fake_opensandbox: _FakeOpensandbox):
    # A partially set-up sandbox must not survive a failed start(); the handle
    # would otherwise stay attached and a retry would early-return on it.
    created = _FakeSDKSandbox()
    fake_opensandbox.create_results.append(created)
    sandbox = OpenSandboxSandbox(image="python:3.12")

    async def broken_setup(executable_paths):
        raise RuntimeError("executable setup failed")

    sandbox._setup_executable_paths = broken_setup
    with pytest.raises(RuntimeError, match="executable setup failed"):
        await sandbox.start()

    assert sandbox._sandbox is None
    assert created.destroy_calls == 1  # the partial sandbox was reclaimed
    assert fake_opensandbox.create_calls[0]["image"] == "python:3.12"  # create did happen first


# --------------------------- stop lifecycle ---------------------------


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_stop_destroys_and_clears_state(fake_opensandbox: _FakeOpensandbox):
    sandbox = OpenSandboxSandbox(image="python:3.12")
    await sandbox.start()
    sdk_sandbox = sandbox._sandbox

    await sandbox.stop()

    assert sdk_sandbox.destroy_calls == 1
    assert sandbox._sandbox is None
    assert sandbox.sandbox_id is None


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_stop_is_idempotent_before_start():
    sandbox = OpenSandboxSandbox(image="python:3.12")
    await sandbox.stop()  # must not raise


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_stop_swallows_destroy_failure(fake_opensandbox: _FakeOpensandbox):
    sandbox = OpenSandboxSandbox(image="python:3.12")
    await sandbox.start()
    sdk_sandbox = sandbox._sandbox
    sdk_sandbox.destroy_error = RuntimeError("control plane down")

    await sandbox.stop()  # server-side lifetime still reclaims the instance

    assert sdk_sandbox.destroy_calls == 1
    assert sandbox._sandbox is None


# --------------------------- liveness and timeout recognition ---------------------------


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_is_alive_false_before_start():
    assert await OpenSandboxSandbox(image="python:3.12").is_alive() is False


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_is_alive_follows_health_probe():
    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox())

    sdk_sandbox.health_answers = [True]
    assert await sandbox.is_alive() is True
    sdk_sandbox.health_answers = [False]
    assert await sandbox.is_alive() is False
    sdk_sandbox.health_answers = [RuntimeError("connection lost")]
    assert await sandbox.is_alive() is False  # must never raise


@pytest.mark.cpu
@pytest.mark.level0
def test_recognizes_sdk_transport_timeout_by_name():
    sandbox = OpenSandboxSandbox(image="python:3.12")
    timeout_type = type("SandboxTimeoutException", (Exception,), {})

    assert sandbox._is_timeout_error(timeout_type("retry deadline")) is True
    assert sandbox._is_timeout_error(TimeoutError()) is True
    assert sandbox._is_timeout_error(RuntimeError("other")) is False


# --------------------------- exec result conversion ---------------------------


def _setup_exec(sandbox: OpenSandboxSandbox, sdk_sandbox: _FakeSDKSandbox, execution) -> list[dict]:
    sdk_sandbox.attach_commands([execution] if not isinstance(execution, list) else execution)
    return sdk_sandbox.runs


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_exec_preserves_zero_and_nonzero_exit_codes():
    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox())
    sdk_sandbox.attach_commands(
        [
            _execution(0, stdout="out one\n", stderr=""),
            _execution(2, stdout="partial out\n", stderr="boom\n"),
        ]
    )

    first = await sandbox.exec(["echo", "hi"])
    second = await sandbox.exec(["false"])

    assert first == ExecResult(exit_code=0, stdout="out one\n", stderr="")
    assert second == ExecResult(exit_code=2, stdout="partial out\n", stderr="boom\n")
    assert sdk_sandbox.runs[0]["command"] == ["echo", "hi"]


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_exec_forwards_opts_to_sdk():
    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox())
    _setup_exec(sandbox, sdk_sandbox, _execution(0))

    await sandbox.exec(["pip", "install", "requests"], timeout=5, workdir="/workspace", env={"A": "1"})

    opts = sdk_sandbox.runs[0]["opts"]
    assert isinstance(opts, _FakeRunCommandOpts)
    assert opts.timeout == timedelta(seconds=5)
    assert opts.working_directory == "/workspace"
    assert opts.envs == {"A": "1"}


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_exec_without_budget_omits_timeout():
    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox())
    _setup_exec(sandbox, sdk_sandbox, _execution(0))

    await sandbox.exec(["echo", "hi"])
    assert sdk_sandbox.runs[0]["opts"].timeout is None

    await sandbox.exec(["echo", "hi"], timeout=0)  # non-positive budget means "no deadline"
    assert sdk_sandbox.runs[1]["opts"].timeout is None


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_exec_maps_budgeted_signal_termination_to_timeout():
    # The server kills the command at the caller deadline; execd reports -1.
    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox())
    execution = _execution(-1, stdout="", stderr="", error=SimpleNamespace(value="signal: killed"))
    _setup_exec(sandbox, sdk_sandbox, execution)

    result = await sandbox.exec(["sleep", "999"], timeout=4)

    # Shared policy: timeout -> exit_code -1 with the error on stderr.
    assert result.exit_code == -1
    assert "timed out" in result.stderr


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_exec_unbudgeted_signal_termination_downgrades_while_alive():
    # No caller budget: the signal came from elsewhere (e.g. OOM) -> 127 while alive.
    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox())
    execution = _execution(-1, stdout="", stderr="", error=SimpleNamespace(value="signal: killed"))
    _setup_exec(sandbox, sdk_sandbox, execution)

    result = await sandbox.exec(["sleep", "999"])

    assert result.exit_code == 127
    assert "terminated by a signal" in result.stderr


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_exec_maps_legacy_kill_shape_to_timeout_when_budgeted():
    # Servers predating error.exit_code omit the code and carry "signal: killed".
    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox())
    execution = _execution(None, stdout="", stderr="", error=SimpleNamespace(value="signal: killed"))
    _setup_exec(sandbox, sdk_sandbox, execution)

    result = await sandbox.exec(["sleep", "999"], timeout=4)

    assert result.exit_code == -1
    assert "timed out" in result.stderr


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_exec_surfaces_lost_results_as_errors():
    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox())
    sdk_sandbox.attach_commands(
        [
            _execution(None, stdout="", stderr="", error=SimpleNamespace(value='exec: "nope" not found')),
            _execution(None, stdout="", stderr="", error=None),
        ]
    )

    first = await sandbox.exec(["nope"])
    second = await sandbox.exec(["gone"])

    assert first.exit_code == 127
    assert "result lost" in first.stderr
    assert second.exit_code == 127
    assert "no completion event" in second.stderr


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_exec_reraises_when_sandbox_died():
    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox())
    execution = _execution(-1, stdout="", stderr="", error=SimpleNamespace(value="signal: killed"))
    _setup_exec(sandbox, sdk_sandbox, execution)
    sdk_sandbox.health_answers = [False]  # dead: the error must escape, not become 127

    with pytest.raises(RuntimeError, match="terminated by a signal"):
        await sandbox.exec(["echo", "hi"])


# --------------------------- files ---------------------------


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_read_file_returns_bytes():
    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox())
    reads: list[str] = []

    async def read_bytes(path):
        reads.append(path)
        return b"\x00read\xff"

    sdk_sandbox.files = SimpleNamespace(read_bytes=read_bytes)
    assert await sandbox.read_file("/tmp/data.bin") == b"\x00read\xff"
    assert reads == ["/tmp/data.bin"]


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_write_file_encodes_strings_and_keeps_bytes():
    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox())
    writes: list[tuple[str, object]] = []

    async def write_file(path, data, **kwargs):
        writes.append((path, data))

    sdk_sandbox.files = SimpleNamespace(read_bytes=None, write_file=write_file)
    await sandbox.write_file("/tmp/a.txt", "héllo")
    await sandbox.write_file("/tmp/b.bin", b"\x00byte\xff")

    assert writes == [
        ("/tmp/a.txt", "héllo".encode()),
        ("/tmp/b.bin", b"\x00byte\xff"),
    ]


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_upload_file_streams_local_file(tmp_path: Path):
    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox())
    source = tmp_path / "payload.bin"
    source.write_bytes(b"\x11upload\x22")
    received: list[tuple[str, object]] = []

    async def write_file(path, data, **kwargs):
        received.append((path, data.read() if hasattr(data, "read") else data))

    sdk_sandbox.files = SimpleNamespace(read_bytes=None, write_file=write_file)
    await sandbox.upload_file(source, "/remote/payload.bin")

    remote_path, payload = received[0]
    assert remote_path == "/remote/payload.bin"
    assert payload == b"\x11upload\x22"


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_download_file_creates_local_parents(tmp_path: Path):
    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox())

    async def read_bytes(path):
        return b"\x33download\x44"

    sdk_sandbox.files = SimpleNamespace(read_bytes=read_bytes)
    destination = tmp_path / "nested" / "out.bin"
    await sandbox.download_file("/remote/out.bin", destination)

    assert destination.read_bytes() == b"\x33download\x44"


# --------------------------- ports ---------------------------


@pytest.mark.asyncio
@pytest.mark.cpu
@pytest.mark.level0
async def test_expose_port_uses_connection_protocol_for_scheme_less_endpoints():
    # Scheme-less endpoints take the connection protocol (http by default), not https.
    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox(protocol="http"))
    sdk_sandbox.endpoint_answers = {8080: SimpleNamespace(endpoint="8080.example.svc.cluster.local", headers={})}
    assert await sandbox.expose_port(8080) == "http://8080.example.svc.cluster.local"

    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox(protocol="https"))
    sdk_sandbox.endpoint_answers = {8080: SimpleNamespace(endpoint="8080.example.svc.cluster.local", headers={})}
    assert await sandbox.expose_port(8080) == "https://8080.example.svc.cluster.local"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_expose_port_preserves_explicit_schemes_and_returns_raw_endpoint():
    sandbox = OpenSandboxSandbox(image="python:3.12")
    sdk_sandbox = _attach(sandbox, _FakeSDKSandbox(protocol="http"))
    sdk_sandbox.endpoint_answers = {
        9000: SimpleNamespace(endpoint="https://9000.example.svc.cluster.local", headers={"X-Route": "a"}),
    }

    assert await sandbox.expose_port(9000) == "https://9000.example.svc.cluster.local"
    endpoint = await sandbox.get_port_endpoint(9000)
    assert endpoint.headers == {"X-Route": "a"}
