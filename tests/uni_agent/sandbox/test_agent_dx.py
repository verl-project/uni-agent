from __future__ import annotations

import asyncio
import shlex
import sys
import threading
from types import SimpleNamespace

import pytest

from uni_agent.sandbox import ExecResult, Sandbox, SandboxConfig, agent_dx, build_sandbox

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def result(code=0, stdout="", stderr="", status="SUCCEEDED", **kwargs):
    return SimpleNamespace(exit_code=code, stdout=stdout, stderr=stderr, status=status, **kwargs)


@pytest.fixture
def sdk(monkeypatch):
    state = SimpleNamespace(
        creates=[],
        deletes=[],
        commands=[],
        shell_commands=[],
        shell_creates=[],
        shell_kills=0,
        kills=0,
        closes=0,
        results=[result(stdout="hello")],
        shell_result=result(),
        kill_error=None,
        delete_error=None,
        create_error=None,
        command_error=None,
        create_started=threading.Event(),
        release_create=None,
        files={},
        transfers=[],
    )

    class Value:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Connection(Value):
        @classmethod
        def from_env(cls, **kwargs):
            return cls(server_address="adx.example:443", token="test-token", **kwargs)

    class Handle:
        async def wait_async(self, timeout=None):
            await asyncio.sleep(0)
            if state.command_error:
                raise state.command_error
            return state.results.pop(0) if len(state.results) > 1 else state.results[0]

    class Commands:
        def run(self, command, **kwargs):
            state.commands.append((command, kwargs))
            return Handle()

    class Shell:
        async def run(self, command, **kwargs):
            state.shell_commands.append((command, kwargs))
            return state.shell_result

        async def kill(self):
            state.shell_kills += 1

    class Shells:
        async def create(self, **kwargs):
            state.shell_creates.append(kwargs)
            return Shell()

    class Files:
        def read(self, path, *, format):
            assert format == "bytes"
            return state.files[path]

        def write(self, path, data):
            state.files[path] = data

        def copy_from_local(self, source, target):
            state.transfers.append(("upload", source, target))

        def copy_to_local(self, source, target):
            state.transfers.append(("download", source, target))

    class Environment:
        def __init__(self, **kwargs):
            state.creates.append(kwargs)
            state.create_started.set()
            if state.release_create is not None:
                assert state.release_create.wait(5)
            if state.create_error:
                raise state.create_error
            self.environment_id = "env-test"
            self.commands = Commands()
            self.shells = Shells()
            self.files = Files()

        def kill(self):
            state.kills += 1
            if state.kill_error:
                raise state.kill_error

        def close(self):
            state.closes += 1

        @classmethod
        def delete(cls, environment_id, *, connection):
            state.deletes.append(environment_id)
            if state.delete_error:
                raise state.delete_error

        def is_running(self):
            return True

        def get_port_url(self, port):
            return f"https://adx.example/env-test/{port}"

        def get_tunnel_url(self):
            return "http://127.0.0.1:38197"

        def get_port_auth_headers(self):
            return {"X-Auth": "test-token"}

    module = SimpleNamespace(
        Environment=Environment,
        ConnectionConfig=Connection,
        Mount=Value,
        S3Config=Value,
        PortForwarding=Value,
        SDK_CAPABILITIES={"local-close", "explicit-connection-config", "recoverable-commands", "tunnel-proxy-port"},
    )
    monkeypatch.setattr(agent_dx, "_load_sdk", lambda: module)
    state.module = module
    return state


def test_registry_stays_lazy_and_consumes_common_config(sdk):
    sandbox = build_sandbox(SandboxConfig(provider="agent_dx", image="task:image", runtime_timeout=42))
    assert isinstance(sandbox, agent_dx.AgentDxSandbox)
    assert sandbox.runtime_timeout == 42
    assert not sdk.creates
    assert sandbox.exec.__func__ is Sandbox.exec


def test_missing_sdk_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "adx_environment", None)
    with pytest.raises(ImportError, match="adx-environment"):
        agent_dx._load_sdk()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image": ""},
        {"runtime_timeout": 0},
        {"runtime_timeout": float("nan")},
        {"cpu": 0},
        {"memory": -1},
        {"cpu_limit": 1},
        {"proxy_port": 1},
        {"proxy_port": True},
        {"port_forwardings": [8080, 8080]},
        {"port_forwardings": [50090]},
        {"proxy_port": 38197, "upstream": "localhost:8000", "port_forwardings": [38196]},
        {"add_to_path": "bad"},
        {"env": {"BAD-NAME": "value"}},
        {"cwd": "relative"},
        {"name": " "},
    ],
)
def test_invalid_configuration_fails_before_network(kwargs, sdk):
    with pytest.raises((ValueError, TypeError)):
        agent_dx.AgentDxSandbox(**{"image": "task:image", **kwargs})
    assert not sdk.creates


def test_unknown_sdk_options_are_rejected():
    with pytest.raises(TypeError, match="detached"):
        agent_dx.AgentDxSandbox(image="task:image", detached=True)


@pytest.mark.asyncio
async def test_mount_targets_are_unique_after_path_normalization(sdk):
    sandbox = agent_dx.AgentDxSandbox(
        image="task:image",
        mounts=[{"target": "/tools/", "image_url": "tools:1"}, {"target": "/tools", "image_url": "tools:2"}],
    )
    with pytest.raises(ValueError, match="unique target"):
        await sandbox.start()
    assert not sdk.creates


@pytest.mark.asyncio
async def test_start_maps_resources_mounts_connection_and_explicit_name(sdk):
    connection = sdk.module.ConnectionConfig(server_address="private", token="explicit-token")
    sandbox = build_sandbox(
        SandboxConfig(
            provider="agent_dx",
            image="task:image",
            image_mounts=[{"image": "tools:image", "mount_path": "/opt/tools"}],
            sandbox_kwargs={
                "connection": connection,
                "name": "explicit-name",
                "mounts": [{"target": "/data", "s3_config": {"endpoint": "s3", "bucket": "b", "object": "o"}}],
                "env": {"FOO": "bar"},
                "upstream": "127.0.0.1:8000",
                "proxy_port": 38197,
                "port_forwardings": [8080],
            },
        )
    )
    await sandbox.start()
    await sandbox.start()
    create = sdk.creates[0]
    assert len(sdk.creates) == 1
    assert create["cpu"] == 2000 and create["memory"] == 4096
    assert create["cpu_limit"] == 8000 and create["mem_limit"] == 12288
    assert create["name"] == "explicit-name" and create["connection"] is connection
    assert create["env"] == {"FOO": "bar"}
    assert create["upstream"] == "127.0.0.1:8000" and create["proxy_port"] == 38197
    assert create["mounts"][0].s3_config.bucket == "b"
    assert create["mounts"][1].image_url == "tools:image"
    assert "runtime_timeout" not in create and create["cwd"] is None
    await sandbox.stop()
    await sandbox.stop()
    assert sdk.kills == 1


@pytest.mark.asyncio
async def test_create_failure_never_resubmits_on_same_instance(sdk):
    sdk.create_error = RuntimeError("lost create response")
    sandbox = agent_dx.AgentDxSandbox(image="task:image")
    for _ in range(2):
        with pytest.raises(agent_dx.AgentDxSandboxError, match="CREATE_FAILED"):
            await sandbox.start()
    assert len(sdk.creates) == 1


@pytest.mark.asyncio
async def test_executable_setup_failure_rolls_back_environment(sdk, monkeypatch):
    sandbox = agent_dx.AgentDxSandbox(image="task:image", executable_paths={"tool": "/opt/tool/bin/tool"})

    async def fail(paths):
        assert paths == {"tool": "/opt/tool/bin/tool"}
        raise RuntimeError("configured executable is unavailable")

    monkeypatch.setattr(sandbox, "_setup_executable_paths", fail)
    with pytest.raises(RuntimeError, match="configured executable"):
        await sandbox.start()
    assert sdk.kills == 1
    assert not await sandbox.is_alive()


@pytest.mark.asyncio
async def test_name_prefix_is_used_only_when_explicit_name_is_missing(sdk, monkeypatch):
    monkeypatch.setenv("SANDBOX_NAME_PREFIX", "fixture-")
    sandbox = agent_dx.AgentDxSandbox(image="task:image")
    await sandbox.start()
    assert sdk.creates[0]["name"].startswith("fixture-")
    await sandbox.stop()


@pytest.mark.asyncio
async def test_missing_recoverable_command_capability_fails_before_create(sdk):
    sdk.module.SDK_CAPABILITIES = {"local-close"}
    sandbox = agent_dx.AgentDxSandbox(image="task:image")
    with pytest.raises(RuntimeError, match="recoverable commands"):
        await sandbox.start()
    assert not sdk.creates


@pytest.mark.asyncio
async def test_cancelled_create_collects_late_environment_and_deletes_it(sdk):
    sdk.release_create = threading.Event()
    sandbox = agent_dx.AgentDxSandbox(image="task:image")
    task = asyncio.create_task(sandbox.start())
    try:
        assert await asyncio.to_thread(sdk.create_started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        sdk.release_create.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sdk.kills == 1
    assert not await sandbox.is_alive()


@pytest.mark.asyncio
async def test_delete_failure_uses_public_delete_and_retains_id_for_retry(sdk):
    sandbox = agent_dx.AgentDxSandbox(image="task:image")
    await sandbox.start()
    sdk.kill_error = RuntimeError("kill failed after closing client")
    sdk.delete_error = RuntimeError("delete unavailable")
    with pytest.raises(agent_dx.AgentDxSandboxError, match="CLEANUP_FAILED"):
        await sandbox.stop()
    assert sandbox.environment_id == "env-test"
    sdk.delete_error = None
    await sandbox.stop()
    assert sdk.deletes == ["env-test", "env-test"]


@pytest.mark.asyncio
async def test_runtime_timeout_deletes_and_prevents_successful_context_exit(sdk):
    sandbox = agent_dx.AgentDxSandbox(image="task:image", runtime_timeout=0.01)
    with pytest.raises(agent_dx.AgentDxSandboxError, match="RUNTIME_TIMEOUT"):
        async with sandbox.entered(retry=1):
            for _ in range(100):
                if sdk.kills:
                    break
                await asyncio.sleep(0.005)
            assert sdk.kills == 1


@pytest.mark.asyncio
async def test_command_preserves_argv_nonzero_cwd_env_and_fractional_budget(sdk):
    sdk.results = [result(2, b"partial\xff", "request timed out after 5 seconds", status="FAILED")]
    sandbox = agent_dx.AgentDxSandbox(image="task:image", cwd="/work dir", add_to_path=["/opt/tool dir"])
    await sandbox.start()
    value = await sandbox.exec(["printf", "a b", "$(literal)"], timeout=0.25, env={"FOO": "bar"})
    command, kwargs = sdk.commands[-1]
    assert command == "export PATH='/opt/tool dir':\"${PATH:-}\"; exec printf 'a b' '$(literal)'"
    assert kwargs["timeout"] == 1 and kwargs["cwd"] == "/work dir" and kwargs["envs"] == {"FOO": "bar"}
    assert kwargs["background"] is True and kwargs["command_id"].startswith("uni-agent-")
    assert value == ExecResult(2, "partial\ufffd", "request timed out after 5 seconds")
    await sandbox.stop()


@pytest.mark.asyncio
async def test_running_wait_result_observes_same_command_without_resubmitting(sdk):
    sdk.results = [result(None, status="RUNNING", error_code="WAIT_TIMEOUT"), result(stdout="done")]
    sandbox = agent_dx.AgentDxSandbox(image="task:image")
    await sandbox.start()
    assert (await sandbox.exec(["echo", "done"])).stdout == "done"
    assert len(sdk.commands) == 1
    await sandbox.stop()


@pytest.mark.asyncio
async def test_authoritative_command_timeout_uses_shared_error_policy(sdk):
    sdk.results = [result(-1, stderr="deadline", status="TIMED_OUT")]
    sandbox = agent_dx.AgentDxSandbox(image="task:image")
    await sandbox.start()
    value = await sandbox.exec(["sleep", "30"], timeout=1)
    assert value.exit_code == -1 and "timed out" in value.stderr
    assert await sandbox.is_alive()
    await sandbox.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_result", [result(truncated=True), result(None), result(-1, status="KILLED")])
async def test_incomplete_command_results_are_fatal_even_when_environment_is_alive(sdk, bad_result):
    sdk.results = [bad_result]
    sandbox = agent_dx.AgentDxSandbox(image="task:image")
    await sandbox.start()
    with pytest.raises(agent_dx.AgentDxSandboxError):
        await sandbox.exec(["example"])
    with pytest.raises(agent_dx.AgentDxSandboxError):
        sandbox.raise_if_unavailable()
    await sandbox.stop()


@pytest.mark.asyncio
async def test_sdk_fault_is_not_downgraded_to_exit_127(sdk):
    sdk.command_error = RuntimeError("lost command result")
    sandbox = agent_dx.AgentDxSandbox(image="task:image")
    await sandbox.start()
    with pytest.raises(agent_dx.AgentDxSandboxError) as error:
        await sandbox.exec(["example"])
    assert error.value.command_id and error.value.environment_id == "env-test"
    await sandbox.stop()


@pytest.mark.asyncio
async def test_transport_timeout_is_fatal_instead_of_confirmed_command_timeout(sdk):
    sdk.command_error = TimeoutError("connection timed out, command outcome unknown")
    sandbox = agent_dx.AgentDxSandbox(image="task:image")
    await sandbox.start()
    with pytest.raises(agent_dx.AgentDxSandboxError, match="COMMAND_UNAVAILABLE"):
        await sandbox.exec(["example"])
    await sandbox.stop()


@pytest.mark.asyncio
async def test_caught_infrastructure_fault_cannot_return_success_from_context(sdk):
    sdk.command_error = RuntimeError("lost command result")
    sandbox = agent_dx.AgentDxSandbox(image="task:image")
    with pytest.raises(agent_dx.AgentDxSandboxError):
        async with sandbox:
            try:
                await sandbox.exec(["example"])
            except agent_dx.AgentDxSandboxError:
                pass
    assert sdk.kills == 1


@pytest.mark.asyncio
async def test_cancelled_command_marks_environment_unusable_and_deletes_it(sdk):
    sdk.results = [result(None, status="RUNNING", error_code="WAIT_TIMEOUT")]
    sandbox = agent_dx.AgentDxSandbox(image="task:image")
    await sandbox.start()
    task = asyncio.create_task(sandbox.exec(["sleep", "30"]))
    while not sdk.commands:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sdk.kills == 1
    with pytest.raises(agent_dx.AgentDxSandboxError):
        sandbox.raise_if_unavailable()


@pytest.mark.asyncio
async def test_shell_initializes_quoted_cwd_env_and_path(sdk):
    sandbox = agent_dx.AgentDxSandbox(image="task:image", cwd="/base", add_to_path=["/opt/tool"])
    await sandbox.start()
    shell = await sandbox.open_shell(cwd="/work dir", env={"VALUE": "quote ' and $(literal)"})
    assert sdk.shell_creates == [{"cwd": "/", "envs": None}]
    init = sdk.shell_commands[0][0]
    assert "cd -- '/work dir'" in init
    assert f"export VALUE={shlex.quote('quote ' + chr(39) + ' and $(literal)')}" in init
    assert 'export PATH=/opt/tool:"${PATH:-}"' in init
    sdk.shell_result = result(stdout="saved cwd")
    assert await shell.run("pwd") == ExecResult(0, "saved cwd", "")
    await shell.close()
    await shell.close()
    assert sdk.shell_kills == 1
    await sandbox.stop()


@pytest.mark.asyncio
async def test_shell_timeout_destroys_environment_and_is_not_returned_as_success(sdk):
    sandbox = agent_dx.AgentDxSandbox(image="task:image")
    await sandbox.start()
    shell = await sandbox.open_shell()
    sdk.shell_result = result(-1, stderr="Command timed out after 1s")
    with pytest.raises(agent_dx.AgentDxSandboxError, match="SHELL_STATE_LOST"):
        await shell.run("sleep 30", timeout=1)
    assert sdk.kills == 1


@pytest.mark.asyncio
async def test_native_files_and_ports_use_public_sdk_api(sdk, tmp_path):
    sandbox = agent_dx.AgentDxSandbox(image="task:image", port_forwardings=[8080])
    await sandbox.start()
    await sandbox.write_file("/tmp/text", "hello")
    await sandbox.write_file("/tmp/data", b"\0data\xff")
    assert await sandbox.read_file("/tmp/text") == b"hello"
    assert await sandbox.read_file("/tmp/data") == b"\0data\xff"
    source = tmp_path / "source"
    source.write_bytes(b"data")
    await sandbox.upload_file(source, "/tmp/source")
    await sandbox.download_file("/tmp/source", source)
    assert sdk.transfers == [("upload", str(source), "/tmp/source"), ("download", "/tmp/source", str(source))]
    assert await sandbox.expose_port(8080) == "https://adx.example/env-test/8080"
    assert sandbox.get_tunnel_url() == "http://127.0.0.1:38197"
    assert sandbox.get_port_auth_headers() == {"X-Auth": "test-token"}
    await sandbox.stop()


@pytest.mark.asyncio
async def test_cancelled_file_write_finishes_before_environment_cleanup(sdk):
    started = threading.Event()
    release = threading.Event()
    sandbox = agent_dx.AgentDxSandbox(image="task:image")
    await sandbox.start()

    def write(path, data):
        started.set()
        assert release.wait(5)
        assert sdk.kills == 0
        sdk.files[path] = data

    sandbox._environment.files.write = write
    task = asyncio.create_task(sandbox.write_file("/tmp/data", b"content"))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sdk.files["/tmp/data"] == b"content" and sdk.kills == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
async def test_invalid_command_budget_raises_without_poisoning_environment(sdk, timeout):
    sandbox = agent_dx.AgentDxSandbox(image="task:image")
    await sandbox.start()
    with pytest.raises(ValueError, match="timeout"):
        await sandbox.exec(["example"], timeout=timeout)
    assert not sdk.commands
    assert await sandbox.is_alive()
    await sandbox.stop()
