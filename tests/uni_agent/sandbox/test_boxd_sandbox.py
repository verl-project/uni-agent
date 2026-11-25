from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import types
import uuid
from types import SimpleNamespace

import pytest

from uni_agent.sandbox.base import Sandbox, SandboxConfig
from uni_agent.sandbox.boxd import BoxdSandbox
from uni_agent.sandbox.registry import SANDBOX_MODULES, build_sandbox


class FakeBoxdError(Exception):
    pass


class FakeNotFoundError(FakeBoxdError):
    pass


class FakeFiles:
    def __init__(self, owner: FakeMachines) -> None:
        self.owner = owner
        self.data: dict[str, bytes] = {}

    async def upload(self, machine_id, path, data, *, region=None):
        self.owner.calls.append(("upload", machine_id, path, len(data)))
        self.data[path] = bytes(data)
        return len(data)

    async def download(self, machine_id, path, *, region=None):
        self.owner.calls.append(("download", machine_id, path))
        return self.data[path]


class FakeMachines:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.files = FakeFiles(self)
        self.status = "running"
        self.exec_results: dict[str, SimpleNamespace] = {}
        self.exec_error: BaseException | None = None
        self.exec_delay = 0.0
        self.names: set[str] = set()
        self.dirs: set[str] = set()
        self.create_blocks = False

    async def create(self, name, **kwargs):
        self.calls.append(("create", name, kwargs))
        self.names.add(name)
        if self.create_blocks:
            await asyncio.Event().wait()
        return SimpleNamespace(id="vm-1", name=name)

    async def wait_until_ready(self, machine_id, *, timeout=None, region=None):
        self.calls.append(("ready", machine_id, timeout))

    async def exec(self, machine_id, command, *, env=None, cwd=None, timeout=None, region=None):
        self.calls.append(("exec", machine_id, list(command), env, cwd, timeout))
        if self.exec_delay:
            await asyncio.sleep(self.exec_delay)
        if self.exec_error is not None:
            raise self.exec_error
        if command[-3:-1] == ["test", "-d"]:
            return SimpleNamespace(exit_code=0 if command[-1] in self.dirs else 1, stdout="", stderr="")
        key = " ".join(command[:3])
        return self.exec_results.get(key, SimpleNamespace(exit_code=0, stdout="", stderr=""))

    async def get(self, machine_id, *, region=None):
        self.calls.append(("get", machine_id))
        if self.status == "gone" or (machine_id != "vm-1" and machine_id not in self.names):
            raise FakeNotFoundError(machine_id)
        return SimpleNamespace(id="vm-1", status=self.status)

    async def delete(self, machine_id, *, region=None):
        self.calls.append(("delete", machine_id))


class FakeAsyncBoxd:
    instances: list[FakeAsyncBoxd] = []

    def __init__(self, *, api_key=None, region=None):
        self.api_key = api_key
        self.region = region
        self.machines = FakeMachines()
        self.closed = False
        FakeAsyncBoxd.instances.append(self)

    async def close(self):
        self.closed = True


@pytest.fixture
def fake_boxd(monkeypatch):
    module = types.ModuleType("boxd")
    module.AsyncBoxd = FakeAsyncBoxd
    errors = types.ModuleType("boxd.errors")
    errors.BoxdError = FakeBoxdError
    errors.NotFoundError = FakeNotFoundError
    module.errors = errors
    monkeypatch.setitem(sys.modules, "boxd", module)
    monkeypatch.setitem(sys.modules, "boxd.errors", errors)
    monkeypatch.setenv("BOXD_API_KEY", "test-key")
    monkeypatch.setattr(uuid, "uuid4", lambda: SimpleNamespace(hex="0123456789abcdef"))
    FakeAsyncBoxd.instances.clear()
    return FakeAsyncBoxd


CONTAINER = "uni-agent-0123456789ab"
STAGED = "/tmp/uni-agent-xfer-0123456789abcdef"


def _execs(machines: FakeMachines) -> list[list[str]]:
    return [call[2] for call in machines.calls if call[0] == "exec"]


@pytest.mark.cpu
@pytest.mark.level0
def test_registry_builds_boxd_sandbox_from_config():
    config = SandboxConfig(
        provider="boxd",
        image="python:3.12",
        runtime_timeout=900,
        sandbox_kwargs={"region": "fra", "vcpu": 4, "memory": "16G", "org": "my-org"},
    )

    sandbox = build_sandbox(config)

    assert SANDBOX_MODULES["boxd"] == "uni_agent.sandbox.boxd"
    assert isinstance(sandbox, BoxdSandbox)
    assert sandbox.image == "python:3.12"
    assert sandbox.runtime_timeout == 900
    assert (sandbox.region, sandbox.vcpu, sandbox.memory, sandbox.org) == ("fra", 4, "16G", "my-org")
    assert "_exec" in BoxdSandbox.__dict__
    assert BoxdSandbox.exec is Sandbox.exec


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"runtime_timeout": 0}, "runtime_timeout"),
        ({"image": " "}, "image must be non-empty"),
        ({"pull_policy": "sometimes"}, "pull_policy"),
        ({"auto_suspend_timeout": -1}, "auto_suspend_timeout"),
        ({"ready_timeout": 0}, "ready_timeout"),
        ({"ready_timeout": None}, "ready_timeout"),
        ({"executable_paths": {"tool": "/opt/tool/bin/tool"}}, "executable_paths require an image"),
        ({"from_snapshot": "warm", "vcpu": 2}, "drop vcpu and memory"),
        ({"image_mounts": [SimpleNamespace(image="tool:1", mount_path="/opt/tool")]}, "require an image"),
    ],
)
def test_rejects_invalid_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BoxdSandbox(**kwargs)


@pytest.mark.cpu
@pytest.mark.level0
def test_start_without_api_key_fails_clearly(fake_boxd, monkeypatch):
    monkeypatch.delenv("BOXD_API_KEY")
    sandbox = BoxdSandbox()

    with pytest.raises(ValueError, match="BOXD_API_KEY"):
        asyncio.run(sandbox.start())

    assert fake_boxd.instances == []


@pytest.mark.cpu
@pytest.mark.level0
def test_start_creates_an_isolated_machine_with_a_lifetime(fake_boxd):
    sandbox = BoxdSandbox(
        runtime_timeout=1800.5,
        region="nyc",
        org="my-org",
        from_snapshot="warm",
        name_prefix="rl",
        labels={"run": "7"},
    )
    asyncio.run(sandbox.start())

    client = fake_boxd.instances[0]
    assert (client.api_key, client.region) == ("test-key", "nyc")
    create, ready = client.machines.calls[:2]
    assert create[0] == "create"
    assert create[1].startswith("rl-") and len(create[1]) == len("rl-") + 12
    assert create[2] == {
        "org": "my-org",
        "isolated": True,
        "labels": {"app": "uni-agent", "run": "7"},
        "auto_suspend_timeout": 0,
        "delete_after": 1801,
        "region": "nyc",
        "from_snapshot": "warm",
    }
    assert ready == ("ready", "vm-1", 180.0)
    assert sandbox.machine_id == "vm-1"
    # No image: commands run on the machine itself, so no container is started.
    assert _execs(client.machines) == []


@pytest.mark.cpu
@pytest.mark.level0
def test_api_key_option_wins_over_environment_and_size_is_forwarded(fake_boxd):
    asyncio.run(BoxdSandbox(api_key="explicit", vcpu=4, memory="16G").start())
    client = fake_boxd.instances[0]
    assert client.api_key == "explicit"
    assert {key: client.machines.calls[0][2][key] for key in ("vcpu", "memory")} == {"vcpu": 4, "memory": "16G"}


@pytest.mark.cpu
@pytest.mark.level0
def test_start_with_image_pulls_and_runs_a_container_with_mounts(fake_boxd, monkeypatch):
    sandbox = BoxdSandbox(
        image="python:3.12",
        runtime_timeout=600,
        run_args=["--network", "none"],
        image_mounts=[SimpleNamespace(image="tool:1", mount_path="/opt/tool")],
    )

    def fake_create_client():
        client = FakeAsyncBoxd(api_key="k")
        client.machines.exec_results["docker image inspect"] = SimpleNamespace(exit_code=1, stdout="", stderr="")
        return client

    monkeypatch.setattr(sandbox, "_make_client", fake_create_client)
    asyncio.run(sandbox.start())

    assert _execs(fake_boxd.instances[0].machines) == [
        ["docker", "image", "inspect", "python:3.12"],
        ["docker", "pull", "-q", "python:3.12"],
        ["docker", "image", "inspect", "tool:1"],
        ["docker", "pull", "-q", "tool:1"],
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER,
            "--pull",
            "never",
            "--entrypoint",
            "sleep",
            "--mount",
            "type=image,source=tool:1,destination=/opt/tool",
            "--network",
            "none",
            "python:3.12",
            "600",
        ],
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_pull_policy_never_fails_without_a_local_image(fake_boxd, monkeypatch):
    sandbox = BoxdSandbox(image="python:3.12", pull_policy="never")

    def fake_create_client():
        client = FakeAsyncBoxd(api_key="k")
        client.machines.exec_results["docker image inspect"] = SimpleNamespace(exit_code=1, stdout="", stderr="")
        return client

    monkeypatch.setattr(sandbox, "_make_client", fake_create_client)
    with pytest.raises(RuntimeError, match="not available"):
        asyncio.run(sandbox.start())


@pytest.mark.cpu
@pytest.mark.level0
def test_failed_container_start_is_reported(fake_boxd, monkeypatch):
    sandbox = BoxdSandbox(image="python:3.12", pull_policy="always")

    def fake_create_client():
        client = FakeAsyncBoxd(api_key="k")
        client.machines.exec_results["docker pull -q"] = SimpleNamespace(exit_code=1, stdout="", stderr="denied")
        return client

    monkeypatch.setattr(sandbox, "_make_client", fake_create_client)
    with pytest.raises(RuntimeError, match="denied"):
        asyncio.run(sandbox.start())
    # The machine id is kept so the caller's cleanup stop() can delete it.
    assert sandbox.machine_id == "vm-1"


@pytest.mark.cpu
@pytest.mark.level0
def test_exec_in_container_passes_env_by_name_and_bounds_the_command(fake_boxd):
    sandbox = BoxdSandbox(image="python:3.12")

    async def run():
        await sandbox.start()
        machines = fake_boxd.instances[0].machines
        machines.calls.clear()
        result = await sandbox.exec(["python", "-c", "print(1)"], timeout=12.5, workdir="/repo", env={"A": "one"})
        return machines, result

    machines, result = asyncio.run(run())

    assert result.exit_code == 0
    assert machines.calls == [
        (
            "exec",
            "vm-1",
            [
                "docker",
                "exec",
                "--workdir",
                "/repo",
                "--env",
                "A",
                CONTAINER,
                "timeout",
                "-k",
                "10",
                "15",
                "python",
                "-c",
                "print(1)",
            ],
            {"A": "one"},
            None,
            12.5,
        )
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_exec_on_machine_uses_native_cwd_and_env(fake_boxd):
    sandbox = BoxdSandbox()

    async def run():
        await sandbox.start()
        machines = fake_boxd.instances[0].machines
        machines.exec_results["ls -la"] = SimpleNamespace(exit_code=2, stdout="out", stderr=b"err")
        return machines, await sandbox.exec(["ls", "-la"], workdir="/tmp", env={"B": "two"})

    machines, result = asyncio.run(run())

    assert (result.exit_code, result.stdout, result.stderr) == (2, "out", "err")
    assert machines.calls[-1] == ("exec", "vm-1", ["ls", "-la"], {"B": "two"}, "/tmp", None)


@pytest.mark.cpu
@pytest.mark.level0
def test_exec_timeout_becomes_the_shared_timeout_result(fake_boxd):
    sandbox = BoxdSandbox()

    async def run():
        await sandbox.start()
        machines = fake_boxd.instances[0].machines
        machines.exec_delay = 0.05
        machines.exec_error = FakeBoxdError("exec timed out after 0.01s")
        return await sandbox.exec(["sleep", "5"], timeout=0.01)

    result = asyncio.run(run())

    assert result.exit_code == -1
    assert "timed out" in result.stderr


@pytest.mark.cpu
@pytest.mark.level0
def test_exec_error_on_a_live_machine_is_a_command_error(fake_boxd):
    sandbox = BoxdSandbox()

    async def run():
        await sandbox.start()
        fake_boxd.instances[0].machines.exec_error = FakeBoxdError("transport hiccup")
        return await sandbox.exec(["true"], timeout=30)

    result = asyncio.run(run())

    assert (result.exit_code, result.stderr) == (127, "transport hiccup")


@pytest.mark.cpu
@pytest.mark.level0
def test_exec_error_on_a_gone_machine_raises(fake_boxd):
    sandbox = BoxdSandbox()

    async def run():
        await sandbox.start()
        machines = fake_boxd.instances[0].machines
        machines.status = "destroyed"
        machines.exec_error = FakeBoxdError("machine not found")
        await sandbox.exec(["true"])

    with pytest.raises(FakeBoxdError):
        asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "status, alive",
    [("running", True), ("suspended", True), ("hibernated", True), ("stopped", False), ("failed", False)],
)
def test_is_alive_follows_machine_status(fake_boxd, status, alive):
    sandbox = BoxdSandbox()

    async def run():
        assert await sandbox.is_alive() is False
        await sandbox.start()
        fake_boxd.instances[0].machines.status = status
        return await sandbox.is_alive()

    assert asyncio.run(run()) is alive


@pytest.mark.cpu
@pytest.mark.level0
def test_stop_deletes_the_machine_by_id_once(fake_boxd):
    sandbox = BoxdSandbox(image="python:3.12")

    async def run():
        await sandbox.start()
        await sandbox.stop()
        await sandbox.stop()

    asyncio.run(run())

    client = fake_boxd.instances[0]
    assert [call for call in client.machines.calls if call[0] == "delete"] == [("delete", "vm-1")]
    assert client.closed is True
    assert sandbox.machine_id is None


def _stop_with_failing_delete(fake_boxd, monkeypatch, status_after_error):
    monkeypatch.setattr("uni_agent.sandbox.boxd._DELETE_RETRY_DELAY", 0)
    sandbox = BoxdSandbox()

    async def run():
        await sandbox.start()
        client = fake_boxd.instances[0]

        async def broken_delete(machine_id, *, region=None):
            client.machines.calls.append(("delete", machine_id))
            client.machines.status = status_after_error
            raise FakeBoxdError("unavailable")

        client.machines.delete = broken_delete
        await sandbox.stop()
        return client

    return asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.level0
def test_stop_accepts_a_delete_error_when_the_machine_is_gone(fake_boxd, monkeypatch, caplog):
    client = _stop_with_failing_delete(fake_boxd, monkeypatch, "gone")

    assert [call[0] for call in client.machines.calls if call[0] in {"delete", "get"}] == ["delete", "get"]
    assert "Failed to delete" not in caplog.text
    assert client.closed is True


@pytest.mark.cpu
@pytest.mark.level0
def test_stop_retries_once_then_logs_delete_failures(fake_boxd, monkeypatch, caplog):
    client = _stop_with_failing_delete(fake_boxd, monkeypatch, "running")

    assert [call[0] for call in client.machines.calls if call[0] in {"delete", "get"}] == [
        "delete",
        "get",
        "delete",
        "get",
    ]
    assert "Failed to delete boxd machine vm-1: unavailable" in caplog.text
    assert client.closed is True


@pytest.mark.cpu
@pytest.mark.level0
def test_a_machine_whose_delete_failed_is_deleted_by_a_later_stop(fake_boxd, monkeypatch):
    monkeypatch.setattr("uni_agent.sandbox.boxd._DELETE_RETRY_DELAY", 0)
    sandbox = BoxdSandbox()

    async def run():
        await sandbox.start()
        first = fake_boxd.instances[-1]

        async def broken_delete(machine_id, *, region=None):
            first.machines.calls.append(("delete", machine_id))
            raise FakeBoxdError("unavailable")

        first.machines.delete = broken_delete
        await sandbox.stop()
        assert sandbox.machine_id is None
        await sandbox.stop()
        return first, fake_boxd.instances[-1]

    first, second = asyncio.run(run())

    assert first.closed is True
    assert second is not first
    assert ("delete", "vm-1") in second.machines.calls
    assert second.closed is True
    assert sandbox._cleanup_ids == []


@pytest.mark.cpu
@pytest.mark.level0
def test_files_on_the_machine_use_the_native_transfer(fake_boxd, tmp_path):
    sandbox = BoxdSandbox()
    payload = os.urandom(4096)
    source = tmp_path / "in.bin"
    source.write_bytes(payload)

    async def run():
        await sandbox.start()
        await sandbox.upload_file(source, "/home/boxd/data/in.bin")
        await sandbox.download_file("/home/boxd/data/in.bin", tmp_path / "out" / "out.bin")
        return fake_boxd.instances[0].machines

    machines = asyncio.run(run())

    assert (tmp_path / "out" / "out.bin").read_bytes() == payload
    assert ("exec", "vm-1", ["mkdir", "-p", "/home/boxd/data"], None, None, None) in machines.calls
    assert ("upload", "vm-1", "/home/boxd/data/in.bin", 4096) in machines.calls


@pytest.mark.cpu
@pytest.mark.level0
def test_files_in_the_container_are_staged_and_copied(fake_boxd):
    sandbox = BoxdSandbox(image="python:3.12")

    async def run():
        await sandbox.start()
        machines = fake_boxd.instances[0].machines
        machines.calls.clear()
        await sandbox.write_file("/repo/a.txt", "héllo")
        data = await sandbox.read_file("/repo/a.txt")
        return machines, data

    machines, data = asyncio.run(run())

    assert data == "héllo".encode()
    assert [call[:3] for call in machines.calls] == [
        ("exec", "vm-1", ["docker", "exec", CONTAINER, "test", "-d", "/repo/a.txt"]),
        ("exec", "vm-1", ["docker", "exec", CONTAINER, "mkdir", "-p", "/repo"]),
        ("upload", "vm-1", STAGED),
        ("exec", "vm-1", ["docker", "cp", STAGED, f"{CONTAINER}:/repo/a.txt"]),
        ("exec", "vm-1", ["rm", "-rf", STAGED]),
        ("exec", "vm-1", ["docker", "cp", "-L", f"{CONTAINER}:/repo/a.txt", STAGED]),
        ("download", "vm-1", STAGED),
        ("exec", "vm-1", ["rm", "-rf", STAGED]),
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_relative_container_paths_resolve_against_the_working_directory(fake_boxd):
    sandbox = BoxdSandbox(image="python:3.12", run_args=["--workdir", "/repo"])

    async def run():
        await sandbox.start()
        machines = fake_boxd.instances[0].machines
        machines.exec_results["docker inspect --format"] = SimpleNamespace(exit_code=0, stdout="/repo\n", stderr="")
        machines.calls.clear()
        await sandbox.write_file("sub/a.txt", "x")
        await sandbox.read_file("sub/a.txt")
        await sandbox.read_file("/abs/b.txt")
        return machines

    machines = asyncio.run(run())

    execs = _execs(machines)
    assert execs.count(["docker", "inspect", "--format", "{{.Config.WorkingDir}}", CONTAINER]) == 1
    assert ["docker", "exec", CONTAINER, "mkdir", "-p", "/repo/sub"] in execs
    assert ["docker", "cp", STAGED, f"{CONTAINER}:/repo/sub/a.txt"] in execs
    assert ["docker", "cp", "-L", f"{CONTAINER}:/repo/sub/a.txt", STAGED] in execs
    assert ["docker", "cp", "-L", f"{CONTAINER}:/abs/b.txt", STAGED] in execs


@pytest.mark.cpu
@pytest.mark.level0
def test_write_file_refuses_a_directory_destination(fake_boxd):
    sandbox = BoxdSandbox(image="python:3.12")
    machines: list[FakeMachines] = []

    async def run():
        await sandbox.start()
        machines.append(fake_boxd.instances[0].machines)
        machines[0].dirs.add("/repo")
        machines[0].calls.clear()
        await sandbox.write_file("/repo", "x")

    with pytest.raises(RuntimeError, match="is a directory"):
        asyncio.run(run())

    # Nothing is staged or copied, so no staging-named file lands in the directory.
    assert not [call for call in machines[0].calls if call[0] == "upload"]
    assert not [command for command in _execs(machines[0]) if command[:2] == ["docker", "cp"]]


@pytest.mark.cpu
@pytest.mark.level0
def test_failed_staging_cleanup_keeps_the_original_error(fake_boxd):
    sandbox = BoxdSandbox(image="python:3.12")

    async def run():
        await sandbox.start()
        machines = fake_boxd.instances[0].machines
        machines.exec_results["docker cp -L"] = SimpleNamespace(exit_code=1, stdout="", stderr="No such container")

        async def failing_rm(machine_id, command, **kwargs):
            if command[0] == "rm":
                raise FakeBoxdError("machine gone")
            return await FakeMachines.exec(machines, machine_id, command, **kwargs)

        machines.exec = failing_rm
        await sandbox.read_file("/repo/a.txt")

    with pytest.raises(RuntimeError, match="No such container"):
        asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.level0
def test_invalid_env_names_are_refused(fake_boxd):
    sandbox = BoxdSandbox()

    async def run():
        await sandbox.start()
        return await sandbox.exec(["true"], env={"A;rm -rf /": "x"})

    result = asyncio.run(run())

    assert result.exit_code == 127
    assert "invalid environment variable name" in result.stderr
    assert _execs(fake_boxd.instances[0].machines) == []


@pytest.mark.cpu
@pytest.mark.level0
def test_stop_deletes_a_machine_whose_create_was_cancelled(fake_boxd):
    sandbox = BoxdSandbox()

    client = FakeAsyncBoxd(api_key="k")
    client.machines.create_blocks = True
    sandbox._client = client

    async def run():
        task = asyncio.create_task(sandbox.start())
        await asyncio.sleep(0.01)
        machines = client.machines
        assert machines.names
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sandbox.machine_id is None
        await sandbox.stop()
        return machines

    machines = asyncio.run(run())

    assert ("delete", "vm-1") in machines.calls


@pytest.mark.cpu
@pytest.mark.level0
def test_stop_after_a_create_that_never_happened_deletes_nothing(fake_boxd):
    sandbox = BoxdSandbox()

    async def run():
        sandbox._client = FakeAsyncBoxd(api_key="k")
        sandbox._pending_name = "uni-agent-never-created"
        await sandbox.stop()

    asyncio.run(run())

    calls = fake_boxd.instances[0].machines.calls
    assert ("get", "uni-agent-never-created") in calls
    assert not [call for call in calls if call[0] == "delete"]


@pytest.mark.cpu
@pytest.mark.level0
def test_missing_container_file_raises(fake_boxd):
    sandbox = BoxdSandbox(image="python:3.12")

    async def run():
        await sandbox.start()
        machines = fake_boxd.instances[0].machines
        machines.exec_results["docker cp -L"] = SimpleNamespace(
            exit_code=1, stdout="", stderr="Could not find the file /missing"
        )
        await sandbox.read_file("/missing")

    with pytest.raises(RuntimeError, match="Could not find"):
        asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.level1
@pytest.mark.skipif(os.getenv("BOXD_E2E") != "1", reason="set BOXD_E2E=1 and BOXD_API_KEY to run against boxd")
@pytest.mark.parametrize("image", [None, "python:3.12"])
def test_live_lifecycle(image, tmp_path):
    """End to end against boxd: exec, env, exit codes, timeout, files, directories, delete."""
    sandbox = BoxdSandbox(
        image=image,
        runtime_timeout=1800,
        region=os.getenv("BOXD_REGION", "nyc"),
        org=os.getenv("BOXD_ORG"),
        name_prefix=os.getenv("BOXD_E2E_NAME_PREFIX", "uni-agent-e2e"),
        run_args=["--workdir", "/work"] if image else None,
    )
    payload = os.urandom(5 * 1024 * 1024)
    (tmp_path / "tree" / "sub").mkdir(parents=True)
    (tmp_path / "tree" / "sub" / "f.txt").write_text("tree file\n")
    (tmp_path / "big.bin").write_bytes(payload)
    remote = "/tmp/uni-agent-e2e"

    async def run():
        async with sandbox:
            machine_id = sandbox.machine_id
            print(f"boxd machine {machine_id} ({sandbox.machine_name})")
            result = await sandbox.exec_shell("echo $GREETING; echo oops >&2; exit 7", env={"GREETING": "hi there"})
            assert (result.exit_code, result.stdout, result.stderr) == (7, "hi there\n", "oops\n")
            assert (await sandbox.exec(["pwd"], workdir="/tmp")).stdout.strip() == "/tmp"
            assert (await sandbox.exec_shell("cat", timeout=30)).exit_code == 0
            timed_out = await sandbox.exec(["sleep", "60"], timeout=3)
            assert timed_out.exit_code == -1
            await asyncio.sleep(5)
            leftover = await sandbox.exec_shell("ps -eo args= | grep -cx 'sleep 60' || true")
            assert leftover.stdout.strip() == "0"

            await sandbox.upload(tmp_path / "big.bin", f"{remote}/big.bin")
            await sandbox.download(f"{remote}/big.bin", tmp_path / "back.bin")
            assert hashlib.sha256((tmp_path / "back.bin").read_bytes()).digest() == hashlib.sha256(payload).digest()
            await sandbox.upload(tmp_path / "tree", f"{remote}/tree")
            await sandbox.download(f"{remote}/tree", tmp_path / "tree-back")
            assert (tmp_path / "tree-back" / "sub" / "f.txt").read_text() == "tree file\n"
            await sandbox.write_file(f"{remote}/note.txt", "written\n")
            assert await sandbox.read_file(f"{remote}/note.txt") == b"written\n"
            assert (await sandbox.exec(["ln", "-sfn", "note.txt", f"{remote}/link.txt"])).exit_code == 0
            assert await sandbox.read_file(f"{remote}/link.txt") == b"written\n"
            if image is not None:
                await sandbox.write_file("rel/r.txt", "relative\n")
                assert (await sandbox.exec(["cat", "/work/rel/r.txt"])).stdout == "relative\n"
                assert await sandbox.read_file("rel/r.txt") == b"relative\n"
                with pytest.raises(RuntimeError, match="is a directory"):
                    await sandbox.write_file(remote, "x")
                assert (await sandbox.exec_shell(f"ls {remote} | grep -c uni-agent-xfer || true")).stdout.strip() == "0"
            assert await sandbox.is_alive()
        assert sandbox.machine_id is None
        return machine_id

    machine_id = asyncio.run(run())

    async def gone() -> bool:
        from boxd import AsyncBoxd
        from boxd.errors import NotFoundError

        client = AsyncBoxd(api_key=os.environ["BOXD_API_KEY"], region=os.getenv("BOXD_REGION", "nyc"))
        try:
            await client.machines.get(machine_id)
        except NotFoundError:
            return True
        finally:
            await client.close()
        return False

    assert asyncio.run(gone())
