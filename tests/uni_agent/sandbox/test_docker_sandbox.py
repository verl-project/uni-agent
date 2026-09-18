from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from uni_agent.sandbox.base import ExecResult, Sandbox, SandboxConfig
from uni_agent.sandbox.docker import DockerSandbox
from uni_agent.sandbox.registry import SANDBOX_MODULES, build_sandbox


def _ok(stdout: str = "") -> ExecResult:
    return ExecResult(exit_code=0, stdout=stdout, stderr="")


@pytest.mark.cpu
@pytest.mark.level0
def test_registry_builds_docker_sandbox_from_config():
    config = SandboxConfig(
        provider="docker",
        image="example:local",
        sandbox_kwargs={"run_args": ["--network", "none"]},
    )

    sandbox = build_sandbox(config)

    assert SANDBOX_MODULES["docker"] == "uni_agent.sandbox.docker"
    assert isinstance(sandbox, DockerSandbox)
    assert sandbox.image == "example:local"
    assert sandbox.run_args == ["--network", "none"]
    assert sandbox.pull_policy == "missing"
    assert "_exec" in DockerSandbox.__dict__
    assert "exec" not in DockerSandbox.__dict__
    assert DockerSandbox.exec is Sandbox.exec


@pytest.mark.cpu
@pytest.mark.level0
def test_start_requires_local_image_and_builds_detached_run(monkeypatch):
    sandbox = DockerSandbox(
        image="example:local",
        container_name="agent-test",
        run_args=["--network", "none"],
        pull_policy="never",
    )
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, timeout=None):
        calls.append(args)
        return _ok("sha256:image\n" if args[:2] == ("image", "inspect") else "container-id\n")

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)
    asyncio.run(sandbox.start())

    assert calls == [
        ("image", "inspect", "example:local"),
        (
            "run",
            "--rm",
            "-d",
            "--name",
            "agent-test",
            "--pull",
            "never",
            "--entrypoint",
            "sleep",
            "--network",
            "none",
            "example:local",
            "infinity",
        ),
    ]
    assert sandbox._container_name == "agent-test"


@pytest.mark.cpu
@pytest.mark.level0
def test_start_pulls_missing_image_through_docker_run(monkeypatch):
    sandbox = DockerSandbox(image="registry.example.com/agent:latest", container_name="agent-test")
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, timeout=None):
        calls.append(args)
        return _ok("container-id\n")

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)
    asyncio.run(sandbox.start())

    assert calls == [
        (
            "run",
            "--rm",
            "-d",
            "--name",
            "agent-test",
            "--pull",
            "missing",
            "--entrypoint",
            "sleep",
            "registry.example.com/agent:latest",
            "infinity",
        )
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_start_rejects_missing_local_image(monkeypatch):
    sandbox = DockerSandbox(image="missing:local", pull_policy="never")

    async def fake_run(*args: str, timeout=None):
        return ExecResult(exit_code=1, stdout="", stderr="No such image")

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)

    with pytest.raises(RuntimeError, match="not available locally"):
        asyncio.run(sandbox.start())
    assert sandbox._container_name is None


@pytest.mark.cpu
@pytest.mark.level0
def test_rejects_unknown_pull_policy():
    with pytest.raises(ValueError, match="pull_policy"):
        DockerSandbox(pull_policy="sometimes")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("name", ["pull_timeout", "start_timeout"])
def test_rejects_non_positive_timeouts(name: str):
    with pytest.raises(ValueError, match=name):
        DockerSandbox(**{name: 0})


@pytest.mark.cpu
@pytest.mark.level0
def test_pull_timeout_pulls_missing_image_separately(monkeypatch):
    sandbox = DockerSandbox(
        image="example:local",
        container_name="agent-test",
        pull_timeout=900,
        start_timeout=120,
    )
    calls: list[tuple[tuple[str, ...], float | None]] = []

    async def fake_run(*args: str, timeout=None):
        calls.append((args, timeout))
        if args[:2] == ("image", "inspect"):
            return ExecResult(exit_code=1, stdout="", stderr="No such image")
        return _ok("container-id\n")

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)
    asyncio.run(sandbox.start())

    assert calls == [
        (("image", "inspect", "example:local"), None),
        (("pull", "example:local"), 900.0),
        (
            (
                "run",
                "--rm",
                "-d",
                "--name",
                "agent-test",
                "--pull",
                "never",
                "--entrypoint",
                "sleep",
                "example:local",
                "infinity",
            ),
            120.0,
        ),
    ]
    assert sandbox._container_name == "agent-test"


@pytest.mark.cpu
@pytest.mark.level0
def test_pull_timeout_skips_pull_when_image_is_local(monkeypatch):
    sandbox = DockerSandbox(image="example:local", container_name="agent-test", pull_timeout=900)
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, timeout=None):
        calls.append(args)
        return _ok("sha256:image\n" if args[:2] == ("image", "inspect") else "container-id\n")

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)
    asyncio.run(sandbox.start())

    assert [args[0] for args in calls] == ["image", "run"]


@pytest.mark.cpu
@pytest.mark.level0
def test_pull_policy_always_pulls_without_inspecting(monkeypatch):
    sandbox = DockerSandbox(
        image="example:local",
        container_name="agent-test",
        pull_policy="always",
        pull_timeout=900,
    )
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, timeout=None):
        calls.append(args)
        return _ok("container-id\n")

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)
    asyncio.run(sandbox.start())

    assert [args[0] for args in calls] == ["pull", "run"]


@pytest.mark.cpu
@pytest.mark.level0
def test_pull_timeout_reported_as_timeout_error(monkeypatch):
    sandbox = DockerSandbox(image="example:local", pull_policy="always", pull_timeout=900)

    async def fake_run(*args: str, timeout=None):
        raise asyncio.TimeoutError

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)

    with pytest.raises(TimeoutError, match="exceeded pull_timeout=900s"):
        asyncio.run(sandbox.start())
    assert sandbox._container_name is None


@pytest.mark.cpu
@pytest.mark.level0
def test_start_timeout_removes_partially_created_container(monkeypatch):
    sandbox = DockerSandbox(image="example:local", container_name="agent-test", start_timeout=120)
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, timeout=None):
        calls.append(args)
        if args[0] == "run":
            raise asyncio.TimeoutError
        return _ok()

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)

    with pytest.raises(TimeoutError, match="exceeded start_timeout=120s"):
        asyncio.run(sandbox.start())
    assert calls[-1] == ("rm", "-f", "agent-test")
    assert sandbox._container_name is None


@pytest.mark.cpu
@pytest.mark.level0
def test_exec_forwards_workdir_environment_and_argv(monkeypatch):
    sandbox = DockerSandbox(image="example:local")
    sandbox._container_name = "agent-test"
    calls: list[tuple[tuple[str, ...], float | None]] = []

    async def fake_run(*args: str, timeout=None):
        calls.append((args, timeout))
        return _ok("hello\n")

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)
    result = asyncio.run(
        sandbox._exec(
            ["python", "-c", "print('hello')"],
            timeout=12.0,
            workdir="/workspace",
            env={"A": "1", "B": "two"},
        )
    )

    assert result.stdout == "hello\n"
    assert calls == [
        (
            (
                "exec",
                "--workdir",
                "/workspace",
                "--env",
                "A=1",
                "--env",
                "B=two",
                "agent-test",
                "python",
                "-c",
                "print('hello')",
            ),
            12.0,
        )
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_stop_is_idempotent_and_checks_liveness(monkeypatch):
    sandbox = DockerSandbox(image="example:local")
    sandbox._container_name = "agent-test"
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, timeout=None):
        calls.append(args)
        if args[0] == "inspect":
            return _ok("true\n")
        return _ok()

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)

    async def run() -> None:
        assert await sandbox.is_alive() is True
        await sandbox.stop()
        await sandbox.stop()
        assert await sandbox.is_alive() is False

    asyncio.run(run())
    assert calls == [
        ("inspect", "--format", "{{.State.Running}}", "agent-test"),
        ("rm", "-f", "agent-test"),
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_upload_and_download_use_docker_cp(monkeypatch, tmp_path: Path):
    sandbox = DockerSandbox(image="example:local")
    sandbox._container_name = "agent-test"
    source = tmp_path / "source.bin"
    destination = tmp_path / "nested" / "destination.bin"
    source.write_bytes(b"content")
    docker_calls: list[tuple[str, ...]] = []
    exec_calls: list[list[str]] = []

    async def fake_run(*args: str, timeout=None):
        docker_calls.append(args)
        return _ok()

    async def fake_exec(argv, **kwargs):
        exec_calls.append(argv)
        return _ok()

    monkeypatch.setattr(sandbox, "_run_docker", fake_run)
    monkeypatch.setattr(sandbox, "exec", fake_exec)

    async def run() -> None:
        await sandbox.upload_file(source, "/workspace/data/source.bin")
        await sandbox.download_file("/workspace/data/result.bin", destination)

    asyncio.run(run())

    assert exec_calls == [["mkdir", "-p", "/workspace/data"]]
    assert docker_calls == [
        ("cp", str(source), "agent-test:/workspace/data/source.bin"),
        ("cp", "agent-test:/workspace/data/result.bin", str(destination)),
    ]
    assert destination.parent.is_dir()
