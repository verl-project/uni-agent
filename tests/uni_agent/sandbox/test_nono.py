"""Integration tests for NonoSandbox using the real nono-py SDK.

Requires nono-py and a compatible Linux host. SQLite support is provided
by Python's standard-library sqlite3 module.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

from uni_agent.sandbox.base import ExecResult
from uni_agent.tools.base import ToolError


@pytest.fixture
def nono_backend():
    pytest.importorskip(
        "nono_py",
        reason="Install nono-py to run the nono integration tests.",
    )
    from uni_agent.sandbox import nono

    return nono


@pytest.mark.cpu
@pytest.mark.level1
@pytest.mark.asyncio
async def test_lifecycle(nono_backend, tmp_path, monkeypatch):
    monkeypatch.setattr(nono_backend, "NONO_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(nono_backend, "NONO_SHELL_ROOT", tmp_path / "shell")
    repo_dir = tmp_path / "source-repo"
    repo_dir.mkdir()
    source_file = repo_dir / "prepared.txt"
    source_file.write_text("prepared repository\n", encoding="utf-8")

    sandbox = nono_backend.NonoSandbox(
        repo_dir=repo_dir,
        conda_dir=sys.prefix,
        startup_commands=[],
        instance_id="test-lifecycle",
        runtime_timeout=30,
    )
    assert await sandbox.is_alive() is False

    await sandbox.start()
    try:
        session_id = sandbox.session_id
        session_dir = sandbox.session_dir
        assert session_id is not None
        assert session_dir is not None
        assert session_dir.is_dir()
        assert await sandbox.is_alive() is True
        assert sandbox.workdir != repo_dir
        assert (sandbox.workdir / "prepared.txt").read_text(encoding="utf-8") == "prepared repository\n"

        result = await sandbox.exec(["/bin/echo", "nono lifecycle"], timeout=5)
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "nono lifecycle\n"
        assert result.stderr == ""
        assert sandbox.registry.get(session_id).state == "idle"
    finally:
        await sandbox.stop()

    assert await sandbox.is_alive() is False
    assert not session_dir.exists()
    assert sandbox.registry.get(session_id) is None
    assert source_file.read_text(encoding="utf-8") == "prepared repository\n"

    await sandbox.stop()
    assert await sandbox.is_alive() is False


@pytest.mark.cpu
@pytest.mark.level1
@pytest.mark.asyncio
async def test_exec_result(nono_backend, tmp_path, monkeypatch):
    monkeypatch.setattr(nono_backend, "NONO_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(nono_backend, "NONO_SHELL_ROOT", tmp_path / "shell")
    repo_dir = tmp_path / "source-repo"
    repo_dir.mkdir()

    sandbox = nono_backend.NonoSandbox(
        repo_dir=repo_dir,
        conda_dir=sys.prefix,
        startup_commands=[],
        instance_id="test-exec-result",
        runtime_timeout=30,
    )
    await sandbox.start()
    try:
        result = await sandbox.exec(
            [
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                "printf 'standard output\\n'; printf 'standard error\\n' >&2; exit 7",
            ],
            timeout=5,
        )
        assert isinstance(result, ExecResult)
        assert result.exit_code == 7
        assert result.stdout == "standard output\n"
        assert result.stderr == "standard error\n"
        assert await sandbox.is_alive() is True
        assert sandbox.registry.get(sandbox.session_id).state == "idle"
    finally:
        await sandbox.stop()


@pytest.mark.cpu
@pytest.mark.level1
@pytest.mark.asyncio
async def test_exec_timeout(nono_backend, tmp_path, monkeypatch):
    monkeypatch.setattr(nono_backend, "NONO_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(nono_backend, "NONO_SHELL_ROOT", tmp_path / "shell")
    repo_dir = tmp_path / "source-repo"
    repo_dir.mkdir()

    sandbox = nono_backend.NonoSandbox(
        repo_dir=repo_dir,
        conda_dir=sys.prefix,
        startup_commands=[],
        instance_id="test-exec-timeout",
        runtime_timeout=30,
    )
    await sandbox.start()
    try:
        started = time.monotonic()
        result = await sandbox.exec(
            [
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                "printf '%s\\n' \"$$\"; exec /bin/sleep 10",
            ],
            timeout=0.2,
        )
        elapsed = time.monotonic() - started

        # nono-py returns command timeouts as exit status 124.
        assert result.exit_code == 124
        assert elapsed < 5
        assert result.stdout.strip().isdigit(), result.stdout
        process_id = int(result.stdout.strip())
        with pytest.raises(ProcessLookupError):
            os.kill(process_id, 0)

        assert await sandbox.is_alive() is True
        assert sandbox.registry.get(sandbox.session_id).state == "idle"
        followup = await sandbox.exec(["/bin/true"], timeout=5)
        assert followup.exit_code == 0, followup.stderr
    finally:
        await sandbox.stop()


@pytest.mark.cpu
@pytest.mark.level1
@pytest.mark.asyncio
async def test_dead_backend(nono_backend, tmp_path, monkeypatch):
    monkeypatch.setattr(nono_backend, "NONO_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(nono_backend, "NONO_SHELL_ROOT", tmp_path / "shell")
    repo_dir = tmp_path / "source-repo"
    repo_dir.mkdir()

    sandbox = nono_backend.NonoSandbox(
        repo_dir=repo_dir,
        conda_dir=sys.prefix,
        startup_commands=[],
        instance_id="test-dead-backend",
        runtime_timeout=30,
    )
    await sandbox.start()
    try:
        assert await sandbox.is_alive() is True
        assert sandbox.workdir is not None
        sandbox.workdir.rmdir()
        assert await sandbox.is_alive() is False

        with pytest.raises(RuntimeError, match="working directory"):
            await sandbox.exec(["/bin/echo", "must not run"], timeout=5)

        assert sandbox.registry.get(sandbox.session_id).state == "idle"
    finally:
        await sandbox.stop()


@pytest.mark.cpu
@pytest.mark.level1
@pytest.mark.asyncio
async def test_environment_forwarding(nono_backend, tmp_path, monkeypatch):
    monkeypatch.setattr(nono_backend, "NONO_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(nono_backend, "NONO_SHELL_ROOT", tmp_path / "shell")
    monkeypatch.setenv("NONO_TEST_HOST_ONLY", "host-only")
    repo_dir = tmp_path / "source-repo"
    repo_dir.mkdir()

    sandbox = nono_backend.NonoSandbox(
        repo_dir=repo_dir,
        conda_dir=sys.prefix,
        startup_commands=[
            "export NONO_TEST_VALUE=startup",
            "export NONO_TEST_REMOVED=temporary",
            "unset NONO_TEST_REMOVED",
        ],
        instance_id="test-environment-forwarding",
        runtime_timeout=30,
    )
    await sandbox.start()
    try:
        base_env = sandbox.env.copy()
        command_dir = sandbox.workdir / "nested"
        command_dir.mkdir()
        script = (
            "import json, os; "
            "print(json.dumps({"
            "'cwd': os.getcwd(), "
            "'home': os.getenv('HOME'), "
            "'tmpdir': os.getenv('TMPDIR'), "
            "'conda_prefix': os.getenv('CONDA_PREFIX'), "
            "'value': os.getenv('NONO_TEST_VALUE'), "
            "'per_call': os.getenv('NONO_TEST_PER_CALL'), "
            "'removed': os.getenv('NONO_TEST_REMOVED'), "
            "'host_only': os.getenv('NONO_TEST_HOST_ONLY')"
            "}))"
        )
        result = await sandbox.exec(
            [sys.executable, "-c", script],
            workdir=str(command_dir),
            env={"NONO_TEST_VALUE": "override", "NONO_TEST_PER_CALL": "one command"},
            timeout=5,
        )
        assert result.exit_code == 0, result.stderr
        expected = {
            "cwd": str(command_dir),
            "home": str(sandbox.session_dir / "home"),
            "tmpdir": str(sandbox.session_dir / "tmp"),
            "conda_prefix": str(sandbox.conda_dir),
            "value": "override",
            "per_call": "one command",
            "removed": None,
            "host_only": None,
        }
        assert json.loads(result.stdout) == expected

        result = await sandbox.exec([sys.executable, "-c", script], timeout=5)
        assert result.exit_code == 0, result.stderr
        expected.update(cwd=str(sandbox.workdir), value="startup", per_call=None)
        assert json.loads(result.stdout) == expected
        assert sandbox.env == base_env
    finally:
        await sandbox.stop()


@pytest.mark.cpu
@pytest.mark.level1
@pytest.mark.asyncio
async def test_binary_file_transfer(nono_backend, tmp_path, monkeypatch):
    monkeypatch.setattr(nono_backend, "NONO_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(nono_backend, "NONO_SHELL_ROOT", tmp_path / "shell")
    repo_dir = tmp_path / "source-repo"
    repo_dir.mkdir()

    sandbox = nono_backend.NonoSandbox(
        repo_dir=repo_dir,
        conda_dir=sys.prefix,
        startup_commands=[],
        instance_id="test-binary-file-transfer",
        runtime_timeout=30,
    )
    await sandbox.start()
    try:
        payload = bytes(range(256)) * 1024
        written_file = str(sandbox.workdir / "written folder" / "payload.bin")
        await sandbox.write_file(written_file, payload)
        assert await sandbox.read_file(written_file) == payload

        await sandbox.write_file(written_file, b"")
        assert await sandbox.read_file(written_file) == b""

        upload_source = tmp_path / "upload.bin"
        upload_source.write_bytes(payload)
        uploaded_file = str(sandbox.workdir / "uploaded folder" / "payload.bin")
        await sandbox.upload_file(upload_source, uploaded_file)
        assert await sandbox.read_file(uploaded_file) == payload

        download_target = tmp_path / "downloaded" / "nested" / "roundtrip.bin"
        await sandbox.download_file(uploaded_file, download_target)
        assert download_target.parent.is_dir()
        assert download_target.read_bytes() == payload

        await sandbox.download_file(written_file, download_target)
        assert download_target.read_bytes() == b""
        assert upload_source.read_bytes() == payload
        assert sandbox.registry.get(sandbox.session_id).state == "idle"
    finally:
        await sandbox.stop()


@pytest.mark.cpu
@pytest.mark.level1
@pytest.mark.asyncio
async def test_filesystem_restriction(nono_backend, tmp_path, monkeypatch):
    monkeypatch.setattr(nono_backend, "NONO_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(nono_backend, "NONO_SHELL_ROOT", tmp_path / "shell")
    repo_dir = tmp_path / "source-repo"
    repo_dir.mkdir()
    source_file = repo_dir / "source.txt"
    source_file.write_bytes(b"original repository\n")

    # Keep negative write checks away from the real Python environment.
    conda_dir = tmp_path / "conda-env"
    (conda_dir / "bin").mkdir(parents=True)
    environment_file = conda_dir / "marker.txt"
    environment_file.write_bytes(b"read-only environment\n")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "private.txt"
    outside_file.write_bytes(b"private host file\n")
    denied_new_file = outside_dir / "must-not-exist.txt"

    sandbox = nono_backend.NonoSandbox(
        repo_dir=repo_dir,
        conda_dir=conda_dir,
        startup_commands=[],
        instance_id="test-filesystem-restriction",
        runtime_timeout=30,
    )
    await sandbox.start()
    try:
        writable_file = str(sandbox.workdir / "source.txt")
        await sandbox.write_file(writable_file, b"modified working copy\n")
        assert await sandbox.read_file(writable_file) == b"modified working copy\n"
        assert source_file.read_bytes() == b"original repository\n"

        result = await sandbox.exec(["/bin/cat", str(environment_file)], timeout=5)
        assert result.exit_code == 0, result.stderr
        assert result.stdout == "read-only environment\n"

        escape_link = sandbox.workdir / "outside-link"
        escape_link.symlink_to(outside_file)
        relative_escape = os.path.relpath(outside_file, sandbox.workdir)
        for path in (
            str(source_file),
            str(outside_file),
            str(escape_link),
            relative_escape,
            sandbox.registry.db_path,
        ):
            denied = await sandbox.exec(["/bin/cat", path], timeout=5)
            assert denied.exit_code != 0, path
            assert denied.stdout == "", path
            assert "Permission denied" in denied.stderr, denied.stderr

        for path in (
            str(source_file),
            str(environment_file),
            str(outside_file),
            str(escape_link),
            relative_escape,
            str(denied_new_file),
        ):
            with pytest.raises(ToolError, match="Permission denied"):
                await sandbox.write_file(path, b"must not be written\n")
            assert source_file.read_bytes() == b"original repository\n"
            assert environment_file.read_bytes() == b"read-only environment\n"
            assert outside_file.read_bytes() == b"private host file\n"
            assert not denied_new_file.exists()

        assert await sandbox.is_alive() is True
        record = sandbox.registry.get(sandbox.session_id)
        assert record.state == "idle"
        assert record.owner_token is None
    finally:
        await sandbox.stop()
