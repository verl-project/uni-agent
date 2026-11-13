"""Exercise real subprocess pipes and the writer shell without requiring Docker."""

from __future__ import annotations

import asyncio
import os
import stat
import sys

import pytest

from uni_agent.sandbox.docker import DockerSandbox

pytestmark = pytest.mark.usefixtures("no_unhandled_asyncio_errors")


@pytest.fixture
def local_docker(tmp_path):
    # Replace only the Docker transport: production write_file/_run_docker and
    # bash/cat run unchanged. exec avoids introducing an extra child process.
    executable = tmp_path / "docker"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "assert sys.argv[1:4] == ['exec', '-i', 'test-container']\n"
        "assert max(map(len, sys.argv)) < 4096\n"
        "os.execvp(sys.argv[4], sys.argv[4:])\n"
    )
    executable.chmod(0o700)
    sandbox = DockerSandbox(docker_binary=str(executable))
    sandbox._container_name = "test-container"
    return sandbox


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "content",
    [b"", "", "中文 😀\n", bytes(range(256)) * 4096],
    ids=["empty-bytes", "empty-text", "unicode", "binary-1MiB"],
)
def test_write_file_roundtrip(local_docker, tmp_path, content):
    target = tmp_path / "nested" / "space ' \" $HOME $(touch injected) `touch injected2`\n.bin"
    asyncio.run(local_docker.write_file(str(target), content))
    assert target.read_bytes() == (content.encode() if isinstance(content, str) else content)
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.cpu
@pytest.mark.level0
def test_write_file_relative_option_like_path(local_docker, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    asyncio.run(local_docker.write_file("-dir/-file", b"content"))
    assert (tmp_path / "-dir/-file").read_bytes() == b"content"


@pytest.mark.cpu
@pytest.mark.level0
def test_overwrite_preserves_inode_permissions_and_symlink(local_docker, tmp_path):
    target = tmp_path / "file"
    target.write_bytes(b"old contents" * 100)
    target.chmod(0o640)
    inode = target.stat().st_ino
    link = tmp_path / "link"
    link.symlink_to(target)
    asyncio.run(local_docker.write_file(str(link), b"new"))
    assert link.is_symlink()
    assert target.read_bytes() == b"new"
    assert target.stat().st_ino == inode
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("failure", ["directory", "parent_file", "read_only"])
def test_write_file_propagates_shell_errors(local_docker, tmp_path, failure):
    target = tmp_path / "target"
    if failure == "directory":
        target.mkdir()
    elif failure == "parent_file":
        target.write_text("parent is a file")
        target = target / "child"
    else:
        if os.geteuid() == 0:
            pytest.skip("root can write read-only files")
        target.write_text("original")
        target.chmod(0o400)
    with pytest.raises(RuntimeError, match="write_file .* failed:"):
        asyncio.run(local_docker.write_file(str(target), b"new" * (1024 * 1024)))
    if failure == "read_only":
        assert target.read_text() == "original"


@pytest.mark.cpu
@pytest.mark.level0
def test_write_file_requires_started_container(tmp_path):
    with pytest.raises(RuntimeError, match="not started"):
        asyncio.run(DockerSandbox().write_file(str(tmp_path / "file"), b"x"))


@pytest.mark.cpu
@pytest.mark.level0
def test_write_file_missing_docker(tmp_path):
    sandbox = DockerSandbox(docker_binary=str(tmp_path / "missing-docker"))
    sandbox._container_name = "test-container"
    with pytest.raises(RuntimeError, match="Docker executable .* was not found"):
        asyncio.run(sandbox.write_file("/tmp/file", b"x"))


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("mode", ["cancel", "timeout"])
def test_interrupted_pipe_reaps_client(tmp_path, monkeypatch, mode):
    executable = tmp_path / "docker"
    executable.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    executable.chmod(0o700)
    sandbox = DockerSandbox(docker_binary=str(executable))
    sandbox._container_name = "test-container"
    real_create = asyncio.create_subprocess_exec

    async def run():
        started = asyncio.Event()
        processes = []

        async def create(*args, **kwargs):
            proc = await real_create(*args, **kwargs)
            processes.append(proc)
            started.set()
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        if mode == "cancel":
            task = asyncio.create_task(sandbox.write_file("/tmp/file", b"x" * (4 * 1024 * 1024)))
            await asyncio.wait_for(started.wait(), 5)
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        else:
            with pytest.raises(asyncio.TimeoutError):
                await sandbox._run_docker("exec", timeout=0.1, input_data=b"x" * (4 * 1024 * 1024))
        assert len(processes) == 1
        assert processes[0].returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(processes[0].pid, 0)

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.level0
def test_repeated_cancellation_waits_for_client_reap(monkeypatch):
    async def run():
        writing = asyncio.Event()
        cleaning = asyncio.Event()
        release = asyncio.Event()
        killed = asyncio.Event()
        reaped = False

        class Process:
            stdin = None
            returncode = None

            def kill(self):
                self.returncode = -9
                killed.set()

            async def communicate(self, input=None):
                nonlocal reaped
                if input is not None:
                    writing.set()
                    await killed.wait()
                cleaning.set()
                await release.wait()
                reaped = True
                return b"", b""

        async def create(*args, **kwargs):
            return Process()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        sandbox = DockerSandbox()
        sandbox._container_name = "test-container"
        task = asyncio.create_task(sandbox.write_file("/tmp/file", b"x"))
        await asyncio.wait_for(writing.wait(), 5)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert reaped

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("entry", ["stdin", "exec", "startup"])
@pytest.mark.parametrize("cancel_count", [1, 2])
def test_timeout_then_cancellation_finishes_cleanup_and_propagates(monkeypatch, entry, cancel_count):
    # Disable the outer startup deadline so the provider's own timeout reaches
    # __aenter__ directly. Cancellation must not become a retryable failure.
    monkeypatch.setenv("SANDBOX_STARTUP_TIMEOUT", "0")

    async def run():
        killed = asyncio.Event()
        cleaning = asyncio.Event()
        release = asyncio.Event()
        reaped = False
        stdin_closed = False
        communications = 0
        attempts = 0
        stops = 0
        backoffs = []
        real_sleep = asyncio.sleep

        async def no_retry_delay(delay):
            if delay:
                backoffs.append(delay)
            await real_sleep(0)

        class Stdin:
            async def wait_closed(self):
                nonlocal stdin_closed
                cleaning.set()
                await release.wait()
                stdin_closed = True

        class Process:
            stdin = Stdin() if entry == "stdin" else None
            returncode = None

            def kill(self):
                self.returncode = -9
                killed.set()

            async def communicate(self, input=None):
                nonlocal reaped, communications
                communications += 1
                await killed.wait()
                if self.stdin is None:
                    cleaning.set()
                    await release.wait()
                reaped = True
                return b"", b""

        process = Process()

        async def create(*args, **kwargs):
            return process

        sandbox = DockerSandbox()
        sandbox._container_name = "test-container"

        async def start():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                await sandbox._run_docker("run", timeout=0.01)

        async def stop():
            nonlocal stops
            assert reaped, "owner teardown started before client cleanup finished"
            stops += 1

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        monkeypatch.setattr(asyncio, "sleep", no_retry_delay)
        monkeypatch.setattr(sandbox, "start", start)
        monkeypatch.setattr(sandbox, "stop", stop)
        if entry == "startup":
            operation = sandbox.__aenter__(retry=2)
        elif entry == "exec":
            operation = sandbox.exec(["true"], timeout=0.01)
        else:
            operation = sandbox._run_docker("exec", timeout=0.01, input_data=b"payload")
        task = asyncio.create_task(operation)
        try:
            # kill() is only reached after the real wait_for timeout. The
            # cleanup gate prevents the subprocess/pipe from finishing yet.
            await asyncio.wait_for(cleaning.wait(), 5)
            assert killed.is_set() and not task.done()
            for _ in range(cancel_count):
                assert task.cancel()
                await real_sleep(0)
                assert not task.done(), "cancellation abandoned client cleanup"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            assert task.cancelled()
            assert reaped and communications == 1 and process.returncode == -9
            assert stdin_closed == (entry == "stdin")
            assert attempts == (1 if entry == "startup" else 0)
            assert stops == (1 if entry == "startup" else 0)
            assert not backoffs, "cancelled startup entered retry backoff"
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
