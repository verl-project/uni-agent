"""Opt in with UNI_AGENT_DOCKER_WRITE_IMAGE=ubuntu:24.04 (pre-pull the image)."""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager

import pytest

from uni_agent.sandbox.base import Sandbox
from uni_agent.sandbox.docker import DockerSandbox


@asynccontextmanager
async def docker_sandbox():
    image = os.environ.get("UNI_AGENT_DOCKER_WRITE_IMAGE")
    if not image:
        pytest.skip("set UNI_AGENT_DOCKER_WRITE_IMAGE to run real-Docker tests")
    sandbox = DockerSandbox(image=image, pull_policy="never", run_args=["--network", "none"])
    await sandbox.start()
    name = sandbox._require_container()
    try:
        yield sandbox
    finally:
        await sandbox.stop()
        result = await sandbox._run_docker("inspect", name)
        assert result.exit_code != 0, "test container survived teardown"


@pytest.mark.cpu
@pytest.mark.level1
def test_linux_e2big_negative_control_and_large_editor_rewrite():
    from uni_agent.tools.edit_file import EditFileTool

    async def run():
        async with docker_sandbox() as sandbox:
            assert sys.platform == "linux" and os.sysconf("SC_PAGE_SIZE") == 4096
            path = "/tmp/large.py"
            original = "# filler\n" * 12000 + "answer = 1\n"
            # Run the actual unfixed base implementation through DockerSandbox.
            with pytest.raises(RuntimeError, match="Argument list too long"):
                await Sandbox.write_file(sandbox, path, original)
            await sandbox.write_file(path, original)
            tool = EditFileTool(sandbox)
            result = await tool.run(
                {"command": "str_replace", "path": path, "old_str": "answer = 1", "new_str": "answer = 2"}
            )
            assert result.status == "ok"
            assert await sandbox.read_file(path) == original.replace("answer = 1", "answer = 2").encode()

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.level1
def test_real_docker_binary_unicode_empty_permissions_and_errors():
    async def run():
        async with docker_sandbox() as sandbox:
            path = "/tmp/nested/space ' \" $HOME $(touch /tmp/injected) `touch /tmp/injected2`\n.bin"
            for content in [bytes(range(256)) * 4096, "中文 😀\n", b"", ""]:
                await sandbox.write_file(path, content)
                assert await sandbox.read_file(path) == (content.encode() if isinstance(content, str) else content)
            assert (await sandbox.exec(["test", "!", "-e", "/tmp/injected"])).exit_code == 0
            assert (await sandbox.exec(["test", "!", "-e", "/tmp/injected2"])).exit_code == 0
            assert (await sandbox.exec(["chmod", "640", path])).exit_code == 0
            before = await sandbox.exec(["stat", "-c", "%i:%a", path])
            await sandbox.write_file(path, b"replacement")
            after = await sandbox.exec(["stat", "-c", "%i:%a", path])
            assert before.exit_code == after.exit_code == 0
            assert before.stdout == after.stdout
            with pytest.raises(RuntimeError, match="write_file .* failed:"):
                await sandbox.write_file("/proc/uni-agent-no-write", b"x")
            with pytest.raises(RuntimeError, match="write_file .* failed:"):
                await sandbox.write_file("/tmp", b"x")

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.level1
def test_real_docker_cancelled_write_reaps_client_and_owner_stops_container(monkeypatch):
    async def run():
        async with docker_sandbox() as sandbox:
            assert (await sandbox.exec(["mkfifo", "/tmp/blocked-writer"])).exit_code == 0
            create = asyncio.create_subprocess_exec
            started = asyncio.Event()
            clients = []

            async def capture(*args, **kwargs):
                proc = await create(*args, **kwargs)
                if args[1:3] == ("exec", "-i"):
                    clients.append(proc)
                    started.set()
                return proc

            monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
            task = asyncio.create_task(sandbox.write_file("/tmp/blocked-writer", b"x" * (4 * 1024 * 1024)))
            await asyncio.wait_for(started.wait(), 5)
            await asyncio.sleep(0.2)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
            assert len(clients) == 1 and clients[0].returncode is not None
            with pytest.raises(ProcessLookupError):
                os.kill(clients[0].pid, 0)
            # Client cancellation alone does not guarantee remote process cleanup.
            # docker_sandbox's owner teardown verifies container removal.

    asyncio.run(run())
