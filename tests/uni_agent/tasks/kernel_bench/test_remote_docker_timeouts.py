import asyncio

import pytest

from examples.claude_code_kernel_task.remote_docker import RemoteDockerSandbox
from uni_agent.sandbox.base import ExecResult
from uni_agent.sandbox.docker import DockerSandbox


pytestmark = [pytest.mark.cpu, pytest.mark.level0]

def test_remote_docker_lifecycle_timeouts(monkeypatch):
    calls = []

    async def run(self, *args, timeout=None):
        calls.append((args, timeout))
        return ExecResult(0, "/workspace\n", "")

    monkeypatch.setattr(DockerSandbox, "_run_docker", run)
    sandbox = RemoteDockerSandbox(docker_host="ssh://test", image="test", npu_lock_dir="/locks")

    async def lifecycle():
        await sandbox.start()
        await sandbox.exec(["pwd"], timeout=30)
        await sandbox.stop()

    asyncio.run(lifecycle())
    assert [args[2] for args, _ in calls] == ["image", "run", "exec", "rm"]
    assert [timeout for _, timeout in calls] == [30, 60, 30, 30]
    assert all(args[:2] == ("--host", "ssh://test") for args, _ in calls)
