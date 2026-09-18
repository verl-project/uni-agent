"""Opt-in real-Docker tests; set UNI_AGENT_DOCKER_TEST_IMAGE to a local Bash image."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from uni_agent.agents import AgentResult
from uni_agent.sandbox import SandboxConfig
from uni_agent.sandbox.docker import DockerSandbox
from uni_agent.tasks.swe_bench.task import SWEBenchTask, SWEBenchTaskConfig

pytestmark = [pytest.mark.cpu, pytest.mark.level1]


@pytest.fixture
def sandbox():
    image = os.getenv("UNI_AGENT_DOCKER_TEST_IMAGE")
    if not image:
        pytest.skip("set UNI_AGENT_DOCKER_TEST_IMAGE to a local image with bash, base64, and sleep")
    return DockerSandbox(image=image, pull_policy="never", run_args=["--network", "none"])


async def assert_removed(sandbox, label):
    result = await sandbox._run_docker("container", "ls", "-aq", "--filter", f"label={label}", timeout=10)
    assert result.exit_code == 0, result.stderr
    assert not result.stdout.strip()
    assert not await sandbox.is_alive()


def test_isolated_workspaces_and_file_transfer(sandbox, tmp_path: Path):
    other = DockerSandbox(image=sandbox.image, pull_policy="never", run_args=["--network", "none"])

    async def run():
        async with sandbox, other:
            labels = sandbox._cleanup_label, other._cleanup_label
            assert labels[0] != labels[1]
            content = b"binary\x00\xff\n" + "你好".encode()
            await sandbox.write_file("/testbed/file with spaces", content)
            assert await sandbox.read_file("/testbed/file with spaces") == content
            assert (await other.exec(["test", "-e", "/testbed/file with spaces"])).exit_code == 1
            local_file = tmp_path / "download.bin"
            await sandbox.download_file("/testbed/file with spaces", local_file)
            assert local_file.read_bytes() == content
            await other.upload_file(local_file, "/testbed/upload.bin")
            assert await other.read_file("/testbed/upload.bin") == content
            result = await sandbox.exec_shell(
                'printf "%s:%s" "$PWD" "$VALUE"', workdir="/testbed", env={"VALUE": "a b"}
            )
            assert result.stdout == "/testbed:a b"
        await assert_removed(sandbox, labels[0])
        await assert_removed(other, labels[1])

    asyncio.run(run())


@pytest.mark.parametrize("failure", [None, "agent", "reward", "cancel"])
def test_swe_task_shares_agent_mutations_with_reward_and_cleans_up(sandbox, monkeypatch, failure):
    from uni_agent.tasks.swe_bench import reward

    task = SWEBenchTask(SWEBenchTaskConfig(sandbox=SandboxConfig(provider="docker", image=sandbox.image)))
    monkeypatch.setattr(task, "build_sandbox", lambda: sandbox)
    label = None

    async def solve(*, sandbox, workdir, **kwargs):
        nonlocal label
        label = sandbox._cleanup_label
        assert workdir == "/testbed"
        await sandbox.write_file("/testbed/agent-result", "hello")
        if failure == "agent":
            raise RuntimeError("agent crashed")
        if failure == "cancel":
            raise asyncio.CancelledError
        return AgentResult(finished=True)

    async def verify(sample, environment, **kwargs):
        assert environment is sandbox
        assert await environment.read_file("/testbed/agent-result") == b"hello"
        if failure == "reward":
            raise RuntimeError("reward crashed")
        return {"resolved": True}

    agent = AsyncMock()
    agent.run.side_effect = solve
    monkeypatch.setattr(task, "build_agent", lambda: agent)
    monkeypatch.setattr(reward, "compute_reward", verify)

    async def run():
        if failure:
            error = asyncio.CancelledError if failure == "cancel" else RuntimeError
            with pytest.raises(error):
                await task.run()
        else:
            result = await task.run()
            assert result.reward == 1.0
            assert result.finished is True
        assert label is not None
        await assert_removed(sandbox, label)

    asyncio.run(run())


@pytest.mark.parametrize("interrupt", ["timeout", "cancel"])
def test_interrupted_exec_removes_container_and_child_processes(sandbox, interrupt):
    async def run():
        async with sandbox:
            label = sandbox._cleanup_label
            # The child ignores TERM; removing the container must still kill it.
            command = "trap '' TERM; (trap '' TERM; sleep 60) & wait"
            if interrupt == "timeout":
                result = await sandbox.exec_shell(command, timeout=0.2)
                assert result.exit_code == -1
            else:
                task = asyncio.create_task(sandbox.exec_shell(command))
                await asyncio.sleep(0.2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await assert_removed(sandbox, label)

    asyncio.run(run())


def test_cancel_start_after_container_created(sandbox, monkeypatch):
    real_run = sandbox._run_docker

    async def run():
        created = asyncio.Event()

        async def delayed_run(*args, **kwargs):
            result = await real_run(*args, **kwargs)
            if args[0] == "run":
                assert result.exit_code == 0, result.stderr
                created.set()
                await asyncio.Future()
            return result

        monkeypatch.setattr(sandbox, "_run_docker", delayed_run)
        task = asyncio.create_task(sandbox.start())
        try:
            await asyncio.wait_for(created.wait(), timeout=30)
            label = sandbox._cleanup_label
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await assert_removed(sandbox, label)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await sandbox.stop()

    asyncio.run(run())


def test_name_collision_keeps_existing_container(sandbox):
    async def run():
        async with sandbox:
            other = DockerSandbox(image=sandbox.image, pull_policy="never", container_name=sandbox._container_name)
            try:
                with pytest.raises(RuntimeError, match="already in use"):
                    await other.start()
                assert await sandbox.is_alive()
                assert other._cleanup_label is None
            finally:
                await other.stop()

    asyncio.run(run())


@pytest.mark.parametrize("oracle", [False, True], ids=["no-patch", "gold-patch"])
def test_real_swe_bench_verifier(monkeypatch, oracle):
    """Use an explicit public dataset row and its pre-pulled canonical image."""
    sample_path = os.getenv("UNI_AGENT_SWE_BENCH_SAMPLE")
    if not sample_path:
        pytest.skip("set UNI_AGENT_SWE_BENCH_SAMPLE to a raw SWE-bench Verified row in JSON")
    from uni_agent.tasks.config import TaskConfigResolver
    from uni_agent.tasks.swe_bench.preprocess import get_image_name

    sample = json.loads(Path(sample_path).read_text())
    recipe = Path(__file__).resolve().parents[3] / "examples/quickstart/oracle/task_config_docker.yaml"
    config = TaskConfigResolver.from_file(str(recipe)).resolve(
        {
            "name": "swe_bench",
            "metadata": sample,
            "run_oracle_solution": oracle,
            "sandbox": {"image": get_image_name(sample["instance_id"]), "sandbox_kwargs": {"pull_policy": "never"}},
        }
    )
    task = SWEBenchTask(SWEBenchTaskConfig(**config))
    sandbox = task.build_sandbox()
    monkeypatch.setattr(task, "build_sandbox", lambda: sandbox)
    agent = AsyncMock()
    agent.run.return_value = AgentResult(finished=True)
    monkeypatch.setattr(task, "build_agent", lambda: agent)

    async def run():
        result = await task.run()
        assert result.reward == float(oracle), result.extra_info
        assert result.extra_info["eval_report"]["found_eval_status"], result.extra_info
        assert sandbox._cleanup_label is None
        assert not await sandbox.is_alive()

    asyncio.run(run())
