from __future__ import annotations

import asyncio

import pytest

from uni_agent.sandbox import ExecResult, Sandbox, SandboxConfig, build_sandbox


class _LinkSandbox(Sandbox):
    def __init__(self, *, resolved_path: str = "/usr/bin/claude") -> None:
        self.resolved_path = resolved_path
        self.calls: list[list[str]] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def _exec(self, argv, *, timeout=None, workdir=None, env=None) -> ExecResult:
        self.calls.append(argv)
        if argv[:2] == ["bash", "-c"]:
            return ExecResult(0, f"{self.resolved_path}\n", "")
        return ExecResult(0, "", "")


@pytest.mark.cpu
@pytest.mark.level0
def test_executable_paths_are_validated_and_normalized():
    config = SandboxConfig(
        provider="docker",
        image="example/task:latest",
        executable_paths={"claude": "/opt/claude-code/bin/claude"},
    )

    assert config.executable_paths == {"claude": "/opt/claude-code/bin/claude"}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "executable_paths",
    [
        {"bad/name": "/opt/tool"},
        {"with space": "/opt/tool"},
        {"tool": "relative/tool"},
        {"tool": "/"},
        {"tool": "/opt/../tool"},
    ],
)
def test_executable_paths_reject_unsafe_names_and_paths(executable_paths):
    with pytest.raises(ValueError, match="executable"):
        SandboxConfig(provider="docker", image="example/task:latest", executable_paths=executable_paths)


@pytest.mark.cpu
@pytest.mark.level0
def test_setup_executable_paths_force_links_into_usr_bin():
    sandbox = _LinkSandbox()

    asyncio.run(sandbox._setup_executable_paths({"claude": "/opt/claude-code/bin/claude"}))

    assert sandbox.calls == [
        ["ln", "-sfn", "/opt/claude-code/bin/claude", "/usr/bin/claude"],
        ["bash", "-c", "command -v claude"],
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_setup_executable_paths_rejects_a_higher_priority_command():
    sandbox = _LinkSandbox(resolved_path="/usr/local/bin/claude")

    with pytest.raises(RuntimeError, match="/usr/local/bin/claude.*expected.*/usr/bin/claude"):
        asyncio.run(sandbox._setup_executable_paths({"claude": "/opt/claude-code/bin/claude"}))

    assert sandbox.calls[-1] == ["bash", "-c", "command -v claude"]


@pytest.mark.cpu
@pytest.mark.level0
def test_local_rejects_executable_path_overrides():
    config = SandboxConfig(provider="local", executable_paths={"tmux": "/opt/tools/bin/tmux"})

    with pytest.raises(NotImplementedError, match="does not support executable path overrides"):
        build_sandbox(config)
