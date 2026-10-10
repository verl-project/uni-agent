from __future__ import annotations

import asyncio
import logging

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
        ["bash", "-c", "command -v claude"],
        ["ln", "-sfn", "/opt/claude-code/bin/claude", "/usr/bin/claude"],
        ["bash", "-c", "command -v claude"],
        ["test", "/usr/bin/claude", "-ef", "/opt/claude-code/bin/claude"],
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_setup_executable_paths_warns_when_overriding_a_higher_priority_command(caplog):
    sandbox = _LinkSandbox(resolved_path="/usr/local/bin/claude")

    with caplog.at_level(logging.WARNING, logger="uni_agent.sandbox.base"):
        asyncio.run(sandbox._setup_executable_paths({"claude": "/opt/claude-code/bin/claude"}))

    assert sandbox.calls == [
        ["bash", "-c", "command -v claude"],
        ["ln", "-sfn", "/opt/claude-code/bin/claude", "/usr/bin/claude"],
        ["ln", "-sfn", "/opt/claude-code/bin/claude", "/usr/local/bin/claude"],
        ["bash", "-c", "command -v claude"],
        ["test", "/usr/local/bin/claude", "-ef", "/opt/claude-code/bin/claude"],
    ]
    assert "overriding sandbox executable 'claude' at /usr/local/bin/claude" in caplog.text


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("provider", "image"),
    [
        ("local", None),
        ("vefaas", "python:3.12"),
    ],
)
def test_unsupported_providers_reject_executable_path_overrides(provider, image):
    config = SandboxConfig(
        provider=provider,
        image=image,
        executable_paths={"tmux": "/opt/tools/bin/tmux"},
    )

    with pytest.raises(NotImplementedError, match="does not support executable path overrides"):
        build_sandbox(config)
