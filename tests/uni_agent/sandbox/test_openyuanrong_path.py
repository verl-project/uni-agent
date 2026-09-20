"""Unit coverage for OpenYuanrong's non-destructive PATH extension."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from uni_agent.sandbox.openyuanrong import OpenyuanrongSandbox


class _CommandResult:
    exit_code = 0
    stdout = ""
    stderr = ""


class _Commands:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, command: str, **kwargs):
        self.calls.append((command, kwargs))
        return _CommandResult()


class _Shell:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.closed = False

    async def run(self, command: str, *, timeout: int):
        self.calls.append((command, timeout))
        return _CommandResult()

    async def kill(self) -> None:
        self.closed = True


class _Shells:
    def __init__(self, shell: _Shell) -> None:
        self.shell = shell
        self.create_kwargs: dict | None = None

    async def create(self, **kwargs):
        self.create_kwargs = kwargs
        return self.shell


def _sandbox(*, add_to_path: list[str] | None = None) -> tuple[OpenyuanrongSandbox, _Commands, _Shells]:
    commands = _Commands()
    shells = _Shells(_Shell())
    sandbox = OpenyuanrongSandbox(image="example:latest", add_to_path=add_to_path)
    sandbox._sandbox = SimpleNamespace(commands=commands, shells=shells)
    return sandbox, commands, shells


@pytest.mark.cpu
@pytest.mark.level0
def test_exec_prepends_paths_without_overwriting_remote_path():
    sandbox, commands, _ = _sandbox(add_to_path=["/opt/claude-code/bin", "/opt/other/bin"])

    result = asyncio.run(sandbox._exec(["claude", "--version"], env={"TEST": "1"}))

    assert result.exit_code == 0
    assert commands.calls == [
        (
            'export PATH=/opt/claude-code/bin:/opt/other/bin:"${PATH:-}"; claude --version',
            {"envs": {"TEST": "1"}, "cwd": None, "timeout": 60},
        )
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_open_shell_uses_the_same_path_setup_as_exec():
    sandbox, _, shells = _sandbox(add_to_path=["/opt/claude-code/bin"])

    shell = asyncio.run(sandbox.open_shell(cwd="/testbed", env={"TEST": "1"}))

    assert shells.create_kwargs == {"cwd": "/testbed", "envs": {"TEST": "1"}}
    assert shells.shell.calls == [('export PATH=/opt/claude-code/bin:"${PATH:-}"', 60)]
    assert shell is not None


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("value", ["/opt/claude-code/bin", [""], [1]])
def test_add_to_path_requires_non_empty_string_list(value):
    with pytest.raises(ValueError, match="add_to_path"):
        OpenyuanrongSandbox(image="example:latest", add_to_path=value)
