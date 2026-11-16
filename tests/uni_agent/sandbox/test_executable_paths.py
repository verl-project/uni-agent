from __future__ import annotations

import asyncio
import logging
import sys
from types import SimpleNamespace

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


@pytest.mark.cpu
@pytest.mark.level0
def test_e2b_receives_executable_path_overrides_from_config():
    config = SandboxConfig(
        provider="e2b",
        image="python:3.12",
        executable_paths={"claude": "/opt/claude-code/bin/claude"},
    )

    sandbox = build_sandbox(config)

    assert sandbox.executable_paths == {"claude": "/opt/claude-code/bin/claude"}


@pytest.mark.cpu
@pytest.mark.level0
def test_e2b_defaults_to_root_user():
    from uni_agent.sandbox.e2b import E2BSandbox

    assert E2BSandbox(template="ready-template").user == "root"


@pytest.mark.cpu
@pytest.mark.level0
def test_e2b_sets_up_executable_paths_as_root_by_default(monkeypatch):
    from uni_agent.sandbox.e2b import E2BSandbox

    calls = []

    class _CommandExitException(Exception):
        pass

    class _Handle:
        def __init__(self, result):
            self.result = result

        async def wait(self):
            return self.result

    class _Commands:
        async def run(self, command, **kwargs):
            calls.append((command, kwargs["user"]))
            stdout = "/usr/bin/claude\n" if "command -v claude" in command else ""
            return _Handle(SimpleNamespace(exit_code=0, stdout=stdout, stderr=""))

    class _AsyncSandbox:
        @classmethod
        async def create(cls, **kwargs):
            return SimpleNamespace(sandbox_id="sb-1", commands=_Commands())

    monkeypatch.setitem(
        sys.modules,
        "e2b",
        SimpleNamespace(
            AsyncSandbox=_AsyncSandbox,
            CommandExitException=_CommandExitException,
        ),
    )
    sandbox = E2BSandbox(
        template="ready-template",
        executable_paths={"claude": "/opt/claude-code/bin/claude"},
    )

    asyncio.run(sandbox.start())

    assert calls == [
        ("exec bash -c 'command -v claude'", "root"),
        ("exec ln -sfn /opt/claude-code/bin/claude /usr/bin/claude", "root"),
        ("exec bash -c 'command -v claude'", "root"),
        ("exec test /usr/bin/claude -ef /opt/claude-code/bin/claude", "root"),
    ]
    assert sandbox.user == "root"
