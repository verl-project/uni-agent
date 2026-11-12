"""E2B options for backends that provide templates and tool mounts themselves."""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from uni_agent.sandbox import SandboxConfig, build_sandbox

MOUNTS = [{"image": "registry.local/opencode-tool:1.18.18", "mount_path": "/opt/opencode"}]


def _fake_e2b(monkeypatch, *, missing=()):
    created: dict = {}
    commands: list[str] = []

    class _CommandExitException(Exception):
        pass

    class _Handle:
        def __init__(self, exit_code):
            self.exit_code = exit_code

        async def wait(self):
            return SimpleNamespace(exit_code=self.exit_code, stdout="", stderr="")

        async def kill(self):
            return None

    class _Commands:
        async def run(self, command, **kwargs):
            commands.append(command)
            return _Handle(1 if any(path in command for path in missing) else 0)

    class _AsyncSandbox:
        @classmethod
        async def create(cls, **kwargs):
            created.update(kwargs)

            async def _kill():
                created["killed"] = True

            return SimpleNamespace(sandbox_id="sb-1", commands=_Commands(), kill=_kill)

    def _no_template_build(*args, **kwargs):
        raise AssertionError("image_as_template must not build a template")

    template_api = SimpleNamespace(exists=_no_template_build, build=_no_template_build)
    monkeypatch.setitem(
        sys.modules,
        "e2b",
        SimpleNamespace(
            AsyncSandbox=_AsyncSandbox,
            CommandExitException=_CommandExitException,
            Template=template_api,
        ),
    )
    return created, commands


def _config(**sandbox_kwargs):
    return SandboxConfig(
        provider="e2b",
        image="registry.local/swe-rebench/task-a:latest",
        image_mounts=MOUNTS,
        sandbox_kwargs=sandbox_kwargs,
    )


@pytest.mark.cpu
@pytest.mark.level0
def test_backend_mounts_use_image_as_template_and_verify_mount(monkeypatch):
    created, commands = _fake_e2b(monkeypatch)
    sandbox = build_sandbox(
        _config(
            backend_image_mounts=True,
            image_as_template=True,
            image_mounts_metadata_key="uni-agent.image-mounts",
            metadata={"owner": "test"},
        )
    )

    asyncio.run(sandbox.start())

    assert created["template"] == "registry.local/swe-rebench/task-a:latest"
    assert created["metadata"]["owner"] == "test"
    assert json.loads(created["metadata"]["uni-agent.image-mounts"]) == MOUNTS
    assert any("test -d /opt/opencode" in command for command in commands)
    assert sandbox.metadata == {"owner": "test"}


@pytest.mark.cpu
@pytest.mark.level0
def test_backend_mounts_without_metadata_key_send_metadata_unchanged(monkeypatch):
    created, _ = _fake_e2b(monkeypatch)
    sandbox = build_sandbox(_config(backend_image_mounts=True, template="prepared-template"))

    asyncio.run(sandbox.start())

    assert created["template"] == "prepared-template"
    assert created["metadata"] == {}


@pytest.mark.cpu
@pytest.mark.level0
def test_missing_backend_mount_fails_startup(monkeypatch):
    _fake_e2b(monkeypatch, missing=("/opt/opencode",))
    sandbox = build_sandbox(_config(backend_image_mounts=True, image_as_template=True))

    with pytest.raises(RuntimeError, match="missing backend image mount '/opt/opencode'"):
        asyncio.run(sandbox.start())


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"image_mounts_metadata_key": "mounts"}, "requires backend_image_mounts"),
        ({"backend_image_mounts": True, "image_mounts_metadata_key": " "}, "must be non-empty"),
    ],
)
def test_invalid_backend_mount_options(kwargs, error):
    with pytest.raises(ValueError, match=error):
        build_sandbox(_config(**kwargs))


@pytest.mark.cpu
@pytest.mark.level0
def test_image_as_template_requires_image():
    from uni_agent.sandbox.e2b import E2BSandbox

    with pytest.raises(ValueError, match="image_as_template requires an image"):
        E2BSandbox(template="prepared-template", image_as_template=True)
