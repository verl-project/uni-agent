"""Unit tests for the ``agentenv`` E2B-compatible sandbox provider.

Every E2B SDK call is replaced by an in-memory fake injected through
``sys.modules["e2b"]``, so these tests never touch the network, credentials or
the real SDK's wire behaviour. The fake records the exact keyword arguments the
provider hands to ``AsyncSandbox.create`` / ``Template.exists`` / ``Template.build``
so the endpoint/credential plumbing can be asserted precisely.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from uni_agent.sandbox import Sandbox, SandboxConfig, build_sandbox

#: Every env var the provider may read, cleared for a deterministic baseline.
_CONNECTION_ENV_VARS = (
    "AGENTENV_API_URL",
    "AGENTENV_SANDBOX_URL",
    "AGENTENV_API_KEY",
    "E2B_API_URL",
    "E2B_SANDBOX_URL",
    "E2B_API_KEY",
)

#: Clearly fake values: never real credentials, only sentinel strings.
_AGENTENV_API_URL = "https://api.agentenv.test"
_AGENTENV_SANDBOX_URL = "https://sandbox.agentenv.test"
_AGENTENV_API_KEY = "test-api-key-not-real"

_BASE_CREATE_KWARGS: dict[str, Any] = {
    "timeout": 3600,
    "metadata": {},
    "envs": {},
    "secure": True,
    "allow_internet_access": True,
}


@pytest.fixture
def clean_connection_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear AGENTENV_*/E2B_* so each test starts from a known configuration."""
    for name in _CONNECTION_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _install_fake_e2b(
    monkeypatch: pytest.MonkeyPatch,
    *,
    template_exists: bool = False,
    create_error: BaseException | None = None,
) -> dict[str, list[Any]]:
    """Inject a fake ``e2b`` module and return the recorded SDK calls."""
    calls: dict[str, list[Any]] = {"create": [], "exists": [], "build": [], "from_image": []}

    class _FakeSandboxHandle:
        sandbox_id = "agentenv-fake-sandbox"

        async def kill(self) -> None:
            return None

        async def is_running(self) -> bool:
            return True

    class _FakeTemplate:
        def __init__(self) -> None:
            self.calls = calls

        def from_image(self, image: str) -> str:
            self.calls["from_image"].append(image)
            return f"template-from-image:{image}"

        @staticmethod
        def exists(name: str, **kwargs: Any) -> bool:
            calls["exists"].append({"name": name, "kwargs": kwargs})
            return template_exists

        @staticmethod
        def build(template: Any, name: str, **kwargs: Any) -> None:
            calls["build"].append({"template": template, "name": name, "kwargs": kwargs})

    class _FakeAsyncSandbox:
        @staticmethod
        async def create(**kwargs: Any) -> _FakeSandboxHandle:
            calls["create"].append(kwargs)
            if create_error is not None:
                raise create_error
            return _FakeSandboxHandle()

    monkeypatch.setitem(
        sys.modules,
        "e2b",
        SimpleNamespace(
            AsyncSandbox=_FakeAsyncSandbox,
            Template=_FakeTemplate,
            CommandExitException=type("CommandExitException", (Exception,), {}),
            TimeoutException=type("TimeoutException", (Exception,), {}),
        ),
    )
    return calls


def _set_agentenv_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTENV_API_URL", _AGENTENV_API_URL)
    monkeypatch.setenv("AGENTENV_SANDBOX_URL", _AGENTENV_SANDBOX_URL)
    monkeypatch.setenv("AGENTENV_API_KEY", _AGENTENV_API_KEY)


@pytest.mark.cpu
@pytest.mark.level0
def test_env_fallback_feeds_connection_kwargs_to_create(
    monkeypatch: pytest.MonkeyPatch, clean_connection_env: None
) -> None:
    calls = _install_fake_e2b(monkeypatch)
    _set_agentenv_env(monkeypatch)

    sandbox = build_sandbox(SandboxConfig(provider="agentenv", sandbox_kwargs={"template": "uni-agent-swe"}))
    asyncio.run(sandbox.start())

    assert calls["create"] == [
        {
            "template": "uni-agent-swe",
            **_BASE_CREATE_KWARGS,
            "api_url": _AGENTENV_API_URL,
            "sandbox_url": _AGENTENV_SANDBOX_URL,
            "api_key": _AGENTENV_API_KEY,
        }
    ]
    # A pre-built template alias skips the template-build API entirely.
    assert calls["exists"] == []
    assert calls["build"] == []


@pytest.mark.cpu
@pytest.mark.level0
def test_e2b_env_vars_are_the_documented_fallback(monkeypatch: pytest.MonkeyPatch, clean_connection_env: None) -> None:
    monkeypatch.setenv("E2B_API_URL", "https://api.e2b-fallback.test")
    monkeypatch.setenv("E2B_SANDBOX_URL", "https://sandbox.e2b-fallback.test")
    monkeypatch.setenv("E2B_API_KEY", "e2b-fallback-key-not-real")

    sandbox = build_sandbox(SandboxConfig(provider="agentenv", sandbox_kwargs={"template": "t"}))

    assert sandbox._connection_kwargs == {
        "api_url": "https://api.e2b-fallback.test",
        "sandbox_url": "https://sandbox.e2b-fallback.test",
        "api_key": "e2b-fallback-key-not-real",
    }


@pytest.mark.cpu
@pytest.mark.level0
def test_agentenv_env_vars_win_over_e2b_env_vars(monkeypatch: pytest.MonkeyPatch, clean_connection_env: None) -> None:
    monkeypatch.setenv("E2B_API_URL", "https://api.e2b-fallback.test")
    monkeypatch.setenv("E2B_SANDBOX_URL", "https://sandbox.e2b-fallback.test")
    monkeypatch.setenv("E2B_API_KEY", "e2b-fallback-key-not-real")
    _set_agentenv_env(monkeypatch)

    sandbox = build_sandbox(SandboxConfig(provider="agentenv", sandbox_kwargs={"template": "t"}))

    assert sandbox._connection_kwargs == {
        "api_url": _AGENTENV_API_URL,
        "sandbox_url": _AGENTENV_SANDBOX_URL,
        "api_key": _AGENTENV_API_KEY,
    }


@pytest.mark.cpu
@pytest.mark.level0
def test_explicit_sandbox_kwargs_win_over_env(monkeypatch: pytest.MonkeyPatch, clean_connection_env: None) -> None:
    calls = _install_fake_e2b(monkeypatch)
    _set_agentenv_env(monkeypatch)

    sandbox = build_sandbox(
        SandboxConfig(
            provider="agentenv",
            sandbox_kwargs={
                "template": "explicit-template",
                "api_url": "https://api.explicit.test",
                "sandbox_url": "https://sandbox.explicit.test",
                "api_key": "explicit-key-not-real",
            },
        )
    )
    asyncio.run(sandbox.start())

    assert calls["create"] == [
        {
            "template": "explicit-template",
            **_BASE_CREATE_KWARGS,
            "api_url": "https://api.explicit.test",
            "sandbox_url": "https://sandbox.explicit.test",
            "api_key": "explicit-key-not-real",
        }
    ]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("missing", "expected_env_var"),
    [
        ("api_url", "AGENTENV_API_URL"),
        ("sandbox_url", "AGENTENV_SANDBOX_URL"),
        ("api_key", "AGENTENV_API_KEY"),
    ],
)
def test_missing_connection_setting_names_the_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
    clean_connection_env: None,
    missing: str,
    expected_env_var: str,
) -> None:
    for name in ("api_url", "sandbox_url", "api_key"):
        if name != missing:
            monkeypatch.setenv(f"AGENTENV_{name.upper()}", f"https://should-not-leak-{name}.test")

    with pytest.raises(ValueError) as excinfo:
        build_sandbox(SandboxConfig(provider="agentenv", sandbox_kwargs={"template": "t"}))

    message = str(excinfo.value)
    assert expected_env_var in message
    assert f"sandbox_kwargs.{missing}" in message
    # Values are never echoed, only names.
    assert "should-not-leak" not in message


@pytest.mark.cpu
@pytest.mark.level0
def test_all_missing_connection_settings_are_listed(
    monkeypatch: pytest.MonkeyPatch, clean_connection_env: None
) -> None:
    with pytest.raises(ValueError) as excinfo:
        build_sandbox(SandboxConfig(provider="agentenv", sandbox_kwargs={"template": "t"}))

    message = str(excinfo.value)
    for env_var in ("AGENTENV_API_URL", "E2B_API_URL", "AGENTENV_SANDBOX_URL", "E2B_SANDBOX_URL"):
        assert env_var in message
    for env_var in ("AGENTENV_API_KEY", "E2B_API_KEY"):
        assert env_var in message


@pytest.mark.cpu
@pytest.mark.level0
def test_image_path_builds_template_with_the_same_connection_kwargs(
    monkeypatch: pytest.MonkeyPatch, clean_connection_env: None
) -> None:
    calls = _install_fake_e2b(monkeypatch)
    _set_agentenv_env(monkeypatch)

    sandbox = build_sandbox(SandboxConfig(provider="agentenv", image="python:3.12"))
    asyncio.run(sandbox.start())

    expected_connection_kwargs = {
        "api_url": _AGENTENV_API_URL,
        "sandbox_url": _AGENTENV_SANDBOX_URL,
        "api_key": _AGENTENV_API_KEY,
    }
    assert len(calls["exists"]) == 1
    assert calls["exists"][0]["kwargs"] == expected_connection_kwargs
    assert calls["build"][0]["kwargs"] == {
        "cpu_count": 2,
        "memory_mb": 4096,
        "on_build_logs": None,
        **expected_connection_kwargs,
    }
    assert calls["from_image"] == ["python:3.12"]
    assert calls["create"][0]["template"] == calls["exists"][0]["name"]
    assert calls["create"][0]["api_key"] == _AGENTENV_API_KEY


@pytest.mark.cpu
@pytest.mark.level0
def test_template_wins_over_image(monkeypatch: pytest.MonkeyPatch, clean_connection_env: None) -> None:
    calls = _install_fake_e2b(monkeypatch)
    _set_agentenv_env(monkeypatch)

    sandbox = build_sandbox(
        SandboxConfig(provider="agentenv", image="python:3.12", sandbox_kwargs={"template": "prebuilt-template"})
    )
    asyncio.run(sandbox.start())

    assert calls["create"][0]["template"] == "prebuilt-template"
    assert calls["exists"] == []
    assert calls["build"] == []


@pytest.mark.cpu
@pytest.mark.level0
def test_api_key_never_appears_in_start_failure(monkeypatch: pytest.MonkeyPatch, clean_connection_env: None) -> None:
    monkeypatch.setenv("AGENTENV_API_URL", _AGENTENV_API_URL)
    monkeypatch.setenv("AGENTENV_SANDBOX_URL", _AGENTENV_SANDBOX_URL)
    monkeypatch.setenv("AGENTENV_API_KEY", "super-secret-agentenv-key")
    _install_fake_e2b(monkeypatch, create_error=RuntimeError("agentenv control plane unreachable"))

    sandbox = build_sandbox(SandboxConfig(provider="agentenv", sandbox_kwargs={"template": "t"}))

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(sandbox.start())

    assert "super-secret-agentenv-key" not in str(excinfo.value)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("template, fail_connect", [(None, False), ("task-template", False), (None, True)])
async def test_agentenv_mounts_images_at_creation(
    monkeypatch: pytest.MonkeyPatch, template: str | None, fail_connect: bool
) -> None:
    import json

    import httpx

    _set_agentenv_env(monkeypatch)
    _install_fake_e2b(monkeypatch)
    sdk = sys.modules["e2b"]
    events: list[str] = []
    path = "/sandboxes" if template else "/sandboxes-cold"

    def control_plane(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-Key"] == _AGENTENV_API_KEY
        events.append(f"{request.method} {request.url.path}")
        if request.method == "DELETE":
            return httpx.Response(204)
        payload = json.loads(request.content)
        assert payload["attachedDrives"] == [
            {
                "driveID": "image_mount_0",
                "source": {"image": "registry.test/dsh-tool:latest"},
                "mountPath": "/opt/dsh",
                "readOnly": True,
            },
        ]
        assert payload["network"] == {"denyOut": ["0.0.0.0/0"]}
        if template:
            assert payload["templateID"] == template
            assert "image" not in payload
        else:
            assert payload["image"] == "registry.test/swebench/example"
            assert "templateID" not in payload
        return httpx.Response(201, json={"sandboxID": "mounted-sandbox"})

    real_client = httpx.AsyncClient

    def test_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(control_plane), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", test_client)

    async def connect(sandbox_id: str, **kwargs: Any) -> Any:
        events.append(f"connect {sandbox_id}")
        assert kwargs["sandbox_url"] == _AGENTENV_SANDBOX_URL
        if fail_connect:
            raise RuntimeError("connect failed")
        return SimpleNamespace(sandbox_id=sandbox_id)

    monkeypatch.setattr(sdk.AsyncSandbox, "connect", connect, raising=False)
    sandbox = build_sandbox(
        SandboxConfig(
            provider="agentenv",
            image="swebench/example",
            image_map=[{"from": "swebench/**", "to": "registry.test/swebench/**"}],
            image_mounts=[{"image": "registry.test/dsh-tool:latest", "mount_path": "/opt/dsh"}],
            sandbox_kwargs={"template": template, "network": {"deny_out": ["0.0.0.0/0"]}},
        )
    )

    if fail_connect:
        with pytest.raises(RuntimeError, match="connect failed"):
            await sandbox.start()
        assert events == [f"POST {path}", "connect mounted-sandbox", "DELETE /sandboxes/mounted-sandbox"]
    else:
        await sandbox.start()
        assert sandbox.sandbox_id == "mounted-sandbox"
        assert events == [f"POST {path}", "connect mounted-sandbox"]


@pytest.mark.cpu
@pytest.mark.level0
def test_agentenv_reuses_e2b_implementation_by_inheritance() -> None:
    from uni_agent.sandbox.agentenv import AgentEnvSandbox
    from uni_agent.sandbox.e2b import E2BSandbox
    from uni_agent.sandbox.registry import get_sandbox_cls

    cls = get_sandbox_cls("agentenv")

    assert cls is AgentEnvSandbox
    assert cls is not E2BSandbox
    assert issubclass(cls, E2BSandbox)
    assert cls.provider == "agentenv"
    # Inheritance, not a copy: the private exec primitive and the shared public
    # exec() wrapper both resolve to the e2b/base implementations.
    assert cls._exec is E2BSandbox._exec
    assert cls.exec is Sandbox.exec

@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_direct_gateway_config_does_not_run_guest_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_agentenv_env(monkeypatch)
    calls = _install_fake_e2b(monkeypatch)
    sandbox = build_sandbox(
        SandboxConfig(provider="agentenv", sandbox_kwargs={"template": "task-template", "proxy_port": 38197})
    )
    await sandbox.start()
    assert await sandbox.is_alive()
    assert len(calls["create"]) == 1
    await sandbox.stop()
