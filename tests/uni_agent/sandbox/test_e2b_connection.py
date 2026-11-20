"""Regression tests for per-call E2B connection parameters in ``E2BSandbox``.

The ``agentenv`` provider needs ``api_key`` / ``api_url`` / ``sandbox_url`` /
``domain`` / ``http_version`` / ``proxy`` to reach every E2B SDK control-plane
call. These tests pin the refactor's compatibility contract with a fake
``sys.modules["e2b"]`` that records the exact keyword arguments:

* nothing configured -> ``AsyncSandbox.create`` / ``Template.build`` receive
  exactly the pre-refactor kwargs (no stray connection keys);
* ``proxy`` keeps being forwarded to both;
* every configured connection parameter is forwarded to both sandbox creation
  and template building.

No network, no credentials, no real SDK behaviour involved.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from uni_agent.sandbox.e2b import E2BSandbox

#: The pre-refactor ``AsyncSandbox.create`` kwargs for a template-backed sandbox.
_LEGACY_CREATE_KWARGS: dict[str, Any] = {
    "timeout": 3600,
    "metadata": {},
    "envs": {},
    "secure": True,
    "allow_internet_access": True,
}

_CONNECTION_PARAMS = ("api_key", "api_url", "sandbox_url", "domain", "http_version")


def _install_fake_e2b(monkeypatch: pytest.MonkeyPatch, *, template_exists: bool = False) -> dict[str, list[Any]]:
    """Inject a recording fake ``e2b`` module and return the recorded calls."""
    calls: dict[str, list[Any]] = {"create": [], "exists": [], "build": [], "from_image": []}

    class _FakeSandboxHandle:
        sandbox_id = "e2b-fake-sandbox"

        async def kill(self) -> None:
            return None

        async def is_running(self) -> bool:
            return True

    class _FakeTemplate:
        def from_image(self, image: str) -> str:
            calls["from_image"].append(image)
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


@pytest.mark.cpu
@pytest.mark.level0
def test_no_connection_params_keeps_legacy_create_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_e2b(monkeypatch)

    sandbox = E2BSandbox(template="ready-template")
    assert sandbox._connection_kwargs == {}
    asyncio.run(sandbox.start())

    assert calls["create"] == [{"template": "ready-template", **_LEGACY_CREATE_KWARGS}]
    assert calls["exists"] == []
    assert calls["build"] == []


@pytest.mark.cpu
@pytest.mark.level0
def test_proxy_is_still_forwarded_to_create(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_e2b(monkeypatch)

    sandbox = E2BSandbox(template="ready-template", proxy="http://proxy.local:3128")
    # ``proxy`` remains a public attribute and now also rides in the connection kwargs.
    assert sandbox.proxy == "http://proxy.local:3128"
    assert sandbox._connection_kwargs == {"proxy": "http://proxy.local:3128"}
    asyncio.run(sandbox.start())

    assert calls["create"] == [
        {"template": "ready-template", **_LEGACY_CREATE_KWARGS, "proxy": "http://proxy.local:3128"}
    ]


@pytest.mark.cpu
@pytest.mark.level0
def test_image_path_without_connection_params_matches_legacy_template_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_e2b(monkeypatch)

    sandbox = E2BSandbox(image="python:3.12")
    asyncio.run(sandbox.start())

    assert calls["from_image"] == ["python:3.12"]
    assert len(calls["exists"]) == 1
    template_name = calls["exists"][0]["name"]
    assert template_name.startswith("uni-agent-python-3-12-")
    # No connection kwargs at all -> exactly the two documented SDK calls.
    assert calls["exists"][0]["kwargs"] == {}
    assert calls["build"] == [
        {
            "template": "template-from-image:python:3.12",
            "name": template_name,
            "kwargs": {"cpu_count": 2, "memory_mb": 4096, "on_build_logs": None},
        }
    ]
    assert calls["create"] == [{"template": template_name, **_LEGACY_CREATE_KWARGS}]


@pytest.mark.cpu
@pytest.mark.level0
def test_image_path_forwards_proxy_to_template_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_e2b(monkeypatch)

    sandbox = E2BSandbox(image="python:3.12", proxy="http://proxy.local:3128")
    asyncio.run(sandbox.start())

    assert calls["exists"][0]["kwargs"] == {"proxy": "http://proxy.local:3128"}
    assert calls["build"][0]["kwargs"] == {
        "cpu_count": 2,
        "memory_mb": 4096,
        "on_build_logs": None,
        "proxy": "http://proxy.local:3128",
    }
    assert calls["create"][0]["proxy"] == "http://proxy.local:3128"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("name", _CONNECTION_PARAMS)
def test_single_connection_param_is_forwarded_to_create(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    calls = _install_fake_e2b(monkeypatch)
    value = "1.1" if name == "http_version" else f"test-{name}-value"

    sandbox = E2BSandbox(template="ready-template", **{name: value})
    assert sandbox._connection_kwargs == {name: value}
    asyncio.run(sandbox.start())

    assert calls["create"] == [{"template": "ready-template", **_LEGACY_CREATE_KWARGS, name: value}]


@pytest.mark.cpu
@pytest.mark.level0
def test_all_connection_params_are_forwarded_to_create_and_template_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_e2b(monkeypatch)
    connection_kwargs = {
        "api_key": "test-api-key-not-real",
        "api_url": "https://api.agentenv.test",
        "sandbox_url": "https://sandbox.agentenv.test",
        "domain": "agentenv.test",
        "http_version": "1.1",
    }

    sandbox = E2BSandbox(image="python:3.12", **connection_kwargs)
    assert sandbox._connection_kwargs == connection_kwargs
    asyncio.run(sandbox.start())

    assert calls["exists"][0]["kwargs"] == connection_kwargs
    assert calls["build"][0]["kwargs"] == {
        "cpu_count": 2,
        "memory_mb": 4096,
        "on_build_logs": None,
        **connection_kwargs,
    }
    assert calls["create"] == [{"template": calls["exists"][0]["name"], **_LEGACY_CREATE_KWARGS, **connection_kwargs}]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("name", _CONNECTION_PARAMS)
def test_blank_connection_param_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match=f"{name} must be non-empty"):
        E2BSandbox(template="ready-template", **{name: "   "})
