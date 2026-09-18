"""CPU checks for ``UNI_AGENT_SANDBOX_PLUGINS`` overlay discovery."""

from __future__ import annotations

import pytest

from uni_agent.sandbox.registry import (
    SANDBOX_PLUGIN_MODULES_ENV,
    SANDBOX_REGISTRY,
    get_sandbox_cls,
)

_PROBE_MODULE = "tests.uni_agent.sandbox.plugin_probe_provider"
_PROBE_NAME = "plugin_probe"


@pytest.fixture(autouse=True)
def _isolate_probe_registry():
    SANDBOX_REGISTRY.pop(_PROBE_NAME, None)
    yield
    SANDBOX_REGISTRY.pop(_PROBE_NAME, None)


@pytest.mark.cpu
@pytest.mark.level0
def test_env_plugin_registers_provider(monkeypatch):
    monkeypatch.setenv(SANDBOX_PLUGIN_MODULES_ENV, _PROBE_MODULE)
    cls = get_sandbox_cls(_PROBE_NAME)
    assert cls.__name__ == "PluginProbeSandbox"
    assert cls.__module__ == _PROBE_MODULE


@pytest.mark.cpu
@pytest.mark.level0
def test_missing_plugin_module_fails_closed(monkeypatch):
    monkeypatch.setenv(SANDBOX_PLUGIN_MODULES_ENV, "not_a_real_sandbox_plugin_module")
    with pytest.raises(ImportError, match="not_a_real_sandbox_plugin_module"):
        get_sandbox_cls(_PROBE_NAME)


@pytest.mark.cpu
@pytest.mark.level0
def test_unknown_provider_stays_fail_closed(monkeypatch):
    monkeypatch.delenv(SANDBOX_PLUGIN_MODULES_ENV, raising=False)
    with pytest.raises(ValueError, match="Unknown sandbox provider"):
        get_sandbox_cls("not-a-real-provider")
