"""External provider discovery and failed-import isolation."""

import sys

import pytest

from uni_agent.sandbox.registry import SANDBOX_REGISTRY, get_sandbox_cls


@pytest.fixture
def plugin(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("UNI_AGENT_SANDBOX_PLUGINS", "external_sandbox")
    original = dict(SANDBOX_REGISTRY)
    yield tmp_path / "external_sandbox.py"
    SANDBOX_REGISTRY.clear()
    SANDBOX_REGISTRY.update(original)
    sys.modules.pop("external_sandbox", None)


@pytest.mark.cpu
@pytest.mark.level0
def test_plugin_discovery_and_missing_module_retry(plugin, monkeypatch):
    plugin.write_text(
        "from uni_agent.sandbox.local import LocalSandbox\n"
        "from uni_agent.sandbox.registry import register_sandbox\n"
        '@register_sandbox("external")\n'
        "class ExternalSandbox(LocalSandbox): pass\n"
    )
    monkeypatch.setenv("UNI_AGENT_SANDBOX_PLUGINS", " external_sandbox, missing_sandbox_plugin ")
    for _ in range(2):
        with pytest.raises(ImportError, match="missing_sandbox_plugin"):
            get_sandbox_cls("external")
    monkeypatch.setenv("UNI_AGENT_SANDBOX_PLUGINS", "external_sandbox")
    assert get_sandbox_cls("external").__name__ == "ExternalSandbox"


@pytest.mark.cpu
@pytest.mark.level0
def test_failed_plugin_registration_is_rolled_back(plugin):
    plugin.write_text(
        "from uni_agent.sandbox.local import LocalSandbox\n"
        "from uni_agent.sandbox.registry import register_sandbox\n"
        '@register_sandbox("external")\n'
        "class ExternalSandbox(LocalSandbox): pass\n"
        "import missing_sandbox_sdk\n"
    )
    for _ in range(2):
        with pytest.raises(ImportError, match="external_sandbox"):
            get_sandbox_cls("external")
        assert "external" not in SANDBOX_REGISTRY


@pytest.mark.cpu
@pytest.mark.level0
def test_plugin_cannot_replace_builtin(plugin):
    plugin.write_text(
        "from uni_agent.sandbox.local import LocalSandbox\n"
        "from uni_agent.sandbox.registry import register_sandbox\n"
        '@register_sandbox("docker")\n'
        "class ExternalSandbox(LocalSandbox): pass\n"
    )
    with pytest.raises(ValueError, match="reserved"):
        get_sandbox_cls("external")
    assert get_sandbox_cls("local").__module__ == "uni_agent.sandbox.local"
