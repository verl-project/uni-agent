"""Request priority stamping for vLLM priority scheduling (``UNI_AGENT_REQUEST_PRIORITY``)."""

import pytest

pytest.importorskip("tensordict")

from uni_agent.framework.framework import _session_metadata  # noqa: E402


@pytest.mark.cpu
@pytest.mark.level0
def test_priority_is_stamped_only_when_enabled(monkeypatch):
    monkeypatch.delenv("UNI_AGENT_REQUEST_PRIORITY", raising=False)
    assert "_request_priority" not in _session_metadata({"a": 1}, None, 7)

    monkeypatch.setenv("UNI_AGENT_REQUEST_PRIORITY", "dispatch_step")
    metadata = _session_metadata({"a": 1}, "/tmp/run", 7, group_order=42)
    assert metadata == {"_trace_identity": {"a": 1}, "_log_dir": "/tmp/run", "_request_priority": 7}
    # Validation runs have no dispatch step and keep the default priority.
    assert "_request_priority" not in _session_metadata({"a": 1}, None, None, group_order=None)

    monkeypatch.setenv("UNI_AGENT_REQUEST_PRIORITY", "group_order")
    metadata_group = _session_metadata({"a": 1}, "/tmp/run", 1, group_order=42)
    assert metadata_group == {"_trace_identity": {"a": 1}, "_log_dir": "/tmp/run", "_request_priority": 42}
    assert "_request_priority" not in _session_metadata({"a": 1}, None, None, group_order=None)
