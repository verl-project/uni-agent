"""Shared fixtures for the framework CPU tests."""

from __future__ import annotations

import pytest


class FakeTransferQueue:
    """Minimal stand-in for the verl TransferQueue client."""

    def __init__(self):
        self.puts = []
        self.batch_puts = []
        self.events = []

    async def async_kv_put(self, *, key, partition_id, tag):
        self.puts.append({"key": key, "partition_id": partition_id, "tag": dict(tag)})
        self.events.append(("put", key, tag.get("status")))

    async def async_kv_batch_put(self, *, keys, fields, tags, partition_id):
        self.batch_puts.append(
            {
                "keys": list(keys),
                "fields": fields,
                "tags": [dict(tag) for tag in tags],
                "partition_id": partition_id,
            }
        )
        self.events.append(("batch_put", tuple(keys)))


@pytest.fixture
def fake_tq(monkeypatch):
    from uni_agent.framework import framework as framework_module

    fake = FakeTransferQueue()
    monkeypatch.setattr(framework_module, "tq", fake)
    monkeypatch.setattr(framework_module, "_session_tq_schema", lambda: None)
    return fake
