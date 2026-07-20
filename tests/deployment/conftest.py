from __future__ import annotations

import pytest

from uni_agent.deployment.local import deployment as mod


@pytest.fixture(autouse=True)
def fixed_ports(monkeypatch):
    """Sandboxes may deny socket.bind; tests never need real free ports."""
    ports = iter(range(20000, 29999))
    monkeypatch.setattr(mod, "_pick_free_port", lambda: next(ports))
