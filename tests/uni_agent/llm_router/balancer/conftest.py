"""conftest for balancer tests.

Applies the _FakeProvider patch ONLY when balancer ut tests are being run.
When st-cpu/e2e tests run (different pytest invocation, different -m filter),
no balancer ut tests are selected, so the patch is a no-op.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _conditional_patch(request):
    """Patch RouteDataProvider + _init_provider — only if balancer ut tests run."""
    has_balancer_ut = any(
        "balancer" in str(item.fspath) and item.get_closest_marker("ut") for item in request.session.items
    )
    if not has_balancer_ut:
        yield
        return

    import uni_agent.llm_router.collectors as _collectors_mod
    from tests.uni_agent.llm_router.balancer._helpers import (
        _fake_init_provider,
        _FakeProvider,
    )
    from uni_agent.llm_router.balancer import KVCAwareBalancer

    _orig_provider = _collectors_mod.RouteDataProvider
    _orig_init = KVCAwareBalancer._init_provider

    _collectors_mod.RouteDataProvider = _FakeProvider
    KVCAwareBalancer._init_provider = _fake_init_provider

    yield

    _collectors_mod.RouteDataProvider = _orig_provider
    KVCAwareBalancer._init_provider = _orig_init
