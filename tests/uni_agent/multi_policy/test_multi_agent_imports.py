"""Smoke tests for examples.multi_policy.framework module imports and basic structure."""

import pytest


@pytest.mark.cpu
@pytest.mark.level0
def test_multi_agent_module_exports():
    from examples.multi_policy.framework import (
        MultiAgentGatewayFramework,
        MultiAgentGatewayManager,
    )

    assert MultiAgentGatewayManager is not None
    assert MultiAgentGatewayFramework is not None


@pytest.mark.cpu
@pytest.mark.level0
def test_manager_inherits_from_gateway_manager():
    from uni_agent.gateway.manager import GatewayManager
    from examples.multi_policy.framework import MultiAgentGatewayManager

    assert issubclass(MultiAgentGatewayManager, GatewayManager)


@pytest.mark.cpu
@pytest.mark.level0
def test_framework_inherits_from_gateway_agent_framework():
    from uni_agent.framework.framework import GatewayAgentFramework
    from examples.multi_policy.framework import MultiAgentGatewayFramework

    assert issubclass(MultiAgentGatewayFramework, GatewayAgentFramework)
