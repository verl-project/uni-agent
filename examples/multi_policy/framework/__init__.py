"""Multi-agent training support.

This module provides manager and framework extensions for multi-agent rollouts
where one runner execution owns multiple role sessions. Each role gets its own
Gateway session and URL; the manager groups them for unified lifecycle control.
"""

from .multi_agent_framework import MultiAgentGatewayFramework
from .multi_agent_gateway_manager import MultiAgentGatewayManager
from .multi_agent_types import SessionGroupHandle

__all__ = [
    "MultiAgentGatewayManager",
    "MultiAgentGatewayFramework",
    "SessionGroupHandle",
]
