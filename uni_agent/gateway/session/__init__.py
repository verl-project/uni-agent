"""Session domain for the gateway: per-session state and model codec.

The gateway is a thin HTTP layer; this package holds the session-side logic it
serves: trajectory buffering and message encoding/decoding.
``SessionHandle`` / ``Trajectory`` are consumed by framework runners, while
``InternalGenerationRequest`` is the adapter-to-session request boundary.
"""

# Bind pure-data types before importing .codec (which imports verl and may trigger
# VERL_USE_EXTERNAL_MODULES hooks that import uni_agent.framework -> Trajectory).
from .types import AgentContext, InternalGenerationRequest, SessionHandle, Trajectory  # noqa: I001
from .codec import MessageCodec
from .session import GatewaySession, TrajectoryBuffer

__all__ = [
    "AgentContext",
    "InternalGenerationRequest",
    "GatewaySession",
    "MessageCodec",
    "SessionHandle",
    "Trajectory",
    "TrajectoryBuffer",
]
