"""Session domain for the gateway: per-session state and model codec.

The gateway is a thin HTTP layer; this package holds the session-side logic it
serves: trajectory buffering and message encoding/decoding.
``SessionHandle`` / ``Trajectory`` are consumed by framework runners, while
``InternalGenerationRequest`` is the adapter-to-session request boundary.
"""

from .codec import MessageCodec
from .session import GatewaySession, TrajectoryBuffer
from .types import InternalGenerationRequest, SessionHandle, Trajectory

__all__ = [
    "InternalGenerationRequest",
    "GatewaySession",
    "MessageCodec",
    "SessionHandle",
    "Trajectory",
    "TrajectoryBuffer",
]
