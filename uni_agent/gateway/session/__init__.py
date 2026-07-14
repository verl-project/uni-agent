"""Session domain for the gateway: per-session state and model codec.

The gateway is a thin HTTP layer; this package holds the session-side logic it
serves: trajectory buffering and message encoding/decoding.
``SessionHandle`` / ``Trajectory`` are consumed by framework runners, while
``InternalGenerationRequest`` (in :mod:`.types`) is the adapter-to-session boundary.
"""

from .codec import MessageCodec
from .session import GatewaySession, TrajectoryBuffer
from .types import SessionHandle, Trajectory

__all__ = [
    "GatewaySession",
    "MessageCodec",
    "SessionHandle",
    "Trajectory",
    "TrajectoryBuffer",
]
