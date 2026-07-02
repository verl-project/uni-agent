"""Session domain for the gateway: per-session state, codec, and wire protocol.

The gateway is a thin HTTP layer; this package holds the session-side logic it
serves — trajectory buffering, message encoding/decoding, and the chat-completion
protocol types. ``SessionHandle`` and ``Trajectory`` are the cross-package public
surface (consumed by the framework runners).
"""

from .codec import MalformedRequestError, MessageCodec
from .protocol import ChatCompletionRequest, ChatCompletionResponse
from .session import GatewaySession
from .types import SessionHandle, Trajectory, TrajectoryBuffer

__all__ = [
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "GatewaySession",
    "MalformedRequestError",
    "MessageCodec",
    "SessionHandle",
    "Trajectory",
    "TrajectoryBuffer",
]
