"""Long-running Lark chat agent.

Listens for inbound IM messages on Lark, dispatches each message to a
multi-step ``AgentInteraction`` loop running on a shared sandbox env,
and replies back to the user via ``lark-cli``. Per-chat message
transcripts are persisted so the agent can hold a real, ongoing
conversation across many turns and process restarts. Long-term user
profile / preferences live separately under the container's
``/workspace/memory/`` (written by the model itself, not by Python).

See ``app/lark_chat/README.md`` for setup and run instructions.
"""

from .listener import LarkEventListener, LarkEventListenerError, fetch_bot_open_id
from .transcript import TranscriptStore

__all__ = [
    "LarkEventListener",
    "LarkEventListenerError",
    "TranscriptStore",
    "fetch_bot_open_id",
]
