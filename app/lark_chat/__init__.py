"""Long-running Lark chat agent.

Listens for inbound IM messages on Lark, dispatches each message to a
multi-step ``AgentInteraction`` loop running on a shared sandbox env,
and replies back to the user via ``lark-cli``. Conversation history is
persisted per-chat so the agent can hold a real, ongoing conversation
across many turns and process restarts.

See ``app/lark_chat/README.md`` for setup and run instructions.
"""

from .conversation import ConversationStore
from .listener import LarkEventListener, LarkEventListenerError, fetch_bot_open_id

__all__ = [
    "ConversationStore",
    "LarkEventListener",
    "LarkEventListenerError",
    "fetch_bot_open_id",
]
