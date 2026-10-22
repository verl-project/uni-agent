"""Extract harness identity without introducing backend-specific routing policy."""

from collections.abc import Mapping

from uni_agent.gateway.adapters.types import MalformedRequestError
from uni_agent.gateway.session.types import AgentContext


def agent_context_from_headers(headers: Mapping[str, str]) -> AgentContext | None:
    """Read Claude Code identity from a case-insensitive HTTP headers mapping.

    Missing parent identity stays unknown. Header availability depends on the
    harness version; requests without identity retain ordinary Gateway behavior.
    """

    def value(name: str) -> str | None:
        raw = headers.get(name)
        if raw is None:
            return None
        if len(raw) > 512 or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in raw):
            raise MalformedRequestError(f"Invalid {name} header")
        return raw.strip() or None

    session_id = value("x-claude-code-session-id")
    agent_id = value("x-claude-code-agent-id")
    parent_agent_id = value("x-claude-code-parent-agent-id")
    if session_id is None:
        if agent_id is not None or parent_agent_id is not None:
            raise MalformedRequestError("Claude agent identity requires x-claude-code-session-id")
        return None
    if parent_agent_id is not None and (agent_id is None or parent_agent_id == agent_id):
        raise MalformedRequestError("Claude parent identity requires a distinct child agent ID")
    return AgentContext(
        provider="claude_code",
        session_id=session_id,
        agent_id=agent_id,
        parent_agent_id=parent_agent_id,
    )
