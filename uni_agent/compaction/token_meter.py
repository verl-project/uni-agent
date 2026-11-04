"""Token estimation for conversation compaction.

Follows the deepseek-harness ``TokenMeter`` design: a fixed-density
heuristic estimator (no real tokenizer dependency) so that the same
content is always priced with the same number, across models.

When the provider reports real usage (``prompt_tokens`` /
``completion_tokens`` from the chat-completions response), callers should
prefer those numbers; this module provides the fallback estimate and the
per-message pricing used to decide *what* to compact and to verify the
summary is actually smaller than the shadowed content.
"""

from __future__ import annotations

from typing import Any

# Fixed-density heuristic constants (deepseek-harness token-meter).
CHARS_PER_TOKEN = 4
BLOCK_OVERHEAD = 4  # per content block JSON structure overhead
ROLE_OVERHEAD = 4  # per message role framing overhead


def estimate_text(text: str) -> int:
    """Heuristic token count for a plain text string."""
    if not text:
        return 0
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN + BLOCK_OVERHEAD


def _estimate_tool_calls(tool_calls: list[dict[str, Any]]) -> int:
    total = 0
    for tc in tool_calls:
        function = tc.get("function") or {}
        name = str(function.get("name") or "")
        arguments = str(function.get("arguments") or "")
        total += (
            (len(name) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN
            + (len(arguments) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN
            + BLOCK_OVERHEAD
        )
    return total


def estimate_message(message: dict[str, Any]) -> int:
    """Heuristic token count for one chat message (any role)."""
    content = message.get("content")
    if content is None:
        content_tokens = 0
    elif isinstance(content, str):
        content_tokens = estimate_text(content)
    elif isinstance(content, list):
        content_tokens = sum(
            estimate_text(block.get("text") or "") if isinstance(block, dict) else estimate_text(str(block))
            for block in content
        )
    else:
        content_tokens = estimate_text(str(content))

    tool_call_tokens = _estimate_tool_calls(message.get("tool_calls") or [])
    return content_tokens + tool_call_tokens + ROLE_OVERHEAD


class TokenMeter:
    """Stateless token estimator for a list of chat messages.

    ``estimate_messages`` prices the full conversation. ``current_tokens``
    prefers the provider-reported usage when available (more accurate) and
    falls back to the heuristic estimate.
    """

    def estimate_messages(self, messages: list[dict[str, Any]]) -> int:
        return sum(estimate_message(m) for m in messages)

    def current_tokens(
        self,
        messages: list[dict[str, Any]],
        last_usage: tuple[int | None, int | None] | None = None,
    ) -> int:
        """Best-available token count for the current conversation.

        ``last_usage`` is the ``(prompt_tokens, completion_tokens)`` pair
        reported by the provider for the most recent model call. When
        present and non-zero, ``prompt_tokens`` already prices the whole
        context as seen by the server, so ``prompt + completion`` is the
        projected pressure for the *next* request.
        """
        prompt_tokens, completion_tokens = last_usage or (None, None)
        if prompt_tokens:
            return int(prompt_tokens) + int(completion_tokens or 0)
        return self.estimate_messages(messages)
