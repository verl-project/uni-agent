"""JSON-backed per-chat conversation store.

One JSON file per ``chat_id`` at ``<base_dir>/<sanitized_chat_id>.json``
holding the OpenAI-shaped message history (with ``tool_calls`` on
assistant entries and matching ``tool_call_id`` on tool responses).
Writes are atomic (write tmp + ``os.replace``) so an interrupted
process never leaves a half-written file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_SAFE_CHARS = re.compile(r"[^a-zA-Z0-9_\-]")


def _safe_filename(chat_id: str) -> str:
    """Sanitize a Lark chat_id (``oc_<hex>``) into a filesystem-safe name."""
    name = _SAFE_CHARS.sub("_", chat_id).strip("_")
    return name or "default"


class ConversationStore:
    """Per-``chat_id`` persistent message store on local disk."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path(self, chat_id: str) -> Path:
        return self.base_dir / f"{_safe_filename(chat_id)}.json"

    def load(self, chat_id: str) -> list[dict]:
        """Return the persisted message list, or ``[]`` if none / unreadable."""
        p = self.path(chat_id)
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(data, list):
            return []
        return data

    def save(self, chat_id: str, messages: list[dict]) -> None:
        """Atomically replace the conversation file with ``messages``."""
        p = self.path(chat_id)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(p)
