"""Shared helpers for gateway adapters and sessions."""

from __future__ import annotations

import json
from typing import Any


def normalize_tool_arguments(arguments: Any) -> dict[str, Any] | str:
    """Keep JSON-object arguments as dicts and all other values as strings."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
        return parsed if isinstance(parsed, dict) else arguments

    try:
        return json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(arguments)
