"""Process-global logging setup: the console sink, the global level, and
third-party noise suppression. Scope counterpart of ``session.py`` (per-run)."""

from __future__ import annotations

import logging

from .context import _QUIET_LOGGERS, _resolve_level
from .handlers import _install_console_sink


def setup_console_logging(level: int | str | None = None) -> None:
    """Configure process-global console logging.

    Installs the aligned+redacting console sink on the root logger, sets the global
    level (explicit ``level`` wins, then ``UNI_AGENT_LOG_LEVEL``, then INFO), and
    quiets chatty third-party loggers. Idempotent: calling it again only re-applies
    the level, so embedders (framework/verl) keep their handlers.

    This is the entry point for components without a per-run LogContext — the
    router being one — that only want stdout logging. Session-scoped file logging
    lives in :mod:`uni_agent.logging.session`.
    """
    resolved = _resolve_level(level)
    root = logging.getLogger()
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    root.setLevel(resolved)
    _install_console_sink(resolved)
