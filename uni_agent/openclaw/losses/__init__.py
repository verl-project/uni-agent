"""Custom OpenClaw RL/OPD losses registered into verl."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_REGISTERED = False
_IMPORT_ERROR: Exception | None = None

try:  # noqa: SIM105 - we want to remember the error
    from uni_agent.openclaw.losses import binary_grpo, combine, opd, topk_select  # noqa: F401

    _REGISTERED = True
except Exception as e:  # pragma: no cover - depends on verl/torch availability
    _IMPORT_ERROR = e
    logger.warning("OpenClaw losses not registered (verl/torch unavailable?): %s", e)


def ensure_registered() -> None:
    """Force-import the loss modules, raising if registration failed."""
    if _REGISTERED:
        return
    # Re-attempt so the real exception surfaces to the caller.
    from uni_agent.openclaw.losses import binary_grpo, combine, opd, topk_select  # noqa: F401


__all__ = ["ensure_registered"]
