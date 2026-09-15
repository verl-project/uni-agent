"""HotpotQA task and reward."""

from __future__ import annotations

from .reward import compute_score
from .task import HotpotQATask, HotpotQATaskConfig

__all__ = ["HotpotQATask", "HotpotQATaskConfig", "compute_score"]
