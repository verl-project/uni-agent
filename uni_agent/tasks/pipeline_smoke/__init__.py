"""Deterministic task used by the single-node training smoke test."""

from __future__ import annotations

from .reward import compute_reward, compute_score, normalize_answer
from .task import PipelineSmokeTask, PipelineSmokeTaskConfig

__all__ = [
    "PipelineSmokeTask",
    "PipelineSmokeTaskConfig",
    "compute_reward",
    "compute_score",
    "normalize_answer",
]
