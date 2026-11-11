"""Dataset adapter for multi-policy external-MAS training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from examples.multi_policy.external_mas.reward import extract_expected_answers
from verl.utils.dataset.rl_dataset import RLHFDataset


def attach_task_metadata(
    row: dict[str, Any],
    *,
    task_name: str = "multi_policy_external_mas",
) -> dict[str, Any]:
    """Copy the sample ground truth into the serialized Task Config.

    ``ExternalMASTask`` runs at the recipe level and therefore needs the
    sample's ground truth in its own config. The top-level ``ground_truth`` is
    the primary source; the metadata copy remains as a fallback and debugging
    aid. ``ground_truth_raw`` is preferred when present because
    ``reward_model.ground_truth`` may encode multiple answers as one
    concatenated string.
    """
    reward_model = row.get("reward_model")
    tools_kwargs = dict(row.get("tools_kwargs") or {})
    task = dict(tools_kwargs.get("task") or {})
    task.setdefault("name", task_name)
    metadata = dict(task.get("metadata") or {})
    if reward_model is not None:
        metadata.setdefault("reward_model", reward_model)
    extra_info = row.get("extra_info")
    if isinstance(extra_info, Mapping):
        ground_truth_raw = extra_info.get("ground_truth_raw")
        if ground_truth_raw is not None:
            metadata.setdefault("ground_truth_raw", ground_truth_raw)
    task["metadata"] = metadata
    if task.get("ground_truth") is None:
        task["ground_truth"] = extract_expected_answers(metadata)
    tools_kwargs["task"] = task
    row["tools_kwargs"] = tools_kwargs
    return row


class MultiPolicyDataset(RLHFDataset):
    """Forward each sample's ground truth into the external-MAS Task config."""

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = super().__getitem__(item)
        return attach_task_metadata(row)
