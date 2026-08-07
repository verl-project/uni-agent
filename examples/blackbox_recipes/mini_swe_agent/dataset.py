"""Dataset adapter for legacy OpenYuanRong and unified SWE task rows."""

from __future__ import annotations

from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset


def extract_image(env_config: dict) -> str:
    """Extract Docker image from env config, supporting both flat and nested formats.

    Flat:   env_config["image"]
    Nested: env_config["deployment"]["image"]
    """
    image = env_config.get("image")
    if image:
        return image
    deployment = env_config.get("deployment")
    if isinstance(deployment, dict):
        image = deployment.get("image")
        if image:
            return image
    return ""


class SWEBenchDataset(RLHFDataset):
    def __getitem__(self, item):
        return normalize_row(super().__getitem__(item))


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one dataset row to carry ``extra_info.tools_kwargs.task``."""
    row = dict(row)
    extra_info = dict(row.get("extra_info") or {})
    tools_kwargs = dict(extra_info.get("tools_kwargs") or {})
    task_config = tools_kwargs.get("task")

    if not isinstance(task_config, dict):
        env_config = tools_kwargs.get("env") or {}
        reward_config = tools_kwargs.get("reward") or {}
        image = extract_image(env_config)
        if not image:
            raise ValueError("SWE row has no sandbox image")
        task_config = {
            "name": reward_config.get("name", "swe_bench"),
            "sandbox": {
                "image": image,
                "sandbox_kwargs": {"post_setup_cmd": env_config.get("post_setup_cmd", "")},
            },
            "prompt": row.get("prompt") or [],
            "metadata": reward_config.get("metadata") or {},
        }
        tools_kwargs["task"] = task_config

    extra_info["tools_kwargs"] = tools_kwargs
    row["extra_info"] = extra_info
    row.setdefault("data_source", task_config.get("name", "unknown"))
    row.setdefault("reward_model", {"ground_truth": task_config.get("metadata") or {}})
    return row
