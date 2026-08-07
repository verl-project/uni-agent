"""SWE-bench-like dataset that injects verl-standard reward fields.

Self-contained for the claude-code recipe; mirrors the mini-swe-agent dataset
so claude_code/ does not depend on mini_swe_agent/.
"""

from collections.abc import Mapping

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
        row_dict = super().__getitem__(item)
        extra_info = row_dict.get("extra_info", {})
        if not isinstance(extra_info, Mapping):
            raise TypeError(f"extra_info must be a mapping, got {type(extra_info).__name__}")

        tools_kwargs = extra_info.get("tools_kwargs", {})
        if not isinstance(tools_kwargs, Mapping):
            raise TypeError(f"extra_info.tools_kwargs must be a mapping, got {type(tools_kwargs).__name__}")

        if "task" in tools_kwargs:
            task_config = tools_kwargs["task"]
            if not isinstance(task_config, Mapping):
                raise TypeError(f"tools_kwargs.task must be a mapping, got {type(task_config).__name__}")
            evaluator_name = task_config.get("name")
            reward_metadata = task_config.get("metadata", {})
            metadata_path = "tools_kwargs.task.metadata"
        else:
            reward_config = tools_kwargs.get("reward", {})
            if not isinstance(reward_config, Mapping):
                raise TypeError(f"tools_kwargs.reward must be a mapping, got {type(reward_config).__name__}")
            evaluator_name = reward_config.get("name")
            reward_metadata = reward_config.get("metadata", {})
            metadata_path = "tools_kwargs.reward.metadata"

        if not isinstance(reward_metadata, Mapping):
            raise TypeError(f"{metadata_path} must be a mapping, got {type(reward_metadata).__name__}")

        row_dict["data_source"] = row_dict.get("data_source") or evaluator_name or "unknown"
        row_dict.setdefault("reward_model", {"ground_truth": reward_metadata})

        return row_dict
