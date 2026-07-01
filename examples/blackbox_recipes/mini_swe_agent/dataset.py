"""SWEBench-specific dataset that injects verl-standard reward fields."""

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
        tools_kwargs = extra_info.get("tools_kwargs", {})
        reward_config = tools_kwargs.get("reward", {})

        row_dict.setdefault("data_source", reward_config.get("name", "unknown"))
        row_dict.setdefault("reward_model", {"ground_truth": {}})

        return row_dict
