"""Reward spec registry: register by name and load by config (mirrors tools/registry)."""

from collections.abc import Callable
from importlib import import_module
from typing import Any

from uni_agent.reward.base import AbstractRewardSpec

REWARD_SPEC_REGISTRY: dict[str, type[AbstractRewardSpec]] = {}

REWARD_SPEC_MODULES: dict[str, str] = {
    "search": "uni_agent.reward.search",
    "swe_bench": "uni_agent.reward.swe_bench",
    "swe_rebench": "uni_agent.reward.swe_rebench",
    "r2e_gym": "uni_agent.reward.r2e_gym",
    "terminal_bench": "uni_agent.reward.terminal_bench",
    "terminal_bench_v2": "uni_agent.reward.terminal_bench",
}


def register_reward_spec(name: str) -> Callable[[type[AbstractRewardSpec]], type[AbstractRewardSpec]]:
    """Decorator to register a reward spec class with a given name."""

    def decorator(cls: type[AbstractRewardSpec]) -> type[AbstractRewardSpec]:
        if name in REWARD_SPEC_REGISTRY and REWARD_SPEC_REGISTRY[name] != cls:
            raise ValueError(f"Reward spec {name} has already been registered: {REWARD_SPEC_REGISTRY[name]} vs {cls}")
        REWARD_SPEC_REGISTRY[name] = cls
        return cls

    return decorator


def _load_reward_spec_module(name: str) -> None:
    module_name = REWARD_SPEC_MODULES.get(name)
    if module_name is None:
        return
    try:
        import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import reward spec {name!r} from {module_name!r}. "
            "Please install the optional dependencies required by this reward spec."
        ) from exc


def load_reward_spec(config: dict[str, Any]) -> AbstractRewardSpec:
    """
    Load a reward spec instance by config.

    Config must contain "name" (registered name). Other keys are passed as kwargs
    to the reward spec class constructor.

    Example:
        config = {"name": "swe_bench", "metadata": {...}}
        spec = load_reward_spec(config)
    """
    if not config or "name" not in config:
        raise ValueError("Reward config must contain 'name'")
    name = config["name"]
    if name not in REWARD_SPEC_REGISTRY:
        _load_reward_spec_module(name)
    if name not in REWARD_SPEC_REGISTRY:
        available = sorted(set(REWARD_SPEC_REGISTRY) | set(REWARD_SPEC_MODULES))
        raise ValueError(f"Unknown reward spec: {name}. Available: {available}")
    kwargs = {k: v for k, v in config.items() if k != "name"}
    return REWARD_SPEC_REGISTRY[name](**kwargs)
