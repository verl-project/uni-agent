"""Config schema for llm_router."""
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

VALID_POLICIES = {"legacy_sticky", "rule_based"}


@dataclass
class LLMRouterConfig:
    policy: str = "legacy_sticky"
    routing_cache_size: int = 10000
    # RuleBasedPolicy knobs — ignored by LegacyStickyPolicy.
    hit_threshold: int = 1
    gpu_hit_threshold: int = 1
    cpu_hit_threshold: int = 1
    load_threshold: int = 1024
    max_prefix_entries_per_server: int = 8192


def parse_config(cfg: DictConfig | dict[str, Any]) -> LLMRouterConfig:
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True) or {}
    policy = cfg.get("policy", "legacy_sticky")
    if policy not in VALID_POLICIES:
        raise ValueError(f"unknown policy: {policy!r} (valid: {sorted(VALID_POLICIES)})")
    hit_threshold = max(0, int(cfg.get("hit_threshold", 1)))
    return LLMRouterConfig(
        policy=policy,
        routing_cache_size=int(cfg.get("routing_cache_size", 10000)),
        hit_threshold=hit_threshold,
        gpu_hit_threshold=max(0, int(cfg.get("gpu_hit_threshold", hit_threshold))),
        cpu_hit_threshold=max(0, int(cfg.get("cpu_hit_threshold", hit_threshold))),
        load_threshold=max(0, int(cfg.get("load_threshold", 1024))),
        max_prefix_entries_per_server=int(cfg.get("max_prefix_entries_per_server", 8192)),
    )
