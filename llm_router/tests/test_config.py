"""Config parsing from OmegaConf."""
from omegaconf import OmegaConf

from llm_router.config import parse_config


def test_default_policy_is_legacy():
    cfg = parse_config(OmegaConf.create({}))
    assert cfg.policy == "legacy_sticky"


def test_explicit_rule_based_policy():
    cfg = parse_config(OmegaConf.create({"policy": "rule_based"}))
    assert cfg.policy == "rule_based"


def test_routing_cache_size_default_is_10000():
    cfg = parse_config(OmegaConf.create({}))
    assert cfg.routing_cache_size == 10000


def test_unknown_policy_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown policy"):
        parse_config(OmegaConf.create({"policy": "magic"}))


def test_hit_threshold_default():
    cfg = parse_config(OmegaConf.create({}))
    assert cfg.hit_threshold == 1
    assert cfg.gpu_hit_threshold == 1
    assert cfg.cpu_hit_threshold == 1


def test_load_threshold_default():
    cfg = parse_config(OmegaConf.create({}))
    assert cfg.load_threshold == 1024


def test_max_prefix_entries_default():
    cfg = parse_config(OmegaConf.create({}))
    assert cfg.max_prefix_entries_per_server == 8192


def test_explicit_thresholds():
    cfg = parse_config(
        OmegaConf.create(
            {
                "policy": "rule_based",
                "hit_threshold": 256,
                "gpu_hit_threshold": 512,
                "cpu_hit_threshold": 128,
                "load_threshold": 32,
                "max_prefix_entries_per_server": 4096,
            }
        )
    )
    assert cfg.policy == "rule_based"
    assert cfg.hit_threshold == 256
    assert cfg.gpu_hit_threshold == 512
    assert cfg.cpu_hit_threshold == 128
    assert cfg.load_threshold == 32
    assert cfg.max_prefix_entries_per_server == 4096


def test_hit_threshold_backfills_tier_thresholds():
    cfg = parse_config(OmegaConf.create({"hit_threshold": 256}))
    assert cfg.gpu_hit_threshold == 256
    assert cfg.cpu_hit_threshold == 256


def test_negative_thresholds_clamped_to_zero():
    cfg = parse_config(
        OmegaConf.create(
            {
                "hit_threshold": -5,
                "gpu_hit_threshold": -8,
                "cpu_hit_threshold": -13,
                "load_threshold": -1,
            }
        )
    )
    assert cfg.hit_threshold == 0
    assert cfg.gpu_hit_threshold == 0
    assert cfg.cpu_hit_threshold == 0
    assert cfg.load_threshold == 0


def test_llm_router_manager_reads_actor_rollout_ref_level_config():
    from unittest.mock import MagicMock, patch

    from llm_router.manager import LLMRouter

    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "llm_router": {"policy": "rule_based", "hit_threshold": 7},
                "model": {},
                "rollout": {"name": "vllm"},
            }
        }
    )
    with (
        patch(
            "verl.experimental.agent_loop.agent_loop._get_rollout_and_model_config",
            return_value=(cfg.actor_rollout_ref.rollout, cfg.actor_rollout_ref.model),
        ),
        patch("verl.workers.rollout.replica.get_rollout_replica_class", return_value=MagicMock()),
    ):
        router = LLMRouter(config=cfg)

    assert router._llm_router_config.policy == "rule_based"
    assert router._llm_router_config.hit_threshold == 7
