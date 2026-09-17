# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for agent_aware_router config dataclasses and parsing.

Sections:
  • KVCAwareStrategyConfig — normal construction, validation, Hydra instantiation
  • KVCAwareConfig.from_config — normal inputs, abnormal inputs, other behaviour
  • E2E — packaged YAML composition
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from uni_agent.agent_aware_router.config import (
    CollectorConfig,
    ConfigError,
    KVCAwareConfig,
    KVCAwareStrategyConfig,
    StrategyConfig,
)

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


# ============================================================
# KVCAwareStrategyConfig
# ============================================================

# -- (1) Normal construction --


def test_strategy_default_config():
    """
    Feature: KVCAwareStrategyConfig has a single persisted field, load_threshold
    Description: construct strategy config with no kwargs
    Expectation: cfg.load_threshold == 0.9 and no other tuning fields exist
    """
    cfg = KVCAwareStrategyConfig()
    assert cfg.load_threshold == 0.9


def test_strategy_normal_fields_parse():
    """
    Feature: load_threshold assigns an explicit value
    Description: construct KVCAwareStrategyConfig with load_threshold kwarg
    Expectation: cfg.load_threshold equals the explicit value
    """
    cfg = KVCAwareStrategyConfig(load_threshold=0.5)
    assert cfg.load_threshold == 0.5


# -- (2) Abnormal construction --


@pytest.mark.parametrize("load_threshold", [0, -1, 1.0, 80])
def test_strategy_load_threshold_out_of_range_raises_config_error(load_threshold):
    """
    Feature: load_threshold outside (0, 1) triggers validation error
    Description: construct strategy config with load_threshold = 0 / -1 / 1.0 / 80
    Expectation: raises ConfigError matching "load_threshold"
    """
    with pytest.raises(ConfigError, match="load_threshold"):
        KVCAwareStrategyConfig(load_threshold=load_threshold)


# -- (3) Hydra instantiation --


def test_strategy_instantiates_with_explicit_fields_and_defaults():
    """
    Feature: Hydra instantiate produces a fully-populated KVCAwareStrategyConfig
    Description: instantiate a strategy entry with only _target_
    Expectation: result is a KVCAwareStrategyConfig/StrategyConfig, defaults filled
    """
    from hydra.utils import instantiate

    entry = OmegaConf.create(
        {
            "_target_": "uni_agent.agent_aware_router.config.strategy.KVCAwareStrategyConfig",
        }
    )
    result = instantiate(entry)

    # concrete type and StrategyConfig base
    assert isinstance(result, KVCAwareStrategyConfig)
    assert isinstance(result, StrategyConfig)
    # defaults filled in
    assert result.load_threshold == 0.9


# ============================================================
# KVCAwareConfig.from_config — normal input
# ============================================================


@pytest.mark.parametrize("wrap", [lambda d: OmegaConf.create(d), lambda d: d], ids=["omegaconf", "plain_dict"])
def test_from_config_assembles_sections(wrap):
    """
    Feature: from_config assembles the sections into a KVCAwareConfig
    Description: from_config with a full strategy/collector config (OmegaConf or plain dict)
    Expectation: result is a KVCAwareConfig with correctly typed sections
    """
    full_config = {
        "strategy": {
            "_target_": "uni_agent.agent_aware_router.config.strategy.KVCAwareStrategyConfig",
            "load_threshold": 0.7,
        },
    }
    result = KVCAwareConfig.from_config(full_config)
    assert isinstance(result, KVCAwareConfig)
    assert isinstance(result.strategy, KVCAwareStrategyConfig)
    assert result.strategy.load_threshold == 0.7
    assert isinstance(result.collector, CollectorConfig)


def test_from_config_omitted_sections_take_defaults():
    """
    Feature: omitted collector takes its Config default
    Description: from_config with a config containing only strategy
    Expectation: result.collector is a CollectorConfig instance
    """
    kwargs = OmegaConf.create(
        {
            "strategy": {
                "_target_": "uni_agent.agent_aware_router.config.strategy.KVCAwareStrategyConfig",
            },
        }
    )
    result = KVCAwareConfig.from_config(kwargs)
    assert isinstance(result.collector, CollectorConfig)


# ============================================================
# KVCAwareConfig.from_config — abnormal input
# ============================================================


@pytest.mark.parametrize(
    "strategy",
    [
        "kvc_aware",  # not a dict
        42,  # not a dict
    ],
)
def test_strategy_invalid_raises_config_error(strategy):
    """
    Feature: invalid strategy node triggers from_config validation error
    Description: from_config with non-dict strategy values
    Expectation: raises ConfigError
    """
    kwargs = OmegaConf.create({"strategy": strategy})
    with pytest.raises(ConfigError):
        KVCAwareConfig.from_config(kwargs)


@pytest.mark.parametrize(
    "strategy,match",
    [
        # missing _target_ on the strategy node
        ({"strategy": {}}, "_target_"),
        # _target_ pointing to a non-StrategyConfig subclass
        (
            {
                "strategy": {
                    "_target_": "uni_agent.agent_aware_router.config.CollectorConfig",
                }
            },
            "StrategyConfig",
        ),
        # empty config — no strategy key at all
        ({}, "strategy"),
        # strategy explicitly null
        ({"strategy": None}, "strategy"),
    ],
)
def test_from_config_rejects_invalid_strategy(strategy, match):
    """
    Feature: from_config rejects invalid strategy configuration
    Description: missing _target_ / non-StrategyConfig _target_ / empty config / null strategy
    Expectation: raises ConfigError matching the relevant keyword
    """
    kwargs = OmegaConf.create(strategy)
    with pytest.raises(ConfigError, match=match):
        KVCAwareConfig.from_config(kwargs)


@pytest.mark.parametrize("bad_cfg", ["invalid", None, 42])
def test_from_config_rejects_non_dict_input(bad_cfg):
    """
    Feature: from_config rejects inputs that are neither DictConfig nor dict
    Description: from_config with str / None / int input
    Expectation: raises ConfigError matching "cfg"
    """
    with pytest.raises(ConfigError, match="cfg"):
        KVCAwareConfig.from_config(bad_cfg)


def test_from_config_aggregates_multiple_errors():
    """
    Feature: from_config raises on errors from any section
    Description: from_config with a missing strategy and invalid collector
    Expectation: raises ConfigError whose message contains a relevant field name
    """
    kwargs = OmegaConf.create(
        {
            "collector": {"http_interval": -1},
        }
    )
    with pytest.raises(ConfigError) as exc_info:
        KVCAwareConfig.from_config(kwargs)
    error_msg = str(exc_info.value)
    assert "strategy" in error_msg or "http_interval" in error_msg


# ============================================================
# KVCAwareConfig other behaviour
# ============================================================


def test_top_level_unknown_keys_ignored():
    """
    Feature: from_config ignores top-level keys outside the config domain
    Description: from_config with a config carrying extra non-domain keys
    Expectation: result is KVCAwareConfig and the extra keys are dropped
    """
    kwargs = OmegaConf.create(
        {
            "_unknown_top_level": "ignored",
            "router_strategy": "should_be_dropped",
            "strategy": {
                "_target_": "uni_agent.agent_aware_router.config.strategy.KVCAwareStrategyConfig",
            },
        }
    )
    result = KVCAwareConfig.from_config(kwargs)
    assert isinstance(result, KVCAwareConfig)
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(result)}
    assert "_unknown_top_level" not in field_names
    assert "router_strategy" not in field_names


def test_compact_repr():
    """
    Feature: KVCAwareConfig renders a multi-line indented repr
    Description: from_config then call repr and inspect output
    Expectation: repr contains newline, starts with KVCAwareConfig(, and includes fields
    """
    kwargs = OmegaConf.create(
        {
            "strategy": {
                "_target_": "uni_agent.agent_aware_router.config.strategy.KVCAwareStrategyConfig",
            },
            "collector": {"http_interval": 5},
        }
    )
    result = KVCAwareConfig.from_config(kwargs)
    r = repr(result)
    assert "\n" in r
    assert r.startswith("KVCAwareConfig(")


# ============================================================
# KVCAwareConfig.apply_override
# ============================================================


def _default_config() -> KVCAwareConfig:
    """Build a KVCAwareConfig from only a strategy node (defaults elsewhere)."""
    return KVCAwareConfig.from_config(
        OmegaConf.create(
            {
                "strategy": {
                    "_target_": "uni_agent.agent_aware_router.config.strategy.KVCAwareStrategyConfig",
                },
            }
        )
    )


def test_apply_override_updates_both_sections():
    """
    Feature: a flat override dict lands on the matching declared fields of both sections
    Description: apply_override({"load_threshold": 0.6, "http_interval": 7})
    Expectation: strategy.load_threshold == 0.6 and collector.http_interval == 7
    """
    cfg = _default_config()
    cfg.apply_override({"load_threshold": 0.6, "http_interval": 7})
    assert cfg.strategy.load_threshold == 0.6
    assert cfg.collector.http_interval == 7


@pytest.mark.parametrize(
    "override,match",
    [
        ({"load_threshold": 1.5}, "load_threshold"),
        ({"http_interval": -1}, "http_interval"),
    ],
)
def test_apply_override_invalid_value_raises(override, match):
    """
    Feature: an out-of-range override is rejected by the rebuilt section's validation
    Description: apply_override with load_threshold=1.5 / http_interval=-1
    Expectation: raises ConfigError mentioning the offending field
    """
    cfg = _default_config()
    with pytest.raises(ConfigError, match=match):
        cfg.apply_override(override)


# ============================================================
# E2E integration
# ============================================================


def _load_pkg_router_yaml() -> dict:
    """Compose ``uni_agent/agent_aware_router/configs/agent_aware_router.yaml`` via Hydra.

    Expands the YAML defaults block into one config dictionary.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    import uni_agent.agent_aware_router.configs as _cfg_pkg

    config_dir = str(next(iter(_cfg_pkg.__path__)))
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="agent_aware_router")
    return OmegaConf.to_container(cfg, resolve=True)


def test_packaged_yaml_e2e():
    """
    Feature: Hydra composition of the packaged router YAML parses end-to-end
    Description: compose configs/router.yaml, then from_config
    Expectation: all fields match the YAML config and non-domain keys are dropped
    """
    loaded = _load_pkg_router_yaml()

    result = KVCAwareConfig.from_config(loaded)

    # ── strategy ──
    strategy = result.strategy
    assert isinstance(strategy, KVCAwareStrategyConfig)
    assert strategy.load_threshold == 0.9
    # FQN top-level keys are harmless to from_config (extra keys are dropped)
    assert loaded["router_class"] == "uni_agent.agent_aware_router.balancer.KVCAwareBalancer"
    assert not hasattr(result, "router_class")
