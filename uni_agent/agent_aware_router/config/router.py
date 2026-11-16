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

"""Top-level KVCAwareConfig and parsing logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Any

from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .base import (
    ConfigError,
    StrategyConfig,
    _multiline_repr,
)
from .collector import CollectorConfig

logger = logging.getLogger(__name__)

# ============================================================
# Top-level KVCAwareConfig
# ============================================================

_DEFAULT_COLLECTOR = CollectorConfig()


@dataclass(repr=False)
class KVCAwareConfig:
    """Top-level config for KVCAwareBalancer, parsed from OmegaConf DictConfig.

    VeRL composes the router config under the rollout ``strategy`` node
    (via the ``router@strategy`` hydra defaults entry — see
    ``verl/trainer/config/rollout/router/kvcaware.yaml``) and passes that node
    to the Balancer constructor.  The Balancer calls ``from_config(cfg)`` to
    obtain this fully-resolved dataclass instance.

    Attributes:
        strategy: Polymorphic strategy config (with ``_target_``).
        collector: Collector module connection-type tuning config.
    """

    strategy: StrategyConfig  # required, no default
    collector: CollectorConfig = field(default_factory=lambda: _DEFAULT_COLLECTOR)

    @classmethod
    def from_config(cls, cfg: DictConfig | dict) -> KVCAwareConfig:
        """Two-step parsing of VeRL-transmitted config.

        Step 1: OmegaConf.merge for auto-recursive dataclass fields
                (collector).

        Step 2: Manual traversal of strategy (dict from Hydra defaults
                composition) — instantiate its ``_target_`` entry.
        """
        if not isinstance(cfg, DictConfig | dict):
            raise ConfigError(f"cfg must be DictConfig or dict, got {type(cfg)}")

        cfg = OmegaConf.create(cfg)

        # ── Extract polymorphic sections before merge ──────────────
        # The strategy is polymorphic (_target_); pull it out before the
        # dataclass-typed merge below. Top-level `_target_` is defensively popped
        # for structured-input compatibility.
        strategy_raw = _extract_strategy(cfg)

        # ── Step 1: merge dataclass-typed fields (collector) ──
        defaults = OmegaConf.create(
            {
                "collector": OmegaConf.structured(CollectorConfig),
            }
        )
        kwargs_for_merge = OmegaConf.create(cfg)
        # Remove polymorphic sections to avoid ReadonlyConfigError
        for key in ("strategy", "_target_"):
            if key in kwargs_for_merge:
                OmegaConf.set_struct(kwargs_for_merge, False)
                kwargs_for_merge.pop(key, None)
                OmegaConf.set_struct(kwargs_for_merge, True)

        # Validate non-dict types for collector
        if (
            "collector" in kwargs_for_merge
            and kwargs_for_merge.collector is not None
            and not isinstance(kwargs_for_merge.collector, dict | DictConfig)
        ):
            raise ConfigError(f"collector must be a dict, got {type(kwargs_for_merge.collector).__name__}")

        merged = OmegaConf.merge(defaults, kwargs_for_merge)
        config_obj = OmegaConf.to_object(merged)

        # Extract resolved dataclass fields
        if isinstance(config_obj, dict):
            collector_cfg = config_obj.get("collector") or CollectorConfig()
        else:
            collector_cfg = getattr(config_obj, "collector", None) or CollectorConfig()

        # ── Step 2: parse strategy (polymorphic) ────────────────────
        if strategy_raw is None:
            raise ConfigError("strategy is required — must be explicitly configured")
        strategy = _parse_polymorphic(strategy_raw, StrategyConfig, "strategy")

        # ── Validate and construct ─────────────────────────────────
        result = cls(
            strategy=strategy,
            collector=collector_cfg,
        )
        result.validate()
        return result

    def apply_override(self, override: dict[str, Any] | None) -> None:
        """Apply a flat runtime override dict onto the declared config fields.

        Keys are matched against the dataclass-declared fields of both config
        sections (strategy and collector). A matched section is rebuilt from
        its current declared values plus the override, re-running each field's
        validation (``__post_init__``) — so an invalid override raises
        ``ConfigError`` instead of silently poisoning the config. Keys that
        match no declared field are ignored with a warning; ``None`` values are
        treated as "not set" and skipped, letting callers pass a partially
        populated override node unconditionally.

        Args:
            override: Flat mapping of field name → value (e.g. the
                ``custom.agent_framework.router`` node of the rollout config).
        """
        if not override:
            return
        declared_fields: set[str] = set()
        for section in (self.strategy, self.collector):
            declared_fields.update(f.name for f in fields(section) if f.init)
        unknown = sorted(set(override) - declared_fields)

        for section_name, section in (("strategy", self.strategy), ("collector", self.collector)):
            declared = {f.name for f in fields(section) if f.init}
            matched = {k: v for k, v in override.items() if k in declared and v is not None}
            if not matched:
                continue
            kwargs = {f.name: getattr(section, f.name) for f in fields(section) if f.init}
            kwargs.update(matched)
            setattr(self, section_name, type(section)(**kwargs))

        if unknown:
            logger.warning(f"apply_override: ignoring unknown override keys: {unknown}")

    def validate(self) -> None:
        """Validate the full config. Raises ConfigError with all violations."""
        if not isinstance(self.strategy, StrategyConfig):
            raise ConfigError(f"strategy must be a StrategyConfig, got {type(self.strategy).__name__}")

    def __repr__(self) -> str:
        """Multi-line indented repr (delegates to the shared _multiline_repr)."""
        return _multiline_repr(self)


# ============================================================
# Helper functions
# ============================================================


def _extract_strategy(cfg: DictConfig) -> Any | None:
    """Extract the strategy node from cfg (None when absent or null)."""
    val = cfg.get("strategy")
    return val if val is not None else None


def _parse_polymorphic(
    item: Any,
    base_class: type,
    name: str,
) -> Any:
    """Parse a polymorphic entry with ``_target_`` for hydra.instantiate.

    Validates that the instantiated object is a subclass of ``base_class``.
    """
    if not isinstance(item, dict | DictConfig):
        raise ConfigError(f"{name} must be a dict, got {type(item)}")

    item_conf = OmegaConf.create(item) if isinstance(item, dict) else item

    if "_target_" not in item_conf:
        raise ConfigError(f"{name} must have '_target_' key, got keys: {list(item_conf.keys())}")

    try:
        parsed = instantiate(item_conf)
    except (InstantiationException, ImportError, AttributeError) as e:
        raise ConfigError(f"{name} failed to instantiate _target_ '{item_conf._target_}': {e}") from e

    if not isinstance(parsed, base_class):
        raise ConfigError(f"{name} _target_ must inherit {base_class.__name__}, got {type(parsed).__name__}")

    return parsed
