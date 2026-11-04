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

"""Strategy-specific configs.

Concrete routing strategy configs. The matching runtime strategy classes
(e.g. ``KVCacheAwareStrategy``) live under ``uni_agent.agent_aware_router.strategies``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import ConfigError, StrategyConfig, _multiline_repr


@dataclass(repr=False)
class KVCAwareStrategyConfig(StrategyConfig):
    """Config for KVCache-Aware routing strategy.

    Persists the user-tunable knob (``load_threshold``); the remaining tuning
    knobs fall back to ``DEFAULT_STRATEGY_KNOBS`` inside
    ``KVCacheAwareStrategy.from_config``.
    """

    load_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.load_threshold < 1:
            raise ConfigError(f"load_threshold must be in (0, 1), got {self.load_threshold}")

    __repr__ = _multiline_repr
