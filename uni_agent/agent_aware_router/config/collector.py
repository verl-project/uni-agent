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

"""Collector config: the only user-tunable collector knob (``http_interval``).

Individual collectors are referenced by name (bound on the runtime strategy —
see ``KVCacheAwareStrategy.COLLECTOR_NAMES``); all other connection-type
tuning parameters live as built-in defaults on the transports themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import ConfigError, _multiline_repr


@dataclass(repr=False)
class CollectorConfig:
    """Config for the collectors module.

    Attributes:
        http_interval: HTTP request polling interval in seconds (HTTP polling transport).
    """

    http_interval: float = 5.0

    def __post_init__(self) -> None:
        if self.http_interval <= 0:
            raise ConfigError(f"http_interval must be > 0, got {self.http_interval}")

    __repr__ = _multiline_repr
