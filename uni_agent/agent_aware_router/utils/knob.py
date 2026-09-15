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

"""Knob value coercion helper.

Consumers that read debug-env overrides (strategy / collectors) get raw
strings; this helper coerces one to the knob's default type so type
validation stays next to the knob. Bad values raise ``ConfigError``.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from ..config.base import ConfigError

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")


def coerce_knob_value(name: str, raw: Any, default: Any) -> Any:
    """Coerce ``raw`` to ``default``'s type; raise ``ConfigError`` on mismatch.

    - bool: matched before int (bool subclasses int; ``bool("false")`` is True)
    - Enum: ``type(default)(raw)`` (validates the value is a known member)
    - dict: JSON object, keys normalized through the default's key enum when
      it has one (e.g. ``layer_weights``)
    - float / int: direct cast
    - otherwise: returned unchanged
    """
    if isinstance(default, bool):
        value = str(raw).strip().lower()
        if value in _TRUTHY:
            return True
        if value in _FALSY:
            return False
        raise ConfigError(f"router knob '{name}' expects a bool ({'/'.join(_TRUTHY + _FALSY)}), got {raw!r}")

    if isinstance(default, Enum):
        try:
            return type(default)(raw)
        except ValueError as exc:
            raise ConfigError(
                f"router knob '{name}' expects one of {[m.value for m in type(default)]}, got {raw!r}"
            ) from exc

    if isinstance(default, dict):
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ConfigError(f"router knob '{name}' expects a JSON object, got {raw!r}") from exc
        if not isinstance(parsed, dict):
            raise ConfigError(f"router knob '{name}' expects a JSON object, got {raw!r}")
        key_type = type(next(iter(default))) if default else str
        if isinstance(key_type, type) and issubclass(key_type, Enum):
            try:
                parsed = {key_type(k): v for k, v in parsed.items()}
            except ValueError as exc:
                raise ConfigError(
                    f"router knob '{name}' keys must be in {[m.value for m in key_type]}, got {set(parsed)}"
                ) from exc
        return parsed

    try:
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, int):
            return int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"router knob '{name}' expects a {type(default).__name__}, got {raw!r}") from exc
    return raw
