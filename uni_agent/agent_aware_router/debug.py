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

"""Debug-mode facility for the router package.

Defines the ``UNI_AGENT_ROUTER_*`` env namespace and provides only its
detection and per-variable query primitives — no knob catalogs, no value
coercion, no assembly. Those belong to the consumers:

  • the inference driver (``examples/agent_aware_router/run_infer.py``) scans
    the namespace for strategy-knob overrides and forwards them on
    ``rollout.custom.agent_framework.router``;
  • the balancer (``balancer.py``) validates knob names against
    ``_DEFAULT_STRATEGY_KNOBS`` and coerces values to the knob types;
  • future submodules (collector granularity, rl-insight toggles) call
    ``is_debug_enabled()`` / ``get_debug_var()`` the same way.

The namespace is inert unless the master switch is on.

Hot-path consumers should cache the result at construction
(``self._debug = is_debug_enabled()``); this module stays stateless.
"""

from __future__ import annotations

import os
from typing import Mapping

DEBUG_ENV = "UNI_AGENT_ROUTER_DEBUG"
ENV_PREFIX = "UNI_AGENT_ROUTER_"

_TRUTHY = ("1", "true", "yes", "on")


def is_debug_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """True when the debug master switch is on (``UNI_AGENT_ROUTER_DEBUG``)."""
    env = os.environ if environ is None else environ
    return env.get(DEBUG_ENV, "").strip().lower() in _TRUTHY


def get_debug_var(name: str, environ: Mapping[str, str] | None = None) -> str | None:
    """Query one ``UNI_AGENT_ROUTER_<NAME>`` variable's raw string value.

    Returns the value only when debug mode is enabled and the variable is set
    to a non-empty string; ``None`` otherwise (switch off, unset, empty).
    ``name`` is case-insensitive (``slow_cut`` ≡ ``SLOW_CUT``).
    """
    if not is_debug_enabled(environ):
        return None
    env = os.environ if environ is None else environ
    return env.get(ENV_PREFIX + name.strip().upper()) or None
