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

"""Checked sleep lifetime and opt-in native utility installation."""

import sys
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from uni_agent.vllm_plugins.lifecycle import CHECKED_SLEEP_METHOD, RESET_BOUNDARY_METHOD, _checked_sleep, register

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_checked_sleep_keeps_the_reset_guard_until_native_pause_finishes():
    future = Future()
    reset = lambda *args, **kwargs: False
    core = SimpleNamespace(reset_prefix_cache=reset, sleep=lambda **kwargs: future)
    assert _checked_sleep(core, 2) is future
    with pytest.raises(RuntimeError, match="confirm"):
        core.reset_prefix_cache(reset_connector=True)
    with pytest.raises(RuntimeError, match="already in progress"):
        _checked_sleep(core)
    future.set_exception(RuntimeError("native pause failed"))
    assert core.reset_prefix_cache is reset
    assert not core._uni_agent_checked_sleep_active
    assert core.reset_prefix_cache() is False


def test_checked_sleep_restores_an_inherited_reset_method_after_failure():
    class Core:
        def reset_prefix_cache(self, **kwargs):
            return False

        def sleep(self, **kwargs):
            self.reset_prefix_cache(reset_connector=True)

    core = Core()
    with pytest.raises(RuntimeError, match="confirm"):
        _checked_sleep(core)
    assert "reset_prefix_cache" not in core.__dict__
    assert core.reset_prefix_cache() is False


def test_lifecycle_utilities_require_their_own_opt_in_and_install_atomically(monkeypatch):
    class Core:
        pass

    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", SimpleNamespace(EngineCore=Core))
    monkeypatch.delenv("UNI_AGENT_ENABLE_CACHE_LIFECYCLE", raising=False)
    register()
    assert not hasattr(Core, CHECKED_SLEEP_METHOD)
    monkeypatch.setenv("UNI_AGENT_ENABLE_CACHE_LIFECYCLE", "1")
    conflict = object()
    setattr(Core, RESET_BOUNDARY_METHOD, conflict)
    with pytest.raises(RuntimeError, match="Refusing"):
        register()
    assert not hasattr(Core, CHECKED_SLEEP_METHOD)
    delattr(Core, RESET_BOUNDARY_METHOD)
    register()
    register()
    assert callable(getattr(Core, CHECKED_SLEEP_METHOD))
    assert callable(getattr(Core, RESET_BOUNDARY_METHOD))
