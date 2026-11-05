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

"""Opt-in checked cache operations in the native engine process."""

import os
import time
from concurrent.futures import Future

CHECKED_SLEEP_METHOD = "uni_agent_checked_sleep"
RESET_BOUNDARY_METHOD = "uni_agent_reset_cache_boundary"
RESET_BOUNDARY_KEY = "uni_agent_cache_reset_boundary"
_OWNER = "uni_agent.cache_lifecycle.v1"


def _checked_sleep(self, level=1, mode="abort"):
    """Preserve native sleep while propagating its implicit reset's result."""
    if getattr(self, "_uni_agent_checked_sleep_active", False):
        raise RuntimeError("a checked sleep is already in progress")
    self._uni_agent_checked_sleep_active = True
    previous = self.__dict__.get("reset_prefix_cache")
    original = self.reset_prefix_cache

    def checked_reset(*args, **kwargs):
        if original(*args, **kwargs) is not True:
            raise RuntimeError("engine did not confirm successful prefix cache reset")
        return True

    def restore(_future=None):
        if previous is None:
            self.__dict__.pop("reset_prefix_cache", None)
        else:
            self.reset_prefix_cache = previous
        self._uni_agent_checked_sleep_active = False

    self.reset_prefix_cache = checked_reset
    callbacks = getattr(self, "_idle_state_callbacks", None)
    previous_callbacks = {id(callback) for callback in callbacks} if callbacks is not None else set()
    try:
        result = self.sleep(level=level, mode=mode)
    except BaseException:
        restore()
        raise
    if isinstance(result, Future):
        # Native deferred pause callbacks do not propagate reset exceptions to
        # their Future. Guard only callbacks created by this sleep so an error
        # completes the RPC and restores the instance method before returning.
        if callbacks is not None:
            for i, callback in enumerate(callbacks):
                if id(callback) in previous_callbacks:
                    continue

                def complete_pause(engine, callback=callback):
                    try:
                        callback(engine)
                    except Exception as exc:
                        if not result.done():
                            result.set_exception(exc)

                callbacks[i] = complete_pause
        result.add_done_callback(restore)
    else:
        restore()
    return result


def _reset_cache_boundary(self, nonce):
    """Publish a successful reset and its caller's marker in one FIFO batch."""
    if getattr(self, "_uni_agent_checked_sleep_active", False):
        raise RuntimeError("cannot recover cache while native sleep is still in progress")
    from vllm.distributed.kv_events import AllBlocksCleared, KVEventBatch

    if not isinstance(nonce, str) or not nonce or len(nonce) > 128:
        raise ValueError("invalid cache reset boundary nonce")
    scheduler = self.scheduler
    publisher = getattr(scheduler, "kv_event_publisher", None)
    if publisher is None or getattr(publisher.get_publisher_config(), "publisher", None) != "zmq":
        raise RuntimeError("cache recovery requires native ZMQ events")
    if self.reset_prefix_cache(reset_connector=True) is not True:
        raise RuntimeError("engine did not confirm successful prefix cache reset")
    events = scheduler.kv_cache_manager.take_events()
    clears = [i for i, event in enumerate(events) if isinstance(event, AllBlocksCleared)]
    if not clears:
        raise RuntimeError("successful cache reset did not emit a native Clear")

    # Utility execution is serialized with scheduler steps. The final Clear
    # belongs to this checked reset; earlier pending generations are obsolete.
    class BoundaryBatch(KVEventBatch):
        cache_reset_boundary: dict | None = None

    marker = {RESET_BOUNDARY_KEY: {"schema_version": 1, "nonce": nonce}}
    # Existing typed KVEventBatch decoders ignore extra array fields. Keeping
    # events standard preserves compatibility with other native subscribers.
    publisher.publish(BoundaryBatch(ts=time.time(), events=events[clears[-1] :], cache_reset_boundary=marker))
    return True


_checked_sleep.__uni_agent_owner__ = _OWNER
_reset_cache_boundary.__uni_agent_owner__ = _OWNER


def register():
    if os.environ.get("UNI_AGENT_ENABLE_CACHE_LIFECYCLE") != "1":
        return
    from vllm.v1.engine.core import EngineCore

    methods = {CHECKED_SLEEP_METHOD: _checked_sleep, RESET_BOUNDARY_METHOD: _reset_cache_boundary}
    for name in methods:
        existing = getattr(EngineCore, name, None)
        if existing is not None and getattr(existing, "__uni_agent_owner__", None) != _OWNER:
            raise RuntimeError(f"Refusing to replace existing EngineCore.{name}")
    for name, method in methods.items():
        if getattr(EngineCore, name, None) is None:
            setattr(EngineCore, name, method)
