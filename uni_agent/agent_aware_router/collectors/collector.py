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

"""Collector — unified collector interface combining Transport + Parser."""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import Any

from ..config.collector import CollectorConfig
from ..debug import get_debug_var, is_debug_enabled
from ..utils.knob import coerce_knob_value
from .parse import InflightBlockUpdate, KVCacheUpdate, MetricsUpdate, Parser, StickyUpdate
from .parse.basic.inflight import InflightParser
from .sink import ContextualInflightParser, UpdateSink
from .transport.base import Transport

logger = logging.getLogger(__name__)


DEFAULT_COLLECTOR_KNOBS: dict[str, float | int] = {
    "http_timeout": 10.0,
    "base_retry_delay": 1.0,
    "max_retry_delay": 30.0,
    "max_retry_attempts": 5,
    "retry_backoff_factor": 2.0,
}


class Collector:
    """Transport lifecycle and typed parse dispatch, with an injectable application sink."""

    def __init__(self, transport: Transport, parser: Parser, sink: UpdateSink | None = None) -> None:
        if sink is None:
            from .updater import default_updater

            sink = default_updater(lambda: getattr(parser, "stats", None))
        self._transport = transport
        self._sink = sink
        self._parser = ContextualInflightParser(parser, sink) if isinstance(parser, InflightParser) else parser
        self._future: Future | None = None
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the collector — launch event-loop thread and subscribe."""

        def run_loop() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        def handler(raw_data: bytes | str, node_id: str) -> None:
            """Handler: parse and dispatch each update to the right store write path."""
            result = self._parser.parse(raw_data, node_id)
            if result is None:
                # None is normal for statistic parsers that skip an event
                # (e.g. StickyParser on_release); demote to debug to avoid per-turn noise.
                logger.debug(f"parser.parse returned no update: {result}")
                return
            updates = result if isinstance(result, list) else [result]
            for update in updates:
                if isinstance(update, KVCacheUpdate):
                    self._sink.apply_kv(update)
                elif isinstance(update, MetricsUpdate):
                    self._sink.apply_metrics(update)
                elif isinstance(update, StickyUpdate):
                    self._sink.apply_sticky(update)
                elif isinstance(update, InflightBlockUpdate):
                    self._sink.apply_inflight_blocks(update)
                else:
                    logger.debug(f"parser.parse returned an unknown update: {update}")

        if getattr(self._transport, "is_async", True):
            self._loop_thread = threading.Thread(
                target=run_loop,
                daemon=True,
            )
            self._loop_thread.start()

            self._future = asyncio.run_coroutine_threadsafe(
                self._transport.subscribe(handler),
                self._loop,
            )
        else:
            # CallbackTransport: subscribe only registers callbacks — run it on a
            # throwaway loop; stop() just unregisters.
            tmp_loop = asyncio.new_event_loop()
            try:
                tmp_loop.run_until_complete(self._transport.subscribe(handler))
            finally:
                tmp_loop.close()

    def stop(self) -> None:
        """
        Stop the collector — cancel tasks, drain cleanup, stop event-loop thread.
        """
        # Transport closes protocol-level resources (sockets/clients);
        # we own task cancellation and finally-block draining below.
        self._transport.stop()

        if self._loop.is_running():
            # Cancel all tasks and wait for their finally blocks inside the loop
            # so that aclose() runs while the loop is still alive.
            async def _cancel_and_drain() -> None:
                current = asyncio.current_task()
                tasks = [t for t in asyncio.all_tasks() if not t.done() and t is not current]
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            drain = asyncio.run_coroutine_threadsafe(_cancel_and_drain(), self._loop)
            try:
                drain.result(timeout=15)
            except Exception as exc:
                logger.debug(f"Error draining tasks on stop: {exc}")

            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10)
            self._loop_thread = None

        self._future = None


# ── Factory function ───────────────────────────────────────────────────


def _debug_knob(name: str) -> Any:
    """Return knob ``name``: the coerced ``UNI_AGENT_ROUTER_<NAME>`` override, or the default."""
    default = DEFAULT_COLLECTOR_KNOBS[name]
    if not is_debug_enabled():
        return default
    raw = get_debug_var(name)
    if raw is None:
        return default
    return coerce_knob_value(name, raw, default)


def get_collector(
    name: str,
    collectors_config: CollectorConfig,
    server_addresses: dict[str, str] | None = None,
    kv_event_endpoints: dict[str, list[str]] | None = None,
    balancer_handler=None,
) -> Collector:
    """Create a Collector by name — one place does both composition and config binding.

    Connection-type knobs fall back to ``DEFAULT_COLLECTOR_KNOBS``; per-collector
    ``UNI_AGENT_ROUTER_*`` debug overrides (when the debug master switch is on)
    are resolved into local values — never written onto ``collectors_config`` —
    and each collector only reads the knobs its transport uses.

    Args:
        name: Collector type — ``"vllm_metrics"`` or ``"vllm_zmq"``.
        collectors_config: ``CollectorConfig`` carrying connection-type knobs.
        server_addresses: ``{node_id: ip:port}`` for HTTP transport
            (used by ``"vllm_metrics"``).
        kv_event_endpoints: ``{node_id: [sub_addr, replay_addr]}`` for ZMQ
            transport (used by ``"vllm_zmq"``).

    Returns:
        Configured ``Collector`` instance.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name == "vllm_metrics":
        from .parse.vllm.metrics import VLLMMetricsParser
        from .transport.http import HTTPTransport

        transport = HTTPTransport(
            endpoints=server_addresses or {},
            interval=collectors_config.http_interval,
            http_timeout=_debug_knob("http_timeout"),
        )
        return Collector(transport, VLLMMetricsParser())

    if name == "vllm_zmq":
        from .parse.vllm.kv import VLLMKVParser
        from .transport.zmq import ZMQTransport

        transport = ZMQTransport(
            endpoints=kv_event_endpoints or {},
            base_retry_delay=_debug_knob("base_retry_delay"),
            max_retry_delay=_debug_knob("max_retry_delay"),
            max_retry_attempts=_debug_knob("max_retry_attempts"),
            retry_backoff_factor=_debug_knob("retry_backoff_factor"),
        )
        return Collector(transport, VLLMKVParser())

    if name == "sticky_stat":
        from .parse.basic.sticky import StickyParser
        from .transport.callback import CallbackTransport

        return Collector(CallbackTransport(balancer_handler), StickyParser())

    if name == "inflight_stat":
        from .transport.callback import CallbackTransport

        return Collector(CallbackTransport(balancer_handler), InflightParser())

    raise ValueError(
        f"Unknown collector: '{name}'. Available: ['vllm_metrics', 'vllm_zmq', 'sticky_stat', 'inflight_stat']"
    )
