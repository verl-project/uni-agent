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
from typing import Any

from ..config.collector import CollectorConfig
from ..debug import get_debug_var, is_debug_enabled
from ..store.data_store import DataStore
from ..utils.knob import coerce_knob_value
from ._loop_runner import _LoopRunner
from .parse import InflightUpdate, KVCacheUpdate, MetricsUpdate, Parser, StickyUpdate
from .parse.basic.inflight import InflightParser
from .sink import UpdateSink
from .transport.base import Transport
from .updater import default_updater

logger = logging.getLogger(__name__)


DEFAULT_COLLECTOR_KNOBS: dict[str, float | int] = {
    "http_timeout": 10.0,
    "base_retry_delay": 1.0,
    "max_retry_delay": 30.0,
    "max_retry_attempts": 5,
    "retry_backoff_factor": 2.0,
}


class Collector:
    """Transport lifecycle and typed parse dispatch with an injectable application sink."""

    def __init__(self, transport: Transport, parser: Parser, sink: UpdateSink | None = None) -> None:
        if sink is None:
            sink = default_updater(lambda: getattr(parser, "stats", None))
        self._transport = transport
        self._parser = parser
        self._sink = sink
        self._dispatch_lock = threading.Lock()
        self._loop_runner = _LoopRunner(transport) if getattr(transport, "is_async", True) else None
        self._sync_started = False
        self._sync_subscription_error: str | None = None

    def _ensure_startable(self) -> None:
        if self._loop_runner is not None:
            self._loop_runner._ensure_startable()
        elif self._sync_started:
            raise RuntimeError("stop the collector before starting it again")

    def start(self) -> None:
        """Start the collector — launch event-loop thread and subscribe."""
        self._ensure_startable()
        if self._loop_runner is not None:
            if not self._loop_runner.start():
                try:
                    self.stop()
                except Exception as exc:
                    raise TimeoutError("collector event loop startup timed out") from exc
                raise TimeoutError("collector event loop startup timed out")
            self._loop_runner.subscribe(self._handle_raw_data)
        else:
            # CallbackTransport: subscribe only registers callbacks — run it on a
            # throwaway loop; stop() just unregisters.
            self._sync_subscription_error = None
            tmp_loop = asyncio.new_event_loop()
            # A failed subscribe may have registered only some callbacks.
            self._sync_started = True
            try:
                tmp_loop.run_until_complete(self._transport.subscribe(self._handle_raw_data))
            except Exception as exc:
                self._sync_subscription_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                tmp_loop.close()

    def get_status(self) -> dict:
        """Return independent subscription and per-source stream diagnostics."""
        subscription = (
            self._loop_runner.get_status()
            if self._loop_runner is not None
            else {
                "active": self._sync_started and self._sync_subscription_error is None,
                "error": self._sync_subscription_error,
            }
        )
        return {
            "transport": type(self._transport).__name__,
            "subscription": subscription,
            "streams": self._transport.get_stream_status(),
        }

    def _handle_raw_data(self, raw_data: Any, replica_id: str) -> None:
        """Parse and apply each message in order, including synchronous callbacks."""
        with self._dispatch_lock:
            result = self._parser.parse(raw_data, replica_id)
            if result is None:
                return
            updates = result if isinstance(result, list) else [result]
            for update in updates:
                if isinstance(update, KVCacheUpdate):
                    self._sink.apply_kv(update)
                elif isinstance(update, MetricsUpdate):
                    self._sink.apply_metrics(update)
                elif isinstance(update, StickyUpdate):
                    self._sink.apply_sticky(update)
                elif isinstance(update, InflightUpdate):
                    self._sink.apply_inflight(update)
                else:
                    logger.debug("parser returned an unknown update: %r", update)

    def stop(self, timeout: float = 15.0) -> None:
        """
        Stop the collector — cancel tasks, drain cleanup, stop event-loop thread.

        Timeouts remain retryable. A dead-loop failure is reported once; later
        calls are no-ops, status retains the error, and the collector must be replaced.
        """
        if self._loop_runner is not None:
            self._loop_runner.stop(timeout)
        else:
            self._transport.stop()
        self._sync_started = False


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
        server_addresses: ``{replica_id: ip:port}`` for HTTP transport
            (used by ``"vllm_metrics"``).
        kv_event_endpoints: ``{replica_id: [sub_addr, replay_addr]}`` for ZMQ
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

        kv_parser = VLLMKVParser()
        data_store = DataStore()

        def reset_stream(replica_id: str) -> None:
            # Gate routing reads before discarding the parser's old hash chain.
            data_store.invalidate_kv_replica(replica_id)
            kv_parser.reset_replica(replica_id)

        def unavailable_stream(replica_id: str) -> None:
            reset_stream(replica_id)
            data_store.enable_kv_dispatch_estimates(replica_id)

        transport = ZMQTransport(
            endpoints=kv_event_endpoints or {},
            on_stream_reset=reset_stream,
            on_stream_ready=data_store.mark_kv_replica_ready,
            on_stream_unavailable=unavailable_stream,
            base_retry_delay=_debug_knob("base_retry_delay"),
            max_retry_delay=_debug_knob("max_retry_delay"),
            max_retry_attempts=_debug_knob("max_retry_attempts"),
            retry_backoff_factor=_debug_knob("retry_backoff_factor"),
        )
        return Collector(transport, kv_parser)

    if name == "sticky_stat":
        from .parse.basic.sticky import StickyParser
        from .transport.callback import CallbackTransport

        return Collector(CallbackTransport(balancer_handler), StickyParser())

    if name == "inflight_stat":
        from .transport.callback import CallbackTransport

        return Collector(CallbackTransport(balancer_handler, accounting=True), InflightParser())

    raise ValueError(
        f"Unknown collector: '{name}'. Available: ['vllm_metrics', 'vllm_zmq', 'sticky_stat', 'inflight_stat']"
    )
