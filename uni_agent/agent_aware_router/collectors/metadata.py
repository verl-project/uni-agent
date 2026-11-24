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

"""Bounded metadata discovery and recovery on a normal collector event loop."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx
import ray

from ..store.data_store import DataStore
from ..types.kv_cache import CacheMetadataUpdate
from .parse.base import Parser
from .transport.base import Transport

logger = logging.getLogger(__name__)


class MetadataParser(Parser):
    def parse(self, raw_data, replica_id):
        return CacheMetadataUpdate(replica_id, raw_data["metadata"], raw_data["revision"])


@dataclass
class _Source:
    handle: Any
    revision: int = -1
    attempts: int = 0
    failures: int = 0
    reason: str | None = None
    last_success: float | None = None
    next_retry: float | None = 0.0
    inflight: bool = False
    initial_done: bool = False
    last_logged: float = 0.0
    task: asyncio.Task | None = None
    ref: Any = None
    metadata_ready: bool = False
    server_address: str | None = None
    kv_endpoint: list[str] | None = None
    published_endpoints: tuple | None = None


class MetadataTransport(Transport):
    """Keep a single remote call per source until it really finishes.

    The startup timeout only bounds the initial wait. A late remote result still
    belongs to that attempt, so it must not overlap a replacement RPC.
    """

    _QUERY_TIMEOUT = 5.0
    _RETRY_DELAYS = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
    _MAX_CONCURRENT = 8

    def __init__(
        self,
        servers: dict[str, Any],
        data_store: DataStore | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        server_addresses: dict[str, str] | None = None,
        kv_event_endpoints: dict[str, list[str]] | None = None,
    ):
        self._store = data_store if data_store is not None else DataStore()
        self._clock = clock
        self._sources = {
            node: _Source(
                handle,
                server_address=(server_addresses or {}).get(node),
                kv_endpoint=(kv_event_endpoints or {}).get(node),
            )
            for node, handle in servers.items()
        }
        self._lock = threading.RLock()
        self._initial_done = threading.Event()
        self._stopped = False
        self._handler = None
        self._on_diagnostics: Callable[[], None] | None = None
        self._on_endpoints: Callable[[str, str, list[str] | None], None] | None = None
        self._requires_kv_events = True
        self._semaphore: asyncio.Semaphore | None = None

    @staticmethod
    def _collector_address(endpoint: Any) -> str:
        """Convert a vLLM TCP URI into the address expected by ZMQTransport."""
        if endpoint is None:
            return ""
        if not isinstance(endpoint, str) or not endpoint.startswith("tcp://"):
            raise RuntimeError(f"unsupported KV-event endpoint: {endpoint!r}")
        return endpoint.removeprefix("tcp://")

    @classmethod
    def _parse_kv_event_sources(cls, sources: Any) -> list[str] | None:
        """Convert vLLM's rank-keyed response into the current single-DP shape."""
        if not isinstance(sources, dict):
            raise RuntimeError(f"vLLM returned invalid KV-event sources: {sources!r}")
        if not sources:
            return None
        if len(sources) > 1:
            raise RuntimeError(
                "agent-aware-router does not support multiple DP ranks per rollout server yet: "
                f"reported ranks={sorted(sources)}"
            )

        rank, config = next(iter(sources.items()))
        if not isinstance(config, dict):
            raise RuntimeError(f"vLLM returned invalid KV-event config for DP rank {rank}: {config!r}")
        publisher = config.get("publisher")
        if publisher != "zmq":
            raise RuntimeError(f"unsupported KV-event publisher for DP rank {rank}: {publisher!r}")
        topic = config.get("topic")
        if not isinstance(topic, str):
            raise RuntimeError(f"invalid KV-event topic for DP rank {rank}: {topic!r}")
        return [
            cls._collector_address(config.get("endpoint")),
            cls._collector_address(config.get("replay_endpoint")),
            publisher,
            topic,
        ]

    @classmethod
    async def _fetch_kv_event_endpoints(cls, server_address: str) -> list[str] | None:
        async with httpx.AsyncClient(timeout=cls._QUERY_TIMEOUT, trust_env=False) as client:
            response = await client.get(f"http://{server_address}/kv_event_sources")
            response.raise_for_status()
            return cls._parse_kv_event_sources(response.json())

    def retry_delay(self, failures: int) -> float:
        return self._RETRY_DELAYS[min(max(failures - 1, 0), len(self._RETRY_DELAYS) - 1)]

    def wait_initial(self, timeout: float = 5.0) -> bool:
        return self._initial_done.wait(timeout)

    def get_stream_status(self) -> dict:
        with self._lock:
            return {
                node: {
                    "ready": source.next_retry is None and not source.inflight,
                    "reason": source.reason,
                    "attempts": source.attempts,
                    "failures": source.failures,
                    "last_success": source.last_success,
                    "next_retry": (
                        time.time() + max(0.0, source.next_retry - self._clock())
                        if source.next_retry is not None
                        else None
                    ),
                    "inflight": source.inflight,
                }
                for node, source in self._sources.items()
            }

    def _cancel_source(self, source: _Source) -> None:
        if source.ref is not None:
            try:
                if isinstance(source.ref, ray.ObjectRef):
                    ray.cancel(source.ref, force=False)
            except Exception:
                logger.debug("Failed to cancel metadata RPC", exc_info=True)
        if source.task is not None:
            source.task.cancel()

    def _failure(self, node: str, source: _Source, reason: str, *, diagnostic: str | None = None, details=None) -> None:
        now = self._clock()
        changed = source.reason != reason
        source.failures += 1
        source.reason = reason
        source.next_retry = now + self.retry_delay(source.failures)
        if diagnostic is not None:
            self._store.record_cache_diagnostic(node, "metadata", diagnostic, details)
            if self._on_diagnostics is not None:
                self._on_diagnostics()
        if changed or now - source.last_logged >= 300:
            logger.warning("Cache metadata unavailable replica=%s reason=%s failures=%s", node, reason, source.failures)
            source.last_logged = now

    async def _start_rpc(self, node, source, method):
        """Submit under the caller's discovery slot and return the original future after at most one wait window."""
        source.ref = method.remote()
        future = asyncio.ensure_future(source.ref)
        try:
            done, _ = await asyncio.wait({future}, timeout=self._QUERY_TIMEOUT)
        except asyncio.CancelledError:
            future.cancel()
            raise
        if not done:
            self._failure(node, source, "query_timeout", diagnostic="query_timeout")
            source.initial_done = True
        return future

    def _publish_endpoints(self, node, source, address, on_endpoints):
        if self._stopped:
            return
        key = address, tuple(source.kv_endpoint) if source.kv_endpoint is not None else None
        if key != source.published_endpoints:
            on_endpoints(node, address, source.kv_endpoint)
            source.published_endpoints = key

    async def _discover_endpoints(self, node, source):
        on_endpoints = self._on_endpoints
        if on_endpoints is None:
            return True
        assert self._semaphore is not None
        if source.server_address is None:
            method = getattr(source.handle, "get_server_address", None)
            if method is None:
                self._failure(node, source, "server_address_method_unavailable", diagnostic="endpoint_discovery_failed")
                return False
            async with self._semaphore:
                if self._stopped:
                    return False
                address_future = await self._start_rpc(node, source, method)
            ip, port = await address_future
            source.ref = None
            if self._stopped or self._store.metadata_revision(node) != source.revision:
                source.next_retry = 0.0
                return False
            source.server_address = f"{ip}:{port}"
        self._publish_endpoints(node, source, source.server_address, on_endpoints)
        if self._requires_kv_events and source.kv_endpoint is None:
            async with self._semaphore:
                if self._stopped:
                    return False
                endpoint = await asyncio.wait_for(
                    self._fetch_kv_event_endpoints(source.server_address), timeout=self._QUERY_TIMEOUT
                )
            if self._stopped or self._store.metadata_revision(node) != source.revision:
                source.next_retry = 0.0
                return False
            if endpoint is None:
                self._failure(node, source, "kv_event_endpoint_unavailable", diagnostic="endpoint_discovery_failed")
                return False
            source.kv_endpoint = endpoint
            self._publish_endpoints(node, source, source.server_address, on_endpoints)
        return True

    async def _query(self, node: str, source: _Source) -> None:
        failure_code = "metadata_query_failed"
        assert self._semaphore is not None
        try:
            async with self._semaphore:
                if self._stopped:
                    return
                source.inflight = True
                source.attempts += 1
                revision = self._store.metadata_revision(node)
                if revision != source.revision:
                    source.metadata_ready = False
                    source.revision = revision
                metadata_future = None
                if not source.metadata_ready:
                    method = getattr(source.handle, "get_kv_cache_metadata", None)
                    if method is not None:
                        metadata_future = await self._start_rpc(node, source, method)
            if not source.metadata_ready:
                metadata = (
                    {"status": "unsupported", "reason": "metadata_method_unavailable"}
                    if metadata_future is None
                    else await metadata_future
                )
                source.ref = None
                if self._stopped or self._store.metadata_revision(node) != revision:
                    source.next_retry = 0.0
                    return
                if not isinstance(metadata, dict):
                    metadata = {"status": "error", "reason": "invalid_metadata_response"}
                self._handler({"metadata": metadata, "revision": revision}, node)
                status = self._store.metadata_status(node)
                reason = None
                if metadata.get("status") != "ok":
                    reason = metadata.get("reason") or metadata.get("status") or "metadata_status_missing"
                elif metadata.get("metadata_errors"):
                    error = next(iter(metadata["metadata_errors"].values()))
                    reason = error.get("reason", "metadata_part_unavailable") if isinstance(error, dict) else str(error)
                elif not status.get("ready", False):
                    reason = status.get("reason") or "metadata_not_ready"
                source.metadata_ready = reason is None
                if reason:
                    self._failure(node, source, reason)
            failure_code = "endpoint_discovery_failed"
            if not await self._discover_endpoints(node, source) or not source.metadata_ready:
                return
            if source.reason is not None:
                logger.info("Cache metadata recovered replica=%s previous_reason=%s", node, source.reason)
            source.reason = None
            source.failures = 0
            source.last_success = time.time()
            source.next_retry = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._stopped:
                self._failure(
                    node,
                    source,
                    f"{failure_code}:{type(exc).__name__}:{exc}",
                    diagnostic=failure_code,
                    details={"error_type": type(exc).__name__, "message": str(exc)},
                )
        finally:
            source.ref = None
            source.inflight = False
            source.initial_done = True

    async def subscribe(self, handler) -> None:
        self._handler = handler
        self._semaphore = asyncio.Semaphore(self._MAX_CONCURRENT)
        self._stopped = False
        self._initial_done.clear()
        with self._lock:
            for source in self._sources.values():
                source.initial_done = False
                source.metadata_ready = False
                source.published_endpoints = None
                source.next_retry = 0.0
        try:
            while not self._stopped:
                now = self._clock()
                requests = self._store.metadata_refresh_requests()
                with self._lock:
                    for node, source in self._sources.items():
                        if source.task is not None and not source.task.done():
                            continue
                        requested = requests.get(node)
                        if requested is not None and requested != source.revision:
                            source.next_retry = 0.0
                        if source.next_retry is not None and source.next_retry <= now:
                            source.task = asyncio.create_task(self._query(node, source))
                    if all(source.initial_done for source in self._sources.values()):
                        self._initial_done.set()
                await asyncio.sleep(0.1)
        finally:
            with self._lock:
                sources = list(self._sources.values())
                for source in sources:
                    self._cancel_source(source)
            await asyncio.gather(
                *(source.task for source in sources if source.task is not None), return_exceptions=True
            )
            self._initial_done.set()

    def stop(self) -> None:
        self._stopped = True
        with self._lock:
            for source in self._sources.values():
                self._cancel_source(source)
        self._initial_done.set()
