"""Ray actor composition for the Global telemetry broker and shadow Reporter."""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import ray

from uni_agent.events import (
    GATEWAY_TELEMETRY_SUBSCRIPTION,
    GENERATION_FINISHED,
    SESSION_CLOSED,
    SESSION_OPENED,
    DeliveryMode,
    GlobalSubscriptionEndpoint,
    GlobalSubscriptionSpec,
    LocalEventBus,
    RayGlobalEventBus,
    Scope,
    SubscriptionSpec,
    TelemetryAck,
    TelemetryAckStatus,
    TelemetryBatch,
    TelemetryReporter,
)


@dataclass(frozen=True, slots=True)
class GlobalTelemetryRuntimeConfig:
    event_types: tuple[str, ...] = (SESSION_OPENED, GENERATION_FINISHED, SESSION_CLOSED)
    max_batch_events: int = 128
    max_batch_bytes: int = 256 * 1024
    broker_max_queue_events: int = 8192
    broker_max_queue_bytes: int = 16 * 1024 * 1024
    subscriber_max_queue_events: int = 4096
    subscriber_max_queue_bytes: int = 8 * 1024 * 1024
    endpoint_max_queue_events: int = 4096
    endpoint_max_queue_bytes: int = 8 * 1024 * 1024
    reporter_max_queue_events: int = 4096
    reporter_max_queue_bytes: int = 8 * 1024 * 1024
    max_retries: int = 3
    retry_backoff_s: float = 0.01
    dedup_entries: int = 8192

    def __post_init__(self) -> None:
        numeric_budgets = (
            self.max_batch_events,
            self.max_batch_bytes,
            self.broker_max_queue_events,
            self.broker_max_queue_bytes,
            self.subscriber_max_queue_events,
            self.subscriber_max_queue_bytes,
            self.endpoint_max_queue_events,
            self.endpoint_max_queue_bytes,
            self.reporter_max_queue_events,
            self.reporter_max_queue_bytes,
            self.max_retries,
            self.dedup_entries,
        )
        if not self.event_types:
            raise ValueError("Global telemetry event_types must be non-empty")
        if min(numeric_budgets) <= 0:
            raise ValueError("Global telemetry runtime budgets must be positive")
        if self.retry_backoff_s < 0:
            raise ValueError("Global telemetry retry_backoff_s must be non-negative")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> GlobalTelemetryRuntimeConfig:
        event_types = data.get("event_types", (SESSION_OPENED, GENERATION_FINISHED, SESSION_CLOSED))
        if isinstance(event_types, str):
            raise ValueError("Global telemetry event_types must be a sequence of event names")
        return cls(
            event_types=tuple(str(event_type) for event_type in event_types),
            max_batch_events=int(data.get("max_batch_events", 128)),
            max_batch_bytes=int(data.get("max_batch_bytes", 256 * 1024)),
            broker_max_queue_events=int(data.get("broker_max_queue_events", 8192)),
            broker_max_queue_bytes=int(data.get("broker_max_queue_bytes", 16 * 1024 * 1024)),
            subscriber_max_queue_events=int(data.get("subscriber_max_queue_events", 4096)),
            subscriber_max_queue_bytes=int(data.get("subscriber_max_queue_bytes", 8 * 1024 * 1024)),
            endpoint_max_queue_events=int(data.get("endpoint_max_queue_events", 4096)),
            endpoint_max_queue_bytes=int(data.get("endpoint_max_queue_bytes", 8 * 1024 * 1024)),
            reporter_max_queue_events=int(data.get("reporter_max_queue_events", 4096)),
            reporter_max_queue_bytes=int(data.get("reporter_max_queue_bytes", 8 * 1024 * 1024)),
            max_retries=int(data.get("max_retries", 3)),
            retry_backoff_s=float(data.get("retry_backoff_s", 0.01)),
            dedup_entries=int(data.get("dedup_entries", 8192)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _GlobalEventBusActor:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = GlobalTelemetryRuntimeConfig.from_mapping(config)
        self._bus = RayGlobalEventBus(
            max_queue_events=self._config.broker_max_queue_events,
            max_queue_bytes=self._config.broker_max_queue_bytes,
            max_batch_events=self._config.max_batch_events,
            max_batch_bytes=self._config.max_batch_bytes,
            dedup_entries=self._config.dedup_entries,
        )

    def register_gateway_reporter(self, endpoint) -> None:
        def send_batch(subscription_id: str, batch: dict[str, Any]) -> Mapping[str, Any]:
            return endpoint.receive_global_batch.remote(subscription_id, batch).future().result()

        self._bus.register_subscription(
            GlobalSubscriptionSpec(
                subscription_id=GATEWAY_TELEMETRY_SUBSCRIPTION,
                event_types=self._config.event_types,
                max_queue_events=self._config.subscriber_max_queue_events,
                max_queue_bytes=self._config.subscriber_max_queue_bytes,
                max_retries=self._config.max_retries,
                retry_backoff_s=self._config.retry_backoff_s,
            ),
            send_batch,
        )

    def publish_global_batch(self, data: Mapping[str, Any]) -> dict[str, Any]:
        try:
            batch = TelemetryBatch.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            return TelemetryAck(
                batch_id=str(data.get("batch_id") or "invalid-batch"),
                status=TelemetryAckStatus.BUSY,
                reason=f"invalid telemetry batch: {type(exc).__name__}",
            ).to_dict()
        return self._bus.publish_batch(batch).to_dict()

    def get_global_status(self) -> dict[str, Any]:
        return asdict(self._bus.health)

    def shutdown(self, timeout_s: float = 5.0) -> bool:
        return self._bus.close(drain=True, timeout_s=timeout_s)


class _GlobalTelemetryReporterActor:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = GlobalTelemetryRuntimeConfig.from_mapping(config)
        self._lock = threading.RLock()
        self._event_counts: Counter[str] = Counter()
        self._source_counts: Counter[str] = Counter()
        self._bus = LocalEventBus()
        self._reporter = TelemetryReporter(
            self._bus,
            SubscriptionSpec(
                subscription_id=GATEWAY_TELEMETRY_SUBSCRIPTION,
                event_types=self._config.event_types,
                scope=Scope.GLOBAL,
                delivery=DeliveryMode.QUEUED,
                max_queue_events=self._config.reporter_max_queue_events,
                max_queue_bytes=self._config.reporter_max_queue_bytes,
            ),
            self._report,
        )
        self._endpoint = GlobalSubscriptionEndpoint(
            self._bus,
            subscription_id=GATEWAY_TELEMETRY_SUBSCRIPTION,
            max_queue_events=self._config.endpoint_max_queue_events,
            max_queue_bytes=self._config.endpoint_max_queue_bytes,
            max_batch_events=self._config.max_batch_events,
            max_batch_bytes=self._config.max_batch_bytes,
            dedup_entries=self._config.dedup_entries,
        )

    def _report(self, event) -> None:
        with self._lock:
            self._event_counts[event.event_type] += 1
            self._source_counts[event.producer_id] += 1

    def receive_global_batch(self, subscription_id: str, data: Mapping[str, Any]) -> dict[str, Any]:
        try:
            batch = TelemetryBatch.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            return TelemetryAck(
                batch_id=str(data.get("batch_id") or "invalid-batch"),
                status=TelemetryAckStatus.BUSY,
                reason=f"invalid telemetry batch: {type(exc).__name__}",
            ).to_dict()
        return self._endpoint.receive_batch(subscription_id, batch).to_dict()

    def get_telemetry_status(self) -> dict[str, Any]:
        with self._lock:
            event_counts = dict(self._event_counts)
            source_counts = dict(self._source_counts)
        return {
            "namespace": "shadow",
            "event_counts": event_counts,
            "source_counts": source_counts,
            "endpoint": asdict(self._endpoint.health),
            "reporter": asdict(self._reporter.health),
        }

    def shutdown(self, timeout_s: float = 5.0) -> bool:
        endpoint_closed = self._endpoint.close(drain=True, timeout_s=timeout_s)
        reporter_closed = self._reporter.close(drain=True, timeout_s=timeout_s)
        self._bus.close()
        return endpoint_closed and reporter_closed


GlobalEventBusActor = ray.remote(_GlobalEventBusActor)
GlobalTelemetryReporterActor = ray.remote(_GlobalTelemetryReporterActor)


class GlobalTelemetryRuntime:
    """Driver-owned handles for one broker and one isolated shadow Reporter."""

    def __init__(self, broker, reporter) -> None:
        self.publish_target = broker
        self.reporter = reporter

    @classmethod
    def start(cls, config: GlobalTelemetryRuntimeConfig) -> GlobalTelemetryRuntime:
        config_data = config.to_dict()
        reporter = GlobalTelemetryReporterActor.remote(config_data)
        broker = GlobalEventBusActor.remote(config_data)
        try:
            ray.get(broker.register_gateway_reporter.remote(reporter))
        except BaseException:
            ray.kill(broker, no_restart=True)
            ray.kill(reporter, no_restart=True)
            raise
        return cls(broker, reporter)

    async def status(self) -> dict[str, Any]:
        """Return sequential diagnostic snapshots, not a cross-actor transaction."""
        broker = await self.publish_target.get_global_status.remote()
        reporter = await self.reporter.get_telemetry_status.remote()
        return {"broker": broker, "reporter": reporter}

    async def shutdown(self, timeout_s: float = 5.0) -> bool:
        broker_closed = await self.publish_target.shutdown.remote(timeout_s)
        reporter_closed = await self.reporter.shutdown.remote(timeout_s)
        return broker_closed and reporter_closed

    def shutdown_blocking(self, timeout_s: float = 5.0) -> bool:
        broker_closed = ray.get(self.publish_target.shutdown.remote(timeout_s))
        reporter_closed = ray.get(self.reporter.shutdown.remote(timeout_s))
        return broker_closed and reporter_closed


__all__ = ["GlobalTelemetryRuntime", "GlobalTelemetryRuntimeConfig"]
