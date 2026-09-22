"""Gateway-owned Task Metrics Projector backed by bounded per-session state."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any

from uni_agent.events import (
    GENERATION_FINISHED,
    SESSION_CLOSED,
    SESSION_OPENED,
    DeliveryMode,
    Event,
    LocalEventBus,
    Scope,
    Subscription,
    SubscriptionSpec,
)

from .model import AggregationType, MetricsFragment, MetricSummary

_GATEWAY_REQUESTS = "gateway.requests"
_GATEWAY_REQUEST_SECONDS = "gateway.request_s"
_GATEWAY_ENCODE_SECONDS = "gateway.encode_s"
_GATEWAY_BACKEND_SECONDS = "gateway.backend_generate_s"
_GATEWAY_DECODE_SECONDS = "gateway.decode_s"


@dataclass(slots=True)
class _MetricAccumulator:
    aggregation: AggregationType
    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    last: float = 0.0

    def observe(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"metric value must be finite, got {value}")
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.last = value

    def snapshot(self) -> MetricSummary:
        if self.count == 0:
            raise ValueError("cannot snapshot an empty metric")
        return MetricSummary(
            aggregation=self.aggregation,
            count=self.count,
            total=self.total,
            minimum=self.minimum,
            maximum=self.maximum,
            last=self.last,
        )


@dataclass(slots=True)
class _SessionMetrics:
    episode_id: str
    metrics: dict[str, _MetricAccumulator] = field(default_factory=dict)
    complete: bool = True
    closed: bool = False
    revision: int = 0
    errors: list[str] = field(default_factory=list)

    def observe(self, name: str, value: float, aggregation: AggregationType) -> None:
        metric = self.metrics.get(name)
        if metric is None:
            metric = _MetricAccumulator(aggregation)
            self.metrics[name] = metric
        elif metric.aggregation is not aggregation:
            raise ValueError(f"aggregation mismatch for {name}: {metric.aggregation.value} != {aggregation.value}")
        metric.observe(value)

    def mark_incomplete(self, error: str) -> None:
        self.complete = False
        if error not in self.errors:
            self.errors.append(error)


class GatewayTaskMetricsProjector:
    """Project Gateway facts into bounded Task metrics, bucketed by session."""

    def __init__(
        self,
        bus: LocalEventBus,
        *,
        source_instance: str,
        subscription_id: str = "gateway-task-metrics",
    ) -> None:
        if not source_instance:
            raise ValueError("source_instance must be non-empty")
        self._source_instance = source_instance
        self._lock = threading.RLock()
        self._sessions: dict[str, _SessionMetrics] = {}
        self._subscription: Subscription = bus.subscribe(
            SubscriptionSpec(
                subscription_id=subscription_id,
                event_types=(SESSION_OPENED, GENERATION_FINISHED, SESSION_CLOSED),
                producer_ids=(source_instance,),
                scope=Scope.LOCAL,
                delivery=DeliveryMode.INLINE,
            ),
            self._handle_event,
        )

    def _handle_event(self, event: Event) -> None:
        session_id = event.context.session_id
        if not session_id:
            return
        episode_id = event.context.episode_id or session_id
        with self._lock:
            metrics = self._sessions.get(session_id)
            if metrics is None:
                metrics = _SessionMetrics(episode_id=episode_id)
                self._sessions[session_id] = metrics
                if event.event_type != SESSION_OPENED:
                    metrics.mark_incomplete(f"first event was {event.event_type}, expected {SESSION_OPENED}")
            elif metrics.episode_id != episode_id:
                metrics.mark_incomplete(
                    f"episode changed for session {session_id}: {metrics.episode_id} -> {episode_id}"
                )

            try:
                if event.event_type == SESSION_OPENED:
                    self._on_session_opened(metrics)
                elif event.event_type == GENERATION_FINISHED:
                    self._on_generation_finished(metrics, event)
                elif event.event_type == SESSION_CLOSED:
                    self._on_session_closed(metrics)
            except (KeyError, TypeError, ValueError) as error:
                metrics.mark_incomplete(f"{event.event_type}: {error}")

    @staticmethod
    def _on_session_opened(metrics: _SessionMetrics) -> None:
        if metrics.revision != 0:
            metrics.mark_incomplete("duplicate SessionOpened")
        metrics.revision += 1

    @staticmethod
    def _duration_seconds(event: Event, field_name: str, *, required: bool) -> float | None:
        raw_value: Any = event.payload.get(field_name)
        if raw_value is None and not required:
            return None
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise TypeError(f"{field_name} must be a non-negative number or null")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative finite number")
        return value / 1_000_000_000

    def _on_generation_finished(self, metrics: _SessionMetrics, event: Event) -> None:
        metrics.observe(_GATEWAY_REQUESTS, 1.0, AggregationType.SUM)
        request_seconds = self._duration_seconds(event, "duration_ns", required=True)
        if request_seconds is not None:
            metrics.observe(_GATEWAY_REQUEST_SECONDS, request_seconds, AggregationType.SUM)

        for payload_name, metric_name in (
            ("prepare_duration_ns", _GATEWAY_ENCODE_SECONDS),
            ("backend_duration_ns", _GATEWAY_BACKEND_SECONDS),
            ("decode_duration_ns", _GATEWAY_DECODE_SECONDS),
        ):
            seconds = self._duration_seconds(event, payload_name, required=False)
            if seconds is not None:
                metrics.observe(metric_name, seconds, AggregationType.SUM)
        metrics.revision += 1

    @staticmethod
    def _on_session_closed(metrics: _SessionMetrics) -> None:
        metrics.closed = True
        metrics.revision += 1

    def finalize_session(self, session_id: str) -> MetricsFragment:
        with self._lock:
            state = self._sessions.pop(session_id, None)
        if state is None:
            return MetricsFragment(
                episode_id=session_id,
                source_role="gateway",
                source_instance=self._source_instance,
                fragment_id=f"{self._source_instance}:gateway:{session_id}",
                schema_version=1,
                revision=1,
                complete=False,
                metrics={},
                incomplete_reasons=("session metrics were not initialized",),
            )

        errors = list(state.errors)
        if not state.closed:
            errors.append("SessionClosed was not observed")
        return MetricsFragment(
            episode_id=state.episode_id,
            source_role="gateway",
            source_instance=self._source_instance,
            fragment_id=f"{self._source_instance}:gateway:{session_id}",
            schema_version=1,
            revision=max(state.revision, 1),
            complete=state.complete and state.closed,
            metrics={name: metric.snapshot() for name, metric in state.metrics.items() if metric.count},
            incomplete_reasons=tuple(errors),
        )

    def discard_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def close(self) -> None:
        self._subscription.close()
        with self._lock:
            self._sessions.clear()
