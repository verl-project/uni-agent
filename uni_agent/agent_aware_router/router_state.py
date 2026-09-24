"""Ownership-aware projection for Router callback state."""

from __future__ import annotations

import threading
from collections import OrderedDict
from enum import Enum
from typing import Any

from .collectors.collector import (
    PROMPT_LEN_ROW_KEY,
    TURN_ROW_KEY,
    commit_inflight_delta,
    log_dispatch_stats,
)
from .collectors.parse import MetricsUpdate, StickyUpdate
from .store import DataStore
from .types import MetricKey


class RouterStateMode(str, Enum):
    LEGACY = "legacy"
    SHADOW = "shadow"
    PROJECTOR = "projector"


class RouterStickyStateProjector:
    """Project one bounded sticky-binding view with an explicit commit owner."""

    def __init__(self, mode: RouterStateMode, store: DataStore, *, max_bindings: int = 10_000) -> None:
        if mode is RouterStateMode.LEGACY:
            raise ValueError("legacy mode does not create a Router sticky Projector")
        if max_bindings <= 0:
            raise ValueError("max_bindings must be positive")
        self._mode = mode
        self._store = store
        self._max_bindings = max_bindings
        self._bindings: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.RLock()
        self._updates = 0
        self._parity_mismatches = 0
        self._last_error: str | None = None

    @property
    def healthy(self) -> bool:
        with self._lock:
            return self._last_error is None

    def apply(self, update: StickyUpdate) -> None:
        """Commit one update when the Projector owns this input family."""
        self._reduce(update, commit=True)

    def observe(self, update: StickyUpdate) -> None:
        """Reduce comparison state after the legacy writer has committed."""
        self._reduce(update, commit=False)

    def binding(self, request_id: str) -> str | None:
        with self._lock:
            return self._bindings.get(request_id)

    def clear(self, *, commit: bool) -> int:
        """Clear sticky bindings through the selected owner and reset comparison state."""
        try:
            with self._lock:
                cleared = self._store.clear_sticky_bindings() if commit else len(self._bindings)
                self._bindings.clear()
                self._updates += 1
                if self._store.sticky_status()["size"] != 0:
                    self._parity_mismatches += 1
                return cleared
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self._mode.value,
                "commit_owner": "projector" if self._mode is RouterStateMode.PROJECTOR else "legacy",
                "healthy": self._last_error is None,
                "last_error": self._last_error,
                "updates": self._updates,
                "binding_count": len(self._bindings),
                "parity_mismatches": self._parity_mismatches,
            }

    def _reduce(self, update: StickyUpdate, *, commit: bool) -> None:
        try:
            touched = self._validate(update)
            with self._lock:
                if commit:
                    self._commit(update)
                self._project(update)
                self._updates += 1
                self._parity_mismatches += sum(
                    self._store.get_sticky_binding(request_id) != self._bindings.get(request_id)
                    for request_id in touched
                )
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    @staticmethod
    def _validate(update: StickyUpdate) -> tuple[str, ...]:
        if update.action == "put":
            if not update.request_id or not update.replica_id:
                raise ValueError("sticky put requires request_id and replica_id")
            return (update.request_id,)
        if update.action == "invalidate":
            if not update.request_id:
                raise ValueError("sticky invalidate requires request_id")
            return (update.request_id,)
        if update.action == "invalidate_replica":
            if not update.replica_ids:
                raise ValueError("sticky replica invalidation requires replica_ids")
            return ()
        raise ValueError(f"unsupported sticky action {update.action!r}")

    def _commit(self, update: StickyUpdate) -> None:
        if update.action == "put":
            self._store.put_sticky_binding(update.request_id, update.replica_id)
        elif update.action == "invalidate":
            self._store.invalidate_sticky_binding(update.request_id)
        else:
            for replica_id in update.replica_ids:
                self._store.invalidate_sticky_replica(replica_id)

    def _project(self, update: StickyUpdate) -> None:
        if update.action == "put":
            self._bindings[update.request_id] = update.replica_id
            self._bindings.move_to_end(update.request_id)
            while len(self._bindings) > self._max_bindings:
                self._bindings.popitem(last=False)
        elif update.action == "invalidate":
            self._bindings.pop(update.request_id, None)
        else:
            removed = set(update.replica_ids)
            self._bindings = OrderedDict(
                (request_id, replica_id)
                for request_id, replica_id in self._bindings.items()
                if replica_id not in removed
            )


class RouterInflightStateProjector:
    """Own or shadow the inflight ``MetricsUpdate`` delta family with one commit owner.

    The legacy writer is the ``inflight_stat`` Collector's
    ``Collector._write_metrics_update`` delta branch; this projector owns the same
    family in projector mode through the shared :func:`commit_inflight_delta`, so
    the store state, insight ``WriteEvent`` stream, and throttled dispatch log
    stay identical regardless of owner. The comparison view reduces the same
    deltas read-only for parity — the family is bounded by replica count times a
    fixed key set, so no eviction is needed (unlike sticky's LRU bindings).
    """

    _FAMILY_KEYS: tuple[str, ...] = (
        MetricKey.INFLIGHT_COUNT,
        MetricKey.INFLIGHT_TOKENS,
        MetricKey.DISPATCHED_COUNT,
        MetricKey.COMPLETED_COUNT,
        MetricKey.PROMPT_LEN_SUM,
        MetricKey.INFLIGHT_TURN_SUM,
    )

    def __init__(self, mode: RouterStateMode, store: DataStore) -> None:
        if mode is RouterStateMode.LEGACY:
            raise ValueError("legacy mode does not create a Router inflight Projector")
        self._mode = mode
        self._store = store
        self._ledger: dict[str, dict[str, float]] = {}
        self._lock = threading.RLock()
        self._updates = 0
        self._parity_mismatches = 0
        self._last_error: str | None = None
        self._dispatch_last_log = 0.0

    @property
    def healthy(self) -> bool:
        with self._lock:
            return self._last_error is None

    def apply(self, update: MetricsUpdate) -> None:
        """Commit one inflight delta update when the Projector owns this family."""
        try:
            self._validate(update)
            commit_inflight_delta(self._store, update)
            with self._lock:
                self._accumulate(update, effective=self._effective_deltas(update))
                self._updates += 1
                self._check_parity(update.node_id)
            self._dispatch_last_log = log_dispatch_stats(self._store, self._dispatch_last_log)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def observe(self, update: MetricsUpdate) -> None:
        """Reduce comparison state after the legacy writer has committed."""
        try:
            self._validate(update)
            with self._lock:
                self._accumulate(update, effective=self._effective_deltas(update))
                self._updates += 1
                self._check_parity(update.node_id)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self._mode.value,
                "commit_owner": "projector" if self._mode is RouterStateMode.PROJECTOR else "legacy",
                "healthy": self._last_error is None,
                "last_error": self._last_error,
                "updates": self._updates,
                "parity_mismatches": self._parity_mismatches,
                "tracked_replicas": len(self._ledger),
                "replica_inflight": {
                    node: values.get(MetricKey.INFLIGHT_COUNT, 0) for node, values in self._ledger.items()
                },
            }

    @staticmethod
    def _validate(update: MetricsUpdate) -> None:
        if not update.is_delta:
            raise ValueError("inflight projection requires a delta MetricsUpdate")
        if not update.node_id:
            raise ValueError("inflight delta requires node_id")

    def _effective_deltas(self, update: MetricsUpdate) -> dict[str, float]:
        """Fold per-request rows into the update's deltas.

        ``commit_inflight_delta`` performs this fold while committing. This
        read-only twin reconstructs the same effective deltas for the comparison
        ledger and always runs after that commit landed: the acquire-side turn
        row already includes the increment the commit applied, and release reads
        rows that persist after the release write — release never deletes them,
        so the post-commit read returns exactly what the commit subtracted.
        """
        effective = dict(update.metrics)
        is_acquire = MetricKey.DISPATCHED_COUNT in effective
        is_release = MetricKey.COMPLETED_COUNT in effective
        if is_acquire and update.request_id is not None:
            effective[MetricKey.INFLIGHT_TURN_SUM] = self._store.get_per_request(update.request_id, TURN_ROW_KEY, 0)
        elif is_release and update.request_id is not None:
            effective[MetricKey.INFLIGHT_TURN_SUM] = -self._store.get_per_request(update.request_id, TURN_ROW_KEY, 0)
            effective[MetricKey.INFLIGHT_TOKENS] = -self._store.get_per_request(
                update.request_id, PROMPT_LEN_ROW_KEY, 0
            )
        return effective

    def _accumulate(self, update: MetricsUpdate, *, effective: dict[str, float]) -> None:
        values = self._ledger.setdefault(update.node_id, {key: 0 for key in self._FAMILY_KEYS})
        for key, delta in effective.items():
            if key in values:
                values[key] += delta

    def _check_parity(self, node_id: str) -> None:
        """Count every family key where the comparison ledger and store disagree.

        Tautological after this projector's own commit unless another writer
        touched the store — which is exactly the drift both modes must surface.
        """
        values = self._ledger.setdefault(node_id, {key: 0 for key in self._FAMILY_KEYS})
        for key in self._FAMILY_KEYS:
            if values.get(key, 0) != float(self._store.get_metric(node_id, key) or 0):
                self._parity_mismatches += 1


__all__ = ["RouterStateMode", "RouterStickyStateProjector", "RouterInflightStateProjector"]
