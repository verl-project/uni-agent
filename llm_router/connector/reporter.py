"""Optional connector-side reporting of actual KV store state to the LB."""
from __future__ import annotations

from typing import Any


class PrefixReporter:
    """Best-effort reporter for connector-observed prefix availability."""

    def __init__(
        self,
        lb_handle: Any = None,
        server_id: str | None = None,
        lb_actor_name: str | None = None,
        lb_actor_namespace: str | None = None,
    ):
        self._lb_handle = lb_handle
        self._server_id = server_id
        self._lb_actor_name = lb_actor_name
        self._lb_actor_namespace = lb_actor_namespace
        self.reports: list[tuple[str, list[tuple[str, str, int]]]] = []
        self.tiered_reports: list[
            tuple[str, str, list[tuple[str, str, int]]]
        ] = []

    @classmethod
    def from_extra_config(cls, extra_config: dict[str, Any]) -> PrefixReporter:
        return cls(
            lb_handle=extra_config.get("load_balancer_handle"),
            server_id=extra_config.get("server_id"),
            lb_actor_name=extra_config.get("load_balancer_actor_name"),
            lb_actor_namespace=extra_config.get("load_balancer_actor_namespace"),
        )

    def report(
        self,
        server_id: str | None,
        prefix_signatures: list[tuple[str, str, int]],
        *,
        tier: str = "gpu",
    ) -> None:
        resolved_server_id = server_id or self._server_id
        if not resolved_server_id or not prefix_signatures:
            return
        sigs = [(str(v), str(h), int(n)) for v, h, n in prefix_signatures]
        self.reports.append((resolved_server_id, sigs))
        self.tiered_reports.append((str(tier), resolved_server_id, sigs))
        lb_handle = self._resolve_load_balancer()
        if lb_handle is None:
            return
        try:
            lb_handle.report_prefixes.remote(resolved_server_id, sigs, tier=tier)
        except TypeError:
            try:
                lb_handle.report_prefixes.remote(resolved_server_id, sigs)
            except Exception:
                return
        except Exception:
            # Routing already has worker/prewarm reports. Connector reports are
            # best-effort hints and must not fail generation.
            return

    def _resolve_load_balancer(self) -> Any:
        if self._lb_handle is not None:
            return self._lb_handle
        if not self._lb_actor_name:
            return None
        try:
            import ray

            if self._lb_actor_namespace:
                self._lb_handle = ray.get_actor(
                    self._lb_actor_name,
                    namespace=self._lb_actor_namespace,
                )
            else:
                self._lb_handle = ray.get_actor(self._lb_actor_name)
        except Exception:
            return None
        return self._lb_handle
