"""MooncakeKVConnector: vLLM v1 KV connector backed by a versioned KV pool.

Implements the data flow described in RFC §5.1:
  1. On scheduler-side `get_num_new_matched_tokens`, look up
     `(hash(prompt_prefix), weight_version)` in the KVStore.
  2. On worker-side `start_load_kv`, fetch the payload and copy KV bytes
     into the PagedAttention buffer at the slot mapping the scheduler
     allocated.
  3. On worker-side `save_kv_layer` (called per layer when a request finishes),
     serialize the layer's KV slice and write it to the KVStore.

Plan B keeps the connector minimal: we adopt the structure of vLLM's
SharedStorageConnector (a shipping reference implementation) and substitute
its "save to disk path" for "save to KVStore.put". Methods that vLLM's
connector framework calls but Plan B does not need (HMA hooks, KV events,
stats) are stubs marked TODO(plan-b-followup).
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

from llm_router.connector.meta import MooncakeConnectorMetadata, MooncakeReqMeta
from llm_router.connector.prefix_hash import (
    coerce_weight_version,
    iter_prefix_signatures,
    make_versioned_key,
)
from llm_router.connector.reporter import PrefixReporter
from llm_router.connector.store.base import KVStore

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


def _build_default_store(extra_config: dict[str, Any]) -> KVStore:
    """Default factory: build a MooncakeKVStore from extra_config.

    Tests monkeypatch this symbol to inject InMemoryKVStore.
    """
    from llm_router.connector.store.mooncake import MooncakeKVStore

    buffer_bytes = int(extra_config.get("mooncake_buffer_bytes", 32 * 1024 * 1024 * 1024))
    backend = str(extra_config.get("mooncake_backend", "auto"))
    device_name = str(extra_config.get("mooncake_device_name", ""))
    return MooncakeKVStore.from_env(
        buffer_bytes=buffer_bytes,
        device_name=device_name,
        backend=backend,
        extra_config=extra_config,
    )


class MooncakeKVConnector(KVConnectorBase_V1):
    """vLLM v1 connector that load/saves KV from/to a versioned KVStore."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional[KVCacheConfig] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self._weight_version = coerce_weight_version(extra.get("weight_version"))
        self._prefix_probe_stride = int(extra.get("prefix_probe_stride", self._block_size))
        self._server_id = extra.get("server_id")
        self._store: KVStore = _build_default_store(extra)
        self._prefix_reporter = PrefixReporter.from_extra_config(extra)
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._requests_need_load: dict[str, Request] = {}
        # Per-request layer buffers: request_id -> layer_name -> stacked block tensor.
        # Flushed as one multi-layer pickle when all layers have arrived.
        self._pending_saves: dict[str, dict[str, torch.Tensor]] = {}
        self._save_meta: dict[str, MooncakeReqMeta] = {}
        self._pending_loads: dict[str, tuple[MooncakeReqMeta, str]] = {}
        self._completed_loads: dict[str, tuple[MooncakeReqMeta, bytes | None]] = {}
        self._load_ready: set[str] = set()
        # Block ids of requests whose load failed; drained by
        # get_block_ids_with_load_errors so vLLM can recompute them.
        self._load_errors: set[int] = set()

    # ---- Scheduler-side ----

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        token_ids = request.prompt_token_ids or []
        if len(token_ids) <= 1:
            return 0, False
        aligned = self._best_available_prefix_len(token_ids)
        if aligned <= num_computed_tokens:
            return 0, False
        self._requests_need_load[request.request_id] = request
        return aligned - num_computed_tokens, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            block_ids = list(new_req.block_ids[0]) if new_req.block_ids else []
            is_load = new_req.req_id in self._requests_need_load
            prefix_len = self._best_available_prefix_len(token_ids) if is_load else None
            meta.add_request(
                request_id=new_req.req_id,
                token_ids=token_ids,
                block_ids=block_ids,
                block_size=self._block_size,
                is_store=not is_load,  # if no external hit, this turn we save
                weight_version=self._weight_version,
                prefix_len=prefix_len,
                server_id=self._server_id,
            )
        # Flush per-step state.
        self._requests_need_load.clear()
        return meta

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # Plan B: keep blocks owned by vLLM (return False), no async transfer params.
        return False, None

    # ---- Worker-side ----

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, MooncakeConnectorMetadata):
            return
        for req in meta.requests:
            if req.is_store:
                continue
            self._begin_load_request(req)

    def wait_for_layer_load(self, layer_name: str) -> None:
        self._drain_ready_loads()
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, MooncakeConnectorMetadata):
            return
        for req in meta.requests:
            if not req.is_store:
                continue
            self._save_request_layer(req, layer_name, kv_layer)

    def wait_for_save(self) -> None:
        # Plan B: save is synchronous in _save_request_layer.
        return

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        self._poll_pending_loads()
        ready = {rid for rid in finished_req_ids if rid in self._load_ready}
        self._load_ready.difference_update(ready)
        return set(), ready

    def get_block_ids_with_load_errors(self) -> set[int]:
        errors = self._load_errors
        self._load_errors = set()
        return errors

    # ---- Private helpers ----

    def _serialize_layers(self, layers: list[torch.Tensor]) -> bytes:
        """Concatenate per-layer KV tensors into a single bytes payload."""
        buf = io.BytesIO()
        torch.save([t.cpu().contiguous() for t in layers], buf)
        return buf.getvalue()

    def _deserialize_layers(
        self, payload: bytes, expected_count: int
    ) -> list[torch.Tensor]:
        buf = io.BytesIO(payload)
        layers = torch.load(buf, weights_only=True)
        if not isinstance(layers, list) or len(layers) != expected_count:
            raise ValueError(
                f"Mooncake payload contains {len(layers) if isinstance(layers, list) else '?'} "
                f"layers, expected {expected_count}"
            )
        return layers

    def _best_available_prefix_len(self, token_ids: list[int]) -> int:
        """Return the longest store key compatible with this request."""
        # vLLM scheduler expects external matches to be block-aligned and to
        # exclude the token currently being extended.
        max_len = (len(token_ids) - 1) // self._block_size * self._block_size
        if max_len <= 0:
            return 0
        candidates = [
            prefix_len
            for _, _, prefix_len in iter_prefix_signatures(
                token_ids[:max_len],
                self._weight_version,
                stride=self._prefix_probe_stride,
            )
            if prefix_len <= max_len and prefix_len % self._block_size == 0
        ]
        # Always include the scheduler-aligned full prefix, even if future
        # stride settings change.
        if max_len not in candidates:
            candidates.append(max_len)
        for prefix_len in sorted(set(candidates), reverse=True):
            key = make_versioned_key(token_ids, prefix_len, self._weight_version)
            if self._store.contains(key):
                return prefix_len
        return 0

    def _begin_load_request(self, req: MooncakeReqMeta) -> None:
        if req.aligned_token_count <= 0:
            return
        key = make_versioned_key(
            req.token_ids, req.aligned_token_count, req.weight_version
        )
        transfer_id = self._store.begin_get(key)
        self._pending_loads[req.request_id] = (req, transfer_id)
        self._poll_pending_loads()

    def _poll_pending_loads(self) -> None:
        for request_id, (req, transfer_id) in list(self._pending_loads.items()):
            result = self._store.poll_get(transfer_id)
            if result.state == "pending":
                continue
            del self._pending_loads[request_id]
            if result.state == "failed":
                self._load_errors.update(req.block_ids)
                self._load_ready.add(request_id)
                continue
            self._completed_loads[request_id] = (req, result.payload)
            self._load_ready.add(request_id)

    def _drain_ready_loads(self) -> None:
        self._poll_pending_loads()
        for request_id, (req, payload) in list(self._completed_loads.items()):
            self._complete_load_request(req, payload)
            del self._completed_loads[request_id]

    def _complete_load_request(self, req: MooncakeReqMeta, payload: bytes | None) -> None:
        if payload is None:
            return
        try:
            layers = self._deserialize_layers(
                payload, expected_count=len(self._kv_caches)
            )
        except Exception:
            self._load_errors.update(req.block_ids)
            return
        for (layer_name, kv_tensor), src in zip(
            self._kv_caches.items(), layers, strict=False
        ):
            num_blocks = min(len(req.block_ids), src.shape[0])
            for i in range(num_blocks):
                kv_tensor[req.block_ids[i]].copy_(src[i])
        self._prefix_reporter.report(req.server_id, [req.prefix_signature], tier="gpu")

    def _load_request(self, req: MooncakeReqMeta) -> None:
        if req.aligned_token_count <= 0:
            return
        key = make_versioned_key(
            req.token_ids, req.aligned_token_count, req.weight_version
        )
        payload = self._store.get(key)
        if payload is None:
            return
        self._complete_load_request(req, payload)

    def _save_request_layer(
        self, req: MooncakeReqMeta, layer_name: str, kv_layer: torch.Tensor
    ) -> None:
        if req.aligned_token_count <= 0 or not req.block_ids:
            return
        blocks = torch.stack([kv_layer[bid].cpu() for bid in req.block_ids], dim=0)
        layers = self._pending_saves.setdefault(req.request_id, {})
        layers[layer_name] = blocks
        self._save_meta[req.request_id] = req
        if len(layers) < len(self._kv_caches):
            return
        # All layers buffered — emit stride-aligned multi-layer pickles keyed
        # by req. This keeps store keys aligned with worker/prewarm routing
        # reports, so a routed prefix hit can actually find a payload.
        ordered_layers = [layers[name] for name in self._kv_caches.keys()]
        stored_signatures_by_location: dict[str, list[tuple[str, str, int]]] = {}
        for prefix_len in self._store_prefix_lengths(req):
            block_count = prefix_len // self._block_size
            payload = self._serialize_layers([layer[:block_count] for layer in ordered_layers])
            key = make_versioned_key(req.token_ids, prefix_len, req.weight_version)
            self._store.put(key, payload)
            signature = (key.weight_version, key.prefix_hash, key.prefix_len)
            for location in self._store.cpu_locations(
                key,
                local_server_id=req.server_id or self._server_id,
            ):
                stored_signatures_by_location.setdefault(location, []).append(signature)
        for location, signatures in stored_signatures_by_location.items():
            self._prefix_reporter.report(location, signatures, tier="cpu")
        del self._pending_saves[req.request_id]
        del self._save_meta[req.request_id]

    def _store_prefix_lengths(self, req: MooncakeReqMeta) -> list[int]:
        aligned = req.aligned_token_count
        if aligned <= 0:
            return []
        lengths = {
            prefix_len
            for _, _, prefix_len in iter_prefix_signatures(
                req.token_ids[:aligned],
                req.weight_version,
                stride=self._prefix_probe_stride,
            )
            if prefix_len <= aligned and prefix_len % self._block_size == 0
        }
        lengths.add(aligned)
        max_blocks = len(req.block_ids)
        return sorted(length for length in lengths if 0 < length // self._block_size <= max_blocks)
