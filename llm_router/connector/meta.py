"""Metadata exchanged between scheduler-side and worker-side connectors.

Mirrors vLLM's reference SharedStorageConnectorMetadata, but stamps the
weight_version on each request so that worker-side load/save can compute
the correct VersionedKey without re-reading scheduler state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from llm_router.connector.prefix_hash import build_prefix_signature_at_len


def _align_to_block_size(num_tokens: int, block_size: int) -> int:
    return (num_tokens // block_size) * block_size


@dataclass
class MooncakeReqMeta:
    request_id: str
    token_ids: list[int]
    block_ids: list[int]
    block_size: int
    is_store: bool
    weight_version: str
    prefix_len: int | None = None
    server_id: str | None = None

    @property
    def aligned_token_count(self) -> int:
        if self.prefix_len is not None:
            return min(int(self.prefix_len), len(self.token_ids))
        return _align_to_block_size(len(self.token_ids), self.block_size)

    @property
    def token_ids_tensor(self) -> torch.Tensor:
        return torch.tensor(
            self.token_ids[: self.aligned_token_count],
            dtype=torch.long,
        )

    @property
    def prefix_signature(self) -> tuple[str, str, int]:
        return build_prefix_signature_at_len(
            self.token_ids,
            self.weight_version,
            self.aligned_token_count,
        )


@dataclass
class MooncakeConnectorMetadata(KVConnectorMetadata):
    requests: list[MooncakeReqMeta] = field(default_factory=list)

    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        weight_version: Any,
        prefix_len: int | None = None,
        server_id: str | None = None,
    ) -> None:
        self.requests.append(
            MooncakeReqMeta(
                request_id=request_id,
                token_ids=list(token_ids),
                block_ids=list(block_ids),
                block_size=block_size,
                is_store=is_store,
                weight_version=str(weight_version) if weight_version is not None else "unknown",
                prefix_len=prefix_len,
                server_id=server_id,
            )
        )
