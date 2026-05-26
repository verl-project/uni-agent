"""KVConnectorMetadata subclass + per-request meta entries."""
import torch

from llm_router.connector.meta import (
    MooncakeConnectorMetadata,
    MooncakeReqMeta,
)


def test_reqmeta_construction():
    m = MooncakeReqMeta(
        request_id="req-1",
        token_ids=[1, 2, 3, 4],
        block_ids=[0, 1],
        block_size=2,
        is_store=False,
        weight_version="v0",
    )
    assert m.request_id == "req-1"
    assert m.is_store is False
    assert m.weight_version == "v0"
    assert m.token_ids_tensor.dtype == torch.long
    assert m.token_ids_tensor.tolist() == [1, 2, 3, 4]


def test_metadata_add_and_iterate():
    meta = MooncakeConnectorMetadata()
    meta.add_request(
        request_id="req-1",
        token_ids=[1, 2],
        block_ids=[0],
        block_size=2,
        is_store=False,
        weight_version="v0",
    )
    meta.add_request(
        request_id="req-2",
        token_ids=[1, 2, 3, 4],
        block_ids=[0, 1],
        block_size=2,
        is_store=True,
        weight_version="v0",
    )
    assert len(meta.requests) == 2
    assert [r.request_id for r in meta.requests] == ["req-1", "req-2"]
    assert [r.is_store for r in meta.requests] == [False, True]


def test_metadata_aligns_token_ids_to_block_size():
    meta = MooncakeConnectorMetadata()
    meta.add_request(
        request_id="req-3",
        token_ids=[1, 2, 3, 4, 5],  # 5 tokens, block_size=2 → align to 4
        block_ids=[0, 1],
        block_size=2,
        is_store=False,
        weight_version="v0",
    )
    r = meta.requests[0]
    assert r.aligned_token_count == 4
    assert r.token_ids_tensor.shape == (4,)


def test_metadata_prefix_len_overrides_block_alignment_for_loads():
    meta = MooncakeConnectorMetadata()
    meta.add_request(
        request_id="req-4",
        token_ids=[1, 2, 3, 4, 5, 6],
        block_ids=[0, 1, 2],
        block_size=4,
        is_store=False,
        weight_version="v0",
        prefix_len=6,
    )
    r = meta.requests[0]
    assert r.aligned_token_count == 6
    assert r.token_ids_tensor.tolist() == [1, 2, 3, 4, 5, 6]


def test_reqmeta_prefix_signature_and_server_id():
    meta = MooncakeConnectorMetadata()
    meta.add_request(
        request_id="req-5",
        token_ids=[1, 2, 3, 4],
        block_ids=[0, 1],
        block_size=2,
        is_store=False,
        weight_version="v0",
        prefix_len=2,
        server_id="server-a",
    )
    r = meta.requests[0]
    assert r.server_id == "server-a"
    assert r.prefix_signature[0] == "v0"
    assert r.prefix_signature[2] == 2
