"""MooncakeKVConnector unit tests using InMemoryKVStore as the backend.

Strategy: We don't drive a full vLLM scheduler — that's covered by
`test_factory_registration.py` (smoke) and out-of-tree integration. Instead
we exercise:
- weight_version is read from kv_transfer_config.kv_connector_extra_config
- get_num_new_matched_tokens returns 0 on miss, full prefix length on hit
- save path serializes payload through KVStore.put
- load path deserializes via KVStore.get
"""
from unittest.mock import MagicMock

import pytest
import torch

from llm_router.connector.connector import MooncakeKVConnector
from llm_router.connector.prefix_hash import make_versioned_key
from llm_router.connector.store.base import TransferResult
from llm_router.connector.store.in_memory import InMemoryKVStore


@pytest.fixture
def store():
    return InMemoryKVStore(max_bytes=1024 * 1024)


@pytest.fixture
def connector(monkeypatch, store):
    """Build a connector wired to InMemoryKVStore."""
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    cfg = MagicMock()
    cfg.cache_config.block_size = 2
    cfg.kv_transfer_config.kv_connector_extra_config = {
        "weight_version": "v0",
        "prefix_probe_stride": 2,
        "server_id": "server-a",
    }
    cfg.kv_transfer_config.engine_id = "engine-test"

    # Bypass the real Mooncake init by injecting our store via a module hook.
    monkeypatch.setattr(
        "llm_router.connector.connector._build_default_store",
        lambda extra: store,
    )
    return MooncakeKVConnector(
        vllm_config=cfg,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=None,
    )


def test_connector_reads_weight_version_from_extra_config(connector):
    assert connector._weight_version == "v0"


def test_default_store_factory_selects_pooled_backend(monkeypatch):
    import llm_router.connector.connector as connector_module

    captured = {}

    class FakeMooncakeStore:
        @classmethod
        def from_env(cls, **kwargs):
            captured.update(kwargs)
            return "pooled-store"

    monkeypatch.setattr(
        "llm_router.connector.store.mooncake.MooncakeKVStore",
        FakeMooncakeStore,
    )

    store = connector_module._build_default_store(
        {
            "mooncake_backend": "pooled",
            "mooncake_master": "127.0.0.1:50051",
            "mooncake_buffer_bytes": 1234,
        }
    )

    assert store == "pooled-store"
    assert captured["backend"] == "pooled"
    assert captured["extra_config"]["mooncake_master"] == "127.0.0.1:50051"
    assert captured["buffer_bytes"] == 1234


def test_get_num_new_matched_tokens_returns_zero_on_miss(connector):
    req = MagicMock()
    req.prompt_token_ids = [1, 2, 3, 4]
    n, async_load = connector.get_num_new_matched_tokens(req, num_computed_tokens=0)
    assert n == 0
    assert async_load is False


def test_get_num_new_matched_tokens_returns_aligned_hit(connector, store):
    # Pre-populate store with a prefix the connector should find.
    key = make_versioned_key([1, 2, 3, 4], prefix_len=4, weight_version="v0")
    store.put(key, b"dummy-kv")

    req = MagicMock()
    req.prompt_token_ids = [1, 2, 3, 4, 5]
    n, async_load = connector.get_num_new_matched_tokens(req, num_computed_tokens=0)
    # The 5-token prompt minus the last token (1 token) = 4 tokens; aligned to block_size=2 → 4.
    assert n == 4
    assert async_load is False


def test_get_num_new_matched_tokens_uses_longest_available_stride_key(connector, store):
    store.put(make_versioned_key([1, 2], prefix_len=2, weight_version="v0"), b"short")
    store.put(make_versioned_key([1, 2, 3, 4], prefix_len=4, weight_version="v0"), b"long")

    req = MagicMock()
    req.request_id = "r-stride"
    req.prompt_token_ids = [1, 2, 3, 4, 5]
    n, async_load = connector.get_num_new_matched_tokens(req, num_computed_tokens=0)

    assert n == 4
    assert async_load is False


def test_different_weight_version_misses(connector, store):
    # Seed with a different version → connector should miss.
    key = make_versioned_key([1, 2, 3, 4], 4, "v1")
    store.put(key, b"old")

    req = MagicMock()
    req.prompt_token_ids = [1, 2, 3, 4, 5]
    n, _ = connector.get_num_new_matched_tokens(req, num_computed_tokens=0)
    assert n == 0


def test_build_connector_meta_carries_matched_prefix_len(connector, store):
    key = make_versioned_key([1, 2, 3, 4], 4, "v0")
    store.put(key, b"dummy-kv")

    req = MagicMock()
    req.request_id = "req-meta"
    req.prompt_token_ids = [1, 2, 3, 4, 5]
    connector.get_num_new_matched_tokens(req, num_computed_tokens=0)

    scheduled = MagicMock()
    scheduled.req_id = "req-meta"
    scheduled.prompt_token_ids = [1, 2, 3, 4, 5]
    scheduled.block_ids = [[0, 1]]
    scheduler_output = MagicMock()
    scheduler_output.scheduled_new_reqs = [scheduled]

    meta = connector.build_connector_meta(scheduler_output)
    assert meta.requests[0].is_store is False
    assert meta.requests[0].aligned_token_count == 4


def test_serialize_and_deserialize_kv_blocks_roundtrip(connector):
    """Worker-side helpers serialize a list of per-layer KV tensors to bytes
    and back. These are private but test them directly for safety."""
    layer_a = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    layer_b = torch.arange(16, 32, dtype=torch.float32).reshape(2, 8)
    payload = connector._serialize_layers([layer_a, layer_b])
    assert isinstance(payload, bytes)
    restored = connector._deserialize_layers(payload, expected_count=2)
    assert len(restored) == 2
    assert torch.equal(restored[0], layer_a)
    assert torch.equal(restored[1], layer_b)


def _populate_kv_caches(connector, num_layers=2, num_blocks=4, block_dim=8):
    """Helper: install fake KV cache tensors into connector for save/load tests."""
    kv_caches = {
        f"layer_{i}": torch.zeros(num_blocks, block_dim, dtype=torch.float32)
        for i in range(num_layers)
    }
    connector.register_kv_caches(kv_caches)
    return kv_caches


def test_save_then_load_roundtrip_across_multiple_layers(connector, store, monkeypatch):
    """Bug 1 regression: concatenated per-layer pickles broke load. This test
    drives the actual save -> load path end-to-end through the store."""
    from llm_router.connector.meta import MooncakeReqMeta

    kv = _populate_kv_caches(connector)
    save_req = MooncakeReqMeta(
        request_id="r1",
        token_ids=[1, 2, 3, 4],
        block_ids=[0, 1],
        block_size=2,
        is_store=True,
        weight_version="v0",
    )

    # Pre-load fake KV with distinguishable values so save captures real bytes.
    kv["layer_0"][0] = torch.full((8,), 11.0)
    kv["layer_0"][1] = torch.full((8,), 12.0)
    kv["layer_1"][0] = torch.full((8,), 21.0)
    kv["layer_1"][1] = torch.full((8,), 22.0)

    # Drive _save_request_layer once per layer, like vLLM does.
    connector._save_request_layer(save_req, "layer_0", kv["layer_0"])
    connector._save_request_layer(save_req, "layer_1", kv["layer_1"])

    # Now zero out the local caches and load back via the store.
    for t in kv.values():
        t.zero_()
    load_req = MooncakeReqMeta(
        request_id="r2",
        token_ids=[1, 2, 3, 4],
        block_ids=[0, 1],
        block_size=2,
        is_store=False,
        weight_version="v0",
    )
    connector._load_request(load_req)

    assert torch.allclose(kv["layer_0"][0], torch.full((8,), 11.0))
    assert torch.allclose(kv["layer_0"][1], torch.full((8,), 12.0))
    assert torch.allclose(kv["layer_1"][0], torch.full((8,), 21.0))
    assert torch.allclose(kv["layer_1"][1], torch.full((8,), 22.0))


def test_save_two_requests_in_one_step_no_crosstalk(connector, store):
    """Bug 2 regression: shared pending buffer mixed two requests' bytes."""
    from llm_router.connector.meta import MooncakeReqMeta

    kv = _populate_kv_caches(connector)

    # Two distinct prompts, both in the same step.
    req_a = MooncakeReqMeta(
        request_id="a",
        token_ids=[1, 2, 3, 4],
        block_ids=[0, 1],
        block_size=2,
        is_store=True,
        weight_version="v0",
    )
    req_b = MooncakeReqMeta(
        request_id="b",
        token_ids=[5, 6, 7, 8],
        block_ids=[2, 3],
        block_size=2,
        is_store=True,
        weight_version="v0",
    )
    kv["layer_0"][0] = torch.full((8,), 1.0)
    kv["layer_0"][1] = torch.full((8,), 2.0)
    kv["layer_0"][2] = torch.full((8,), 3.0)
    kv["layer_0"][3] = torch.full((8,), 4.0)
    kv["layer_1"][0] = torch.full((8,), 10.0)
    kv["layer_1"][1] = torch.full((8,), 20.0)
    kv["layer_1"][2] = torch.full((8,), 30.0)
    kv["layer_1"][3] = torch.full((8,), 40.0)

    # Interleave layer saves like vLLM might (req A layer 0, req B layer 0,
    # req A layer 1, req B layer 1).
    connector._save_request_layer(req_a, "layer_0", kv["layer_0"])
    connector._save_request_layer(req_b, "layer_0", kv["layer_0"])
    connector._save_request_layer(req_a, "layer_1", kv["layer_1"])
    connector._save_request_layer(req_b, "layer_1", kv["layer_1"])

    # Both keys exist in store independently.
    from llm_router.connector.prefix_hash import make_versioned_key

    assert store.contains(make_versioned_key([1, 2, 3, 4], 4, "v0"))
    assert store.contains(make_versioned_key([5, 6, 7, 8], 4, "v0"))


def test_save_stores_stride_aligned_prefix_keys(connector, store):
    from llm_router.connector.meta import MooncakeReqMeta

    kv = _populate_kv_caches(connector, num_layers=2, num_blocks=4)
    for i in range(4):
        kv["layer_0"][i] = torch.full((8,), float(i + 1))
        kv["layer_1"][i] = torch.full((8,), float(10 + i + 1))

    req = MooncakeReqMeta(
        request_id="stride-save",
        token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        block_ids=[0, 1, 2, 3],
        block_size=2,
        is_store=True,
        weight_version="v0",
        server_id="server-a",
    )
    connector._save_request_layer(req, "layer_0", kv["layer_0"])
    connector._save_request_layer(req, "layer_1", kv["layer_1"])

    for prefix_len in [2, 4, 6, 8]:
        assert store.contains(make_versioned_key(req.token_ids, prefix_len, "v0"))
    assert connector._prefix_reporter.reports[-1][0] == "server-a"
    assert [sig[2] for sig in connector._prefix_reporter.reports[-1][1]] == [2, 4, 6, 8]
    assert connector._prefix_reporter.tiered_reports[-1][0] == "cpu"


def test_load_with_corrupted_payload_records_block_errors(connector, store):
    """Bug 3 regression: malformed payload should not crash worker."""
    from llm_router.connector.meta import MooncakeReqMeta
    from llm_router.connector.prefix_hash import make_versioned_key

    _populate_kv_caches(connector)
    bad_key = make_versioned_key([1, 2, 3, 4], 4, "v0")
    # Insert bytes that torch.load will refuse.
    store.put(bad_key, b"not-a-pickle")

    req = MooncakeReqMeta(
        request_id="r",
        token_ids=[1, 2, 3, 4],
        block_ids=[0, 1],
        block_size=2,
        is_store=False,
        weight_version="v0",
    )
    # Must not raise.
    connector._load_request(req)
    errors = connector.get_block_ids_with_load_errors()
    assert errors == {0, 1}
    # Subsequent call returns empty (errors are consumed).
    assert connector.get_block_ids_with_load_errors() == set()


class DelayedStore(InMemoryKVStore):
    def __init__(self):
        super().__init__(max_bytes=1024 * 1024)
        self.pending: dict[str, object] = {}
        self.started = 0

    def begin_get(self, key):
        self.started += 1
        transfer_id = f"t-{self.started}"
        self.pending[transfer_id] = key
        return transfer_id

    def poll_get(self, transfer_id: str):
        value = self.pending.get(transfer_id)
        if value == "pending":
            return TransferResult.pending()
        if value is None:
            return TransferResult.failed("unknown transfer_id")
        key = self.pending.pop(transfer_id)
        return TransferResult.done(self.get(key))


def test_start_load_kv_defers_copy_until_transfer_finishes(connector, monkeypatch):
    from llm_router.connector.meta import MooncakeConnectorMetadata

    delayed = DelayedStore()
    monkeypatch.setattr(connector, "_store", delayed)
    kv = _populate_kv_caches(connector)
    kv["layer_0"][0] = torch.full((8,), 11.0)
    kv["layer_1"][0] = torch.full((8,), 21.0)
    payload = connector._serialize_layers(
        [torch.stack([kv["layer_0"][0]]), torch.stack([kv["layer_1"][0]])]
    )
    delayed.put(make_versioned_key([1, 2], 2, "v0"), payload)
    for t in kv.values():
        t.zero_()

    meta = MooncakeConnectorMetadata()
    meta.add_request(
        request_id="r-async",
        token_ids=[1, 2],
        block_ids=[0],
        block_size=2,
        is_store=False,
        weight_version="v0",
    )
    connector.bind_connector_metadata(meta)
    # Force first poll to stay pending.
    original_poll = delayed.poll_get

    def pending_once(transfer_id):
        delayed.pending[transfer_id] = "pending"
        delayed.poll_get = original_poll
        return TransferResult.pending()

    delayed.poll_get = pending_once
    connector.start_load_kv(MagicMock())
    assert torch.equal(kv["layer_0"][0], torch.zeros(8))
    assert connector.get_finished({"r-async"}) == (set(), set())

    # Restore the transfer key and complete.
    delayed.pending["t-1"] = make_versioned_key([1, 2], 2, "v0")
    sending, loading = connector.get_finished({"r-async"})
    assert sending == set()
    assert loading == {"r-async"}
    connector.wait_for_layer_load("layer_0")
    assert torch.allclose(kv["layer_0"][0], torch.full((8,), 11.0))
    assert torch.allclose(kv["layer_1"][0], torch.full((8,), 21.0))
    assert connector._prefix_reporter.reports[-1][0] == "server-a"
    assert connector._prefix_reporter.reports[-1][1][0][2] == 2
    assert connector._prefix_reporter.tiered_reports[-1][0] == "gpu"
