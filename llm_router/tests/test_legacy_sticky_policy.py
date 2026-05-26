"""LegacyStickyPolicy: byte-for-byte parity with verl GlobalRequestLoadBalancer._acquire_sticky_server."""
import pytest

from llm_router.policy.legacy_sticky import LegacyStickyPolicy


def test_first_request_goes_to_least_loaded():
    p = LegacyStickyPolicy(server_ids=["a", "b", "c"])
    assert p.acquire_server("req-1") == "a"  # 全 0，按字典序最小


def test_same_request_id_sticks_to_same_server():
    p = LegacyStickyPolicy(server_ids=["a", "b", "c"])
    assert p.acquire_server("req-1") == "a"
    p.release_server("a")
    assert p.acquire_server("req-1") == "a"  # sticky binding


def test_different_request_ids_distribute():
    p = LegacyStickyPolicy(server_ids=["a", "b", "c"])
    s1 = p.acquire_server("req-1")  # a (count=1)
    s2 = p.acquire_server("req-2")  # b (a=1, b=0, c=0 → b 最小)
    s3 = p.acquire_server("req-3")  # c
    assert s1 == "a"
    assert s2 == "b"
    assert s3 == "c"


def test_lru_eviction_resets_binding():
    p = LegacyStickyPolicy(server_ids=["a", "b"], max_cache_size=2)
    s1 = p.acquire_server("req-1")
    p.release_server(s1)
    s2 = p.acquire_server("req-2")
    p.release_server(s2)
    s3 = p.acquire_server("req-3")  # 把 req-1 挤出 LRU
    p.release_server(s3)
    # req-1 的绑定已被 LRU 淘汰
    assert "req-1" not in p._cache


def test_release_unknown_server_raises():
    p = LegacyStickyPolicy(server_ids=["a"])
    with pytest.raises(ValueError):
        p.release_server("nonexistent")


def test_release_without_acquire_raises():
    p = LegacyStickyPolicy(server_ids=["a"])
    with pytest.raises(ValueError):
        p.release_server("a")  # 计数为 0


def test_report_prefixes_is_noop_for_legacy():
    """Legacy 不做 prefix 感知,report_prefixes 必须不抛、不影响后续路由。"""
    p = LegacyStickyPolicy(server_ids=["a", "b"])
    p.report_prefixes("a", [("v0", "deadbeef", 1024)])
    # Routing unchanged.
    assert p.acquire_server("req-1") == "a"


def test_report_prefixes_validates_unknown_server():
    """Legacy 的 no-op 也应该校验 server_id 存在,避免静默配置错误。"""
    import pytest
    p = LegacyStickyPolicy(server_ids=["a"])
    with pytest.raises(ValueError, match="Invalid server_id"):
        p.report_prefixes("nonexistent", [])
