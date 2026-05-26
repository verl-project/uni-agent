"""RouterPolicy abstract base contract tests."""
import pytest

from llm_router.policy.base import RouterPolicy


def test_router_policy_is_abstract():
    """RouterPolicy 是抽象类,不能直接实例化。"""
    with pytest.raises(TypeError):
        RouterPolicy(server_ids=["a", "b"])


def test_router_policy_subclass_must_implement_acquire_release_and_report():
    """子类必须实现 acquire_server / release_server / report_prefixes。"""

    class MissingReport(RouterPolicy):
        def acquire_server(self, request_id, **_):
            return "a"

        def release_server(self, server_id):
            pass

    with pytest.raises(TypeError):
        MissingReport(server_ids=["a", "b"])


def test_router_policy_subclass_with_methods_works():
    """实现了三个方法的子类可正常实例化。"""

    class Minimal(RouterPolicy):
        def acquire_server(self, request_id, **_):
            return self.server_ids[0]

        def release_server(self, server_id):
            pass

        def report_prefixes(self, server_id, prefix_signatures):
            pass

    policy = Minimal(server_ids=["a", "b"])
    assert policy.acquire_server("req-1") == "a"
    policy.release_server("a")
    policy.report_prefixes("a", [("v0", "deadbeef", 1024)])
