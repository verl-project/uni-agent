import asyncio

import pytest

from examples.multi_policy.framework import MultiAgentGatewayManager
from uni_agent.gateway import gateway as gateway_module
from uni_agent.gateway import manager as gateway_manager_module
from uni_agent.gateway.session import SessionHandle, Trajectory


class _RemoteMethod:
    def __init__(self, fn=None):
        self._fn = fn
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        async def run():
            if self._fn is None:
                return None
            return self._fn(*args, **kwargs)

        return run()


class _FakeActor:
    """Records session lifecycle calls; create returns a per-session URL."""

    def __init__(self):
        self.start = _RemoteMethod()
        self.shutdown = _RemoteMethod()
        self.create_session = _RemoteMethod(
            lambda session_id, **kwargs: SessionHandle(
                session_id=session_id,
                base_url=f"http://gateway/sessions/{session_id}/v1",
            )
        )
        self.finalize_session = _RemoteMethod(lambda session_id, **kwargs: [Trajectory([1], [2], [1])])
        self.abort_session = _RemoteMethod()

    @classmethod
    def options(cls, **kwargs):
        return cls

    @classmethod
    def remote(cls, *args, **kwargs):
        return cls()


@pytest.fixture
def fake_ray(monkeypatch):
    # The manager picks the actor class from the argument types, so patching
    # GatewayActor also checks that policy mappings select it.
    monkeypatch.setattr(gateway_module, "GatewayActor", _FakeActor)
    monkeypatch.setattr(
        gateway_manager_module.ray,
        "nodes",
        lambda: [{"NodeID": "0" * 56, "Alive": True, "Resources": {"CPU": 1}}],
    )

    # The manager calls ray.get synchronously inside its constructor, which
    # these tests run on an existing event loop. Close the fake start coroutines
    # because this synchronous stub does not schedule or await them.
    def get(refs):
        for ref in refs:
            if asyncio.iscoroutine(ref):
                ref.close()
        return [ref for ref in refs]

    monkeypatch.setattr(gateway_manager_module.ray, "get", get)


def _manager(**kwargs):
    return MultiAgentGatewayManager(
        llm_client={"p1": object(), "p2": object()},
        gateway_count=kwargs.get("gateway_count", 2),
        gateway_actor_config={"p1": object(), "p2": object()},
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_constructor_owns_actor_creation_and_shutdown_clears_state(fake_ray):
    manager = _manager()
    assert len(manager.gateways) == 2
    assert all(isinstance(gateway, _FakeActor) for gateway in manager.gateways)
    assert manager.gateway_count == 2
    assert manager.active_sessions_per_gateway == [0, 0]

    await manager.shutdown()
    assert manager.gateways == []
    assert manager.gateway_count == 0
    assert manager.active_sessions_per_gateway == []
    assert manager._session_to_gateway_index == {}
    await manager.shutdown()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("llm_client", "gateway_actor_config"),
    [({"p1": object()}, object()), (object(), {"p1": object()})],
)
def test_constructor_rejects_mixed_single_and_policy_mapping(fake_ray, llm_client, gateway_actor_config):
    with pytest.raises(ValueError, match="both be policy mappings"):
        MultiAgentGatewayManager(
            llm_client=llm_client,
            gateway_count=1,
            gateway_actor_config=gateway_actor_config,
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_create_group_returns_per_role_urls_and_counts_one_slot(fake_ray):
    manager = _manager(gateway_count=1)
    handle = await manager.create_session(
        "r",
        role_policy_mapping={"planner": "p1", "answerer": "p2"},
        sampling_params={"p1": {"temperature": 0.7}, "p2": {"temperature": 1.0}},
        metadata={"global_steps": 3},
    )
    assert handle.session_group_id == "r"
    assert not hasattr(handle, "base_url")
    assert set(handle.role_base_urls) == {"planner", "answerer"}
    assert handle.role_base_urls["planner"] != handle.role_base_urls["answerer"]
    assert handle.role_base_urls["planner"].endswith("/sessions/r:planner/v1")
    # One rollout occupies one scheduling slot, not one per role.
    assert manager.active_sessions_per_gateway == [1]
    actor = manager.gateways[0]
    assert [kwargs["sampling_params"] for _, kwargs in actor.create_session.calls] == [
        {"temperature": 0.7},
        {"temperature": 1.0},
    ]
    assert actor.create_session.calls[0][1]["policy_name"] == "p1"
    assert actor.create_session.calls[0][1]["metadata"]["role"] == "planner"
    await manager.shutdown()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_create_group_failure_aborts_created_sessions_and_releases_slot(fake_ray):
    manager = _manager(gateway_count=1)
    actor = manager.gateways[0]
    calls = {"count": 0}

    def create(session_id, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("create failed")
        return SessionHandle(session_id=session_id, base_url="http://gateway/x/v1")

    actor.create_session = _RemoteMethod(create)
    with pytest.raises(RuntimeError, match="create failed"):
        await manager.create_session("r", role_policy_mapping={"a": "p1", "b": "p2"})
    assert [kwargs["session_id"] for _, kwargs in actor.abort_session.calls] == [
        "r:a",
        "r:b",
    ]
    assert manager.active_sessions_per_gateway == [0]
    assert "r" not in manager._group_sessions
    assert "r" not in manager._group_gateway_indices
    await manager.shutdown()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_finalize_group_aggregates_and_annotates_each_role(fake_ray):
    manager = _manager(gateway_count=1)
    await manager.create_session("r", role_policy_mapping={"planner": "p1", "answerer": "p2"})
    trajectories = await manager.finalize_session("r")
    assert len(trajectories) == 2
    assert {(t.extra_fields["role"], t.extra_fields["policy_name"]) for t in trajectories} == {
        ("planner", "p1"),
        ("answerer", "p2"),
    }
    assert all(t.extra_fields["session_group_id"] == "r" for t in trajectories)
    assert manager.active_sessions_per_gateway == [0]
    assert "r" not in manager._group_sessions
    await manager.shutdown()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_finalize_failure_aborts_remaining_sessions_and_drops_state(fake_ray):
    manager = _manager(gateway_count=1)
    await manager.create_session("r", role_policy_mapping={"planner": "p1", "answerer": "p2"})
    actor = manager.gateways[0]
    calls = {"count": 0}

    def finalize(session_id, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("finalize failed")
        return [Trajectory([1], [2], [1])]

    actor.finalize_session = _RemoteMethod(finalize)
    with pytest.raises(RuntimeError, match="finalize failed"):
        await manager.finalize_session("r")
    assert len(actor.abort_session.calls) == 2
    assert manager.active_sessions_per_gateway == [0]
    assert "r" not in manager._group_sessions
    await manager.shutdown()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_abort_group_is_idempotent(fake_ray):
    manager = _manager(gateway_count=1)
    await manager.create_session("r", role_policy_mapping={"a": "p1"})
    await manager.abort_session("r")
    await manager.abort_session("r")
    assert manager.active_sessions_per_gateway == [0]
    assert "r" not in manager._group_sessions
    await manager.shutdown()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_closed_manager_rejects_new_group(fake_ray):
    manager = _manager(gateway_count=1)
    await manager.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        await manager.create_session("r", role_policy_mapping={"a": "p1"})


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_duplicate_group_is_rejected(fake_ray):
    manager = _manager(gateway_count=1)
    await manager.create_session("r", role_policy_mapping={"a": "p1"})
    with pytest.raises(ValueError, match="Duplicate session group"):
        await manager.create_session("r", role_policy_mapping={"a": "p1"})
    await manager.shutdown()
