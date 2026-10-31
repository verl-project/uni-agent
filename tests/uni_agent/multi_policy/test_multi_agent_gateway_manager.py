"""Tests for examples.multi_policy.framework gateway manager."""

import asyncio

import pytest

from examples.multi_policy.framework import MultiAgentGatewayManager
from examples.multi_policy.framework.multi_agent_gateway_manager import _RoleSession
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
        self.create_session = _RemoteMethod(
            lambda session_id, **kwargs: SessionHandle(session_id=session_id, base_url=f"http://fake/{session_id}/v1")
        )
        self.finalize_session = _RemoteMethod(lambda session_id, **kwargs: [Trajectory([1], [2], [1], extra_fields={})])
        self.abort_session = _RemoteMethod()


@pytest.fixture
def fake_gateway_pool(monkeypatch):
    actors = [_FakeActor(), _FakeActor()]
    monkeypatch.setattr(gateway_module, "GatewayActor", lambda *_, **__: None)
    monkeypatch.setattr(gateway_manager_module.ray, "get", lambda _: None)
    return actors


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_create_session_returns_handle_with_role_base_urls(fake_gateway_pool, monkeypatch):
    manager = MultiAgentGatewayManager.__new__(MultiAgentGatewayManager)
    manager.gateways = fake_gateway_pool
    manager.active_sessions_per_gateway = [0, 0]
    manager._group_sessions = {}
    manager._group_gateway_indices = {}

    handle = await manager.create_session(
        "group-1",
        role_policy_mapping={"planner": "policy_a", "actor": "policy_b"},
        sampling_params={
            "policy_a": {"temperature": 0.7},
            "policy_b": {"temperature": 0.9},
        },
        metadata={"team": "alpha"},
    )

    assert handle.session_group_id == "group-1"
    assert not hasattr(handle, "base_url")
    assert handle.role_base_urls == {
        "planner": "http://fake/group-1:planner/v1",
        "actor": "http://fake/group-1:actor/v1",
    }

    # Both roles created on the same gateway
    gateway = fake_gateway_pool[0]
    assert len(gateway.create_session.calls) == 2
    planner_call = gateway.create_session.calls[0]
    assert planner_call[1]["session_id"] == "group-1:planner"
    assert planner_call[1]["policy_name"] == "policy_a"
    assert planner_call[1]["metadata"]["role"] == "planner"
    assert planner_call[1]["metadata"]["policy_name"] == "policy_a"
    assert planner_call[1]["metadata"]["session_group_id"] == "group-1"
    assert planner_call[1]["metadata"]["team"] == "alpha"
    assert planner_call[1]["sampling_params"] == {"temperature": 0.7}
    assert manager._group_sessions["group-1"]["planner"].policy_name == "policy_a"
    assert manager._group_sessions["group-1"]["planner"].role_session_id == "group-1:planner"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_duplicate_group_is_rejected_while_first_creation_is_in_flight(
    fake_gateway_pool,
):
    manager = MultiAgentGatewayManager.__new__(MultiAgentGatewayManager)
    manager.gateways = fake_gateway_pool
    manager.active_sessions_per_gateway = [0, 0]
    manager._group_sessions = {}
    manager._group_gateway_indices = {}

    create_started = asyncio.Event()
    allow_create_to_finish = asyncio.Event()
    create_count = 0

    class DelayedCreate:
        calls = []

        async def remote(self, *, session_id, **kwargs):
            nonlocal create_count
            self.calls.append(session_id)
            create_count += 1
            if create_count == 1:
                create_started.set()
                await allow_create_to_finish.wait()
            return SessionHandle(session_id=session_id, base_url=f"http://fake/{session_id}/v1")

    fake_gateway_pool[0].create_session = DelayedCreate()
    first_create = asyncio.create_task(manager.create_session("group-in-flight", role_policy_mapping={"planner": "p1"}))
    await create_started.wait()

    duplicate_create = asyncio.create_task(
        manager.create_session("group-in-flight", role_policy_mapping={"planner": "p1"})
    )
    await asyncio.sleep(0)
    allow_create_to_finish.set()
    await first_create

    with pytest.raises(ValueError, match="Duplicate session group"):
        await duplicate_create

    assert fake_gateway_pool[0].create_session.calls == ["group-in-flight:planner"]
    assert manager.active_sessions_per_gateway == [1, 0]
    assert set(manager._group_sessions) == {"group-in-flight"}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_cancelled_creation_aborts_in_flight_and_pending_roles(fake_gateway_pool):
    manager = MultiAgentGatewayManager.__new__(MultiAgentGatewayManager)
    manager.gateways = fake_gateway_pool
    manager.active_sessions_per_gateway = [0, 0]
    manager._group_sessions = {}
    manager._group_gateway_indices = {}

    allow_create_to_finish = asyncio.Event()

    class CancellationSafeCreate:
        def __init__(self):
            self.calls = []
            self.pending_tasks = []

        def remote(self, *, session_id, **kwargs):
            self.calls.append(session_id)

            async def run():
                await allow_create_to_finish.wait()
                return SessionHandle(
                    session_id=session_id,
                    base_url=f"http://fake/{session_id}/v1",
                )

            task = asyncio.create_task(run())
            self.pending_tasks.append(task)
            return asyncio.shield(task)

    create = CancellationSafeCreate()
    fake_gateway_pool[0].create_session = create

    creation = asyncio.create_task(
        manager.create_session(
            "cancelled-group",
            role_policy_mapping={"planner": "p1", "actor": "p2"},
        )
    )
    await asyncio.sleep(0)
    creation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await creation

    aborted_ids = {call[1]["session_id"] for call in fake_gateway_pool[0].abort_session.calls}
    assert aborted_ids == {
        "cancelled-group:planner",
        "cancelled-group:actor",
    }
    assert manager._group_sessions == {}
    assert manager._group_gateway_indices == {}
    assert manager.active_sessions_per_gateway == [0, 0]

    allow_create_to_finish.set()
    await asyncio.gather(*create.pending_tasks)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_finalize_session_aggregates_role_trajectories(fake_gateway_pool):
    manager = MultiAgentGatewayManager.__new__(MultiAgentGatewayManager)
    manager.gateways = fake_gateway_pool
    manager.active_sessions_per_gateway = [0, 0]
    manager._group_sessions = {}
    manager._group_gateway_indices = {}

    await manager.create_session(
        "group-2",
        role_policy_mapping={"role_a": "policy_x", "role_b": "policy_y"},
    )

    trajectories = await manager.finalize_session(session_id="group-2")

    assert len(trajectories) == 2
    assert trajectories[0].extra_fields["role"] == "role_a"
    assert trajectories[0].extra_fields["policy_name"] == "policy_x"
    assert trajectories[0].extra_fields["session_group_id"] == "group-2"
    assert trajectories[1].extra_fields["role"] == "role_b"
    assert trajectories[1].extra_fields["policy_name"] == "policy_y"
    assert manager._group_sessions == {}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_abort_session_aborts_all_role_sessions(fake_gateway_pool):
    manager = MultiAgentGatewayManager.__new__(MultiAgentGatewayManager)
    manager.gateways = fake_gateway_pool
    manager.active_sessions_per_gateway = [0, 0]
    manager._group_sessions = {}
    manager._group_gateway_indices = {}

    await manager.create_session(
        "group-3",
        role_policy_mapping={"role_a": "policy_x", "role_b": "policy_y"},
    )

    await manager.abort_session(session_id="group-3")

    gateway = fake_gateway_pool[0]
    assert len(gateway.abort_session.calls) == 2
    aborted_ids = {call[1]["session_id"] for call in gateway.abort_session.calls}
    assert aborted_ids == {"group-3:role_a", "group-3:role_b"}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_groups_spread_across_gateway_pool(fake_gateway_pool):
    manager = MultiAgentGatewayManager.__new__(MultiAgentGatewayManager)
    manager.gateways = fake_gateway_pool
    manager.active_sessions_per_gateway = [0, 0]
    manager._group_sessions = {}
    manager._group_gateway_indices = {}

    await manager.create_session("g1", role_policy_mapping={"r": "p"})
    await manager.create_session("g2", role_policy_mapping={"r": "p"})

    # Groups should spread across the two gateways
    assert fake_gateway_pool[0].create_session.calls or fake_gateway_pool[1].create_session.calls
    total_creates = len(fake_gateway_pool[0].create_session.calls) + len(fake_gateway_pool[1].create_session.calls)
    assert total_creates == 2


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_shutdown_clears_group_state_after_parent_shutdown(monkeypatch):
    manager = MultiAgentGatewayManager.__new__(MultiAgentGatewayManager)
    gateway = _FakeActor()
    manager.gateways = [gateway]
    manager.active_sessions_per_gateway = [1]
    manager._group_sessions = {
        "group-1": {
            "planner": _RoleSession("group-1:planner", "p1"),
            "actor": _RoleSession("group-1:actor", "p2"),
        }
    }
    manager._group_gateway_indices = {"group-1": 0}

    parent_shutdown_called = False

    async def fake_parent_shutdown(self):
        nonlocal parent_shutdown_called
        parent_shutdown_called = True
        assert len(gateway.abort_session.calls) == 2
        self.gateways = []
        self.active_sessions_per_gateway = []

    monkeypatch.setattr(gateway_manager_module.GatewayManager, "shutdown", fake_parent_shutdown)

    await manager.shutdown()

    assert parent_shutdown_called
    assert {call[1]["session_id"] for call in gateway.abort_session.calls} == {
        "group-1:planner",
        "group-1:actor",
    }
    assert manager._group_sessions == {}
    assert manager._group_gateway_indices == {}
