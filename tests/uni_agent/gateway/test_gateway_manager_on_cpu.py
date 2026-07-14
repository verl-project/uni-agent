import asyncio
import os

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import httpx
import pytest
import ray

from tests.uni_agent.support import FakeTokenizer, RecordingLLMClient, SequencedBackend


@pytest.fixture(scope="session")
def ray_runtime():
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    yield
    ray.shutdown()


class _FakeRemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeGateway:
    """Minimal gateway stub whose create_session.remote yields at an await, so
    concurrent creates interleave the way real Ray RPCs do."""

    def __init__(self):
        self.created = []

    async def _create(self, session_id, **kwargs):
        await asyncio.sleep(0)  # yield the loop mid-create to expose select/increment races
        self.created.append(session_id)
        return session_id

    @property
    def create_session(self):
        return _FakeRemoteMethod(self._create)


@pytest.mark.asyncio
async def test_gateway_manager_balances_concurrent_session_creation():
    """Concurrently created sessions must spread evenly across gateways. The
    selection counter has to be reserved before the create await; otherwise a
    burst of concurrent creates all read the same stale counts and pile onto the
    lowest-index gateway. Builds over fake gateways (no real spawn) to drive the
    routing logic deterministically.
    """
    from uni_agent.gateway.manager import GatewayManager

    gateways = [_FakeGateway() for _ in range(4)]
    # No real Ray spawn: drive the routing logic over injected fakes.
    manager = GatewayManager.__new__(GatewayManager)
    manager.gateways = gateways
    manager.gateway_count = len(gateways)
    manager.active_sessions_per_gateway = [0] * len(gateways)
    manager._session_to_gateway_index = {}

    await asyncio.gather(*(manager.create_session(f"session-{i}") for i in range(40)))

    counts = manager.active_sessions_per_gateway
    assert sum(counts) == 40
    assert max(counts) - min(counts) <= 1, counts
    assert [len(g.created) for g in gateways] == counts


def test_gateway_manager_rejects_zero_gateway_count():
    """``GatewayManager`` raises ``ValueError`` when ``gateway_count=0``, rather
    than silently spawning a half-initialized manager."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager

    with pytest.raises(ValueError, match="gateway_count must be positive"):
        GatewayManager(
            llm_client=object(),
            gateway_count=0,
            gateway_actor_config=GatewayActorConfig(tokenizer=object()),
        )


@pytest.mark.asyncio
async def test_gateway_manager_round_robins_actors_across_alive_nodes(ray_runtime, monkeypatch):
    """gateway_count > 1 should distribute actors across alive CPU nodes round-robin."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager

    fake_nodes = [
        {"NodeID": "a" * 56, "Alive": True, "Resources": {"CPU": 8.0}},
        {"NodeID": "b" * 56, "Alive": True, "Resources": {"CPU": 8.0}},
        {"NodeID": "c" * 56, "Alive": False, "Resources": {"CPU": 8.0}},
        {"NodeID": "d" * 56, "Alive": True, "Resources": {"GPU": 1.0}},
    ]
    monkeypatch.setattr("uni_agent.gateway.manager.ray.nodes", lambda: fake_nodes)

    captured_node_ids = []

    class _StubStartHandle:
        @staticmethod
        def remote():
            return ray.put(None)

    class _StubActorHandle:
        start = _StubStartHandle()

    class _RecordingActor:
        @classmethod
        def options(cls, *, scheduling_strategy):
            captured_node_ids.append(scheduling_strategy.node_id)
            return cls

        @classmethod
        def remote(cls, config, backend):
            return _StubActorHandle()

    monkeypatch.setattr("uni_agent.gateway.gateway.GatewayActor", _RecordingActor)

    manager = GatewayManager(
        llm_client=object(),
        gateway_count=5,
        gateway_actor_config=GatewayActorConfig(tokenizer=object()),
    )

    assert captured_node_ids == ["a" * 56, "b" * 56, "a" * 56, "b" * 56, "a" * 56], captured_node_ids
    assert len(manager.gateways) == 5
    # No shutdown call needed: stub actors have no real Ray state.


@pytest.mark.asyncio
async def test_gateway_manager_finalizes_each_session_on_its_owning_gateway(ray_runtime):
    """create/finalize must route every session back to the same owning gateway
    across a multi-gateway pool. Two sessions land on different gateways; each is
    tagged with its own reward_info, and finalize must return that session's own
    trajectory -- a routing-table mix-up would surface as a swapped label.

    Asserts the manager's ownership routing (not the selection policy): any
    placement policy must still finalize a session on the gateway that created it.
    """
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager

    manager = GatewayManager(
        llm_client=RecordingLLMClient("OK"),
        gateway_count=2,
        gateway_actor_config=GatewayActorConfig(tokenizer=FakeTokenizer()),
    )

    session_a = await manager.create_session("session-a")
    session_b = await manager.create_session("session-b")
    # Least-loaded routing places the two sessions on distinct gateways, so a
    # mis-routed finalize would hit the other session's gateway.
    assert session_a.base_url != session_b.base_url

    async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
        for session, label in ((session_a, "a"), (session_b, "b")):
            chat = await client.post(
                f"{session.base_url}/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": f"hi {label}"}]},
            )
            assert chat.status_code == 200
            reward = await client.post(session.reward_info_url, json={"reward_info": {"label": label}})
            assert reward.status_code == 200

    trajectories_a = await manager.finalize_session("session-a")
    trajectories_b = await manager.finalize_session("session-b")

    assert [t.reward_info["label"] for t in trajectories_a] == ["a"]
    assert [t.reward_info["label"] for t in trajectories_b] == ["b"]

    await manager.shutdown()


@pytest.mark.asyncio
async def test_gateway_manager_default_chains_config_to_http_finalizes_subagent_before_updated_main(
    ray_runtime,
    monkeypatch,
):
    """Exercise default multi-chain behavior from config wiring through HTTP routes."""
    from omegaconf import OmegaConf

    from uni_agent.framework import entry as entry_module

    class _ModelConfig:
        tokenizer = FakeTokenizer()
        processor = None

    monkeypatch.setattr(entry_module, "omega_conf_to_dataclass", lambda _config: _ModelConfig())
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {},
                "rollout": {
                    "prompt_length": 2048,
                    "response_length": 2048,
                    "multi_turn": {"format": None},
                    "custom": {"agent_framework": {"gateway_count": 1}},
                },
            }
        }
    )
    manager = entry_module.build_gateway_manager(
        config=config,
        llm_client=SequencedBackend(["Mango", "Blue", "Apple"]),
    )

    try:
        session = await manager.create_session("session-manager-multiple-chains")
        main_first = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "name a fruit"},
        ]
        subagent = [
            {"role": "system", "content": "You are a focused subagent."},
            {"role": "user", "content": "name a color"},
        ]
        main_continuation = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "name a fruit"},
            {"role": "assistant", "content": "Mango"},
            {"role": "user", "content": "name another fruit"},
        ]

        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            for messages in (main_first, subagent, main_continuation):
                chat = await client.post(
                    f"{session.base_url}/chat/completions",
                    json={"model": "m", "messages": messages},
                )
                assert chat.status_code == 200

            reward = await client.post(
                session.reward_info_url,
                json={"reward_info": {"label": "manager-multiple-chains"}},
            )
            assert reward.status_code == 200

        trajectories = await manager.finalize_session("session-manager-multiple-chains")

        assert len(trajectories) == 2
        decoded = [FakeTokenizer().decode(trajectory.response_ids) for trajectory in trajectories]
        assert decoded[0] == "Blue"
        assert decoded[1].startswith("Mango")
        assert decoded[1].endswith("Apple")
        assert [trajectory.reward_info["label"] for trajectory in trajectories] == [
            "manager-multiple-chains",
            "manager-multiple-chains",
        ]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_gateway_manager_allows_concurrent_http_requests_within_one_session(ray_runtime):
    """Two HTTP requests must reach backend generation concurrently and both materialize."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager
    from verl.workers.rollout.replica import TokenOutput

    class _BarrierBackend:
        def __init__(self):
            self.started = 0
            self.both_started = asyncio.Event()

        async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
            response_index = self.started
            self.started += 1
            if self.started == 2:
                self.both_started.set()
            await asyncio.wait_for(self.both_started.wait(), timeout=2)
            text = ["FIRST", "SECOND"][response_index]
            return TokenOutput(
                token_ids=[ord(char) for char in text],
                log_probs=[-0.1] * len(text),
                stop_reason="completed",
            )

    manager = GatewayManager(
        llm_client=_BarrierBackend(),
        gateway_count=1,
        gateway_actor_config=GatewayActorConfig(tokenizer=FakeTokenizer(), response_length=128),
    )

    try:
        session = await manager.create_session("session-manager-concurrent-http")
        payload = {"model": "m", "messages": [{"role": "user", "content": "same prompt"}]}
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            first, second = await asyncio.gather(
                client.post(f"{session.base_url}/chat/completions", json=payload),
                client.post(f"{session.base_url}/chat/completions", json=payload),
            )

        assert first.status_code == second.status_code == 200
        trajectories = await manager.finalize_session("session-manager-concurrent-http")
        assert sorted(FakeTokenizer().decode(trajectory.response_ids) for trajectory in trajectories) == [
            "FIRST",
            "SECOND",
        ]
    finally:
        await manager.shutdown()
