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


@pytest.mark.cpu
@pytest.mark.level0
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
    manager._session_owners = {}
    manager._llm_client = object()

    await asyncio.gather(*(manager.create_session(f"session-{i}") for i in range(40)))

    counts = manager.active_sessions_per_gateway
    assert sum(counts) == 40
    assert max(counts) - min(counts) <= 1, counts
    assert [len(g.created) for g in gateways] == counts


@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
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


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_finalizes_each_session_on_its_owning_gateway(ray_runtime):
    """create/finalize must route every session back to the same owning gateway
    across a multi-gateway pool. Two sessions land on different gateways and a
    routing-table mix-up would try to finalize an unknown session on the wrong actor.

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

    trajectories_a = await manager.finalize_session("session-a")
    trajectories_b = await manager.finalize_session("session-b")

    assert len(trajectories_a) == 1
    assert len(trajectories_b) == 1

    await manager.shutdown()


@pytest.mark.cpu
@pytest.mark.level0
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
        hf_config = None

    class _MultiTurnConfig:
        format = None

    class _RolloutConfig:
        name = None
        prompt_length = 2048
        response_length = 2048
        multi_turn = _MultiTurnConfig()

    monkeypatch.setattr(
        entry_module,
        "omega_conf_to_dataclass",
        lambda cfg: _RolloutConfig() if "multi_turn" in cfg else _ModelConfig(),
    )
    config = OmegaConf.create(
        {
            "data": {},
            "actor_rollout_ref": {
                "model": {},
                "rollout": {
                    "prompt_length": 2048,
                    "response_length": 2048,
                    "multi_turn": {"format": None},
                    "custom": {"agent_framework": {"gateway_count": 1}},
                },
            },
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

        trajectories = await manager.finalize_session("session-manager-multiple-chains")

        assert len(trajectories) == 2
        decoded = [FakeTokenizer().decode(trajectory.response_ids) for trajectory in trajectories]
        assert decoded[0] == "Blue"
        assert decoded[1].startswith("Mango")
        assert decoded[1].endswith("Apple")
    finally:
        await manager.shutdown()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_allows_independent_http_requests_when_coalescing_disabled(ray_runtime):
    """Opting out lets identical HTTP requests generate concurrently and both materialize."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager
    from verl.workers.rollout.replica import TokenOutput

    class _BarrierBackend:
        def __init__(self):
            self.started = 0
            self.both_started = asyncio.Event()

        async def generate(
            self,
            request_id,
            *,
            prompt_ids,
            sampling_params,
            image_data=None,
            video_data=None,
            mm_processor_kwargs=None,
        ):
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
        gateway_actor_config=GatewayActorConfig(
            tokenizer=FakeTokenizer(), response_length=128, coalesce_reserved_exact_requests=False
        ),
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


class _LifecycleGateway:
    """Controllable remote-method stub for close failures."""

    def __init__(self, *, finalize_error=None, abort_error=None):
        self.finalize_error = finalize_error
        self.abort_error = abort_error
        self.abort_calls = []
        self.finalize_calls = []

    async def _abort(self, session_id, **kwargs):
        self.abort_calls.append(session_id)
        if self.abort_error is not None:
            raise self.abort_error

    async def _finalize(self, session_id, **kwargs):
        self.finalize_calls.append(session_id)
        if self.finalize_error is not None:
            raise self.finalize_error
        return ["trajectory"]

    async def _close(self, session_id, **kwargs):
        return None

    abort_session = property(lambda self: _FakeRemoteMethod(self._abort))
    finalize_session = property(lambda self: _FakeRemoteMethod(self._finalize))
    close_session = property(lambda self: _FakeRemoteMethod(self._close))


def _new_manager(gateway, *, session_ids=(), llm_client=None):
    from uni_agent.gateway.manager import GatewayManager, _SessionOwner

    manager = GatewayManager.__new__(GatewayManager)
    manager.gateways = [gateway]
    manager.active_sessions_per_gateway = [len(session_ids)]
    manager._session_owners = {session_id: _SessionOwner(0) for session_id in session_ids}
    if isinstance(gateway, _RouteGateway):
        gateway.actor._backend = llm_client or object()
        for session_id, owner in manager._session_owners.items():
            gateway.seed_session(session_id, owner.creation_id)
    return manager


class _DetachedRemoteMethod:
    """Ray-like call: cancelling the caller's await does not stop the actor method."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return asyncio.shield(asyncio.ensure_future(self._fn(*args, **kwargs)))


class _SlowCreateGateway(_LifecycleGateway):
    def __init__(self, *, create_error=None):
        super().__init__()
        self.create_error = create_error
        self.created = []
        self.create_started = asyncio.Event()
        self.create_finished = asyncio.Event()
        self.release_create = asyncio.Event()

    async def _create(self, session_id, **kwargs):
        self.create_started.set()
        await self.release_create.wait()
        self.create_finished.set()
        if self.create_error is not None:
            raise self.create_error
        self.created.append(session_id)
        return session_id

    create_session = property(lambda self: _DetachedRemoteMethod(self._create))
    abort_session = property(lambda self: _DetachedRemoteMethod(self._abort))


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("create_error", [None, RuntimeError("create failed")])
async def test_gateway_manager_cancelled_create_aborts_what_the_actor_created(create_error):
    """After the caller cancels, only the manager can still abort the actor-side session."""
    gateway = _SlowCreateGateway(create_error=create_error)
    manager = _new_manager(gateway)

    create = asyncio.create_task(manager.create_session("session-cancelled"))
    await gateway.create_started.wait()
    create.cancel()
    await asyncio.sleep(0)
    gateway.release_create.set()
    with pytest.raises(asyncio.CancelledError):
        await create
    await asyncio.wait_for(gateway.create_finished.wait(), timeout=1)

    assert gateway.abort_calls == ["session-cancelled"]
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_cancelled_create_retains_owner_after_abort_failure():
    gateway = _SlowCreateGateway()
    gateway.abort_error = RuntimeError("session abort failed before removal")
    manager = _new_manager(gateway)
    create = asyncio.create_task(manager.create_session("session-cancelled"))
    await gateway.create_started.wait()
    create.cancel()
    gateway.release_create.set()
    with pytest.raises(asyncio.CancelledError):
        await create

    assert set(manager._session_owners) == {"session-cancelled"}
    assert manager._session_owners["session-cancelled"].gateway_index == 0
    assert manager.active_sessions_per_gateway == [1]
    gateway.abort_error = None
    await manager.abort_session("session-cancelled")
    assert gateway.abort_calls == ["session-cancelled"] * 2
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_repeated_cancellation_still_settles_remote_create():
    gateway = _SlowCreateGateway()
    manager = _new_manager(gateway)
    create = asyncio.create_task(manager.create_session("session-cancelled"))
    await gateway.create_started.wait()
    for _ in range(3):
        create.cancel()
        await asyncio.sleep(0)
    still_waiting = not create.done()
    gateway.release_create.set()
    with pytest.raises(asyncio.CancelledError):
        await create
    await asyncio.wait_for(gateway.create_finished.wait(), timeout=1)

    assert still_waiting
    assert gateway.abort_calls == ["session-cancelled"]
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_close_failure_retains_mapping_and_owner():
    close_error = RuntimeError("actor-close-unknown")
    gateway = _LifecycleGateway(finalize_error=close_error)
    manager = _new_manager(gateway, session_ids=("session-close",))

    with pytest.raises(RuntimeError, match="actor-close-unknown") as raised:
        await manager.finalize_session("session-close")

    assert raised.value is close_error
    assert gateway.finalize_calls == ["session-close"]
    assert set(manager._session_owners) == {"session-close"}
    assert manager._session_owners["session-close"].gateway_index == 0
    assert manager.active_sessions_per_gateway == [1]

    await manager.abort_session("session-close")

    assert gateway.abort_calls == ["session-close"]
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_rejects_duplicate_create_without_touching_owner():
    gateway = _SlowCreateGateway()
    manager = _new_manager(gateway)
    create = asyncio.create_task(manager.create_session("same-id"))
    await gateway.create_started.wait()

    with pytest.raises(RuntimeError, match="already exists"):
        await manager.create_session("same-id")
    assert set(manager._session_owners) == {"same-id"}
    assert manager._session_owners["same-id"].gateway_index == 0
    assert manager.active_sessions_per_gateway == [1]
    assert gateway.abort_calls == []

    gateway.release_create.set()
    assert await create == "same-id"
    await manager.abort_session("same-id")
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("bad_awaitable", [False, True])
async def test_gateway_manager_releases_reservation_after_synchronous_submission_failure(bad_awaitable):
    class _SubmissionFailureGateway(_LifecycleGateway):
        def _create(self, session_id, **kwargs):
            if bad_awaitable:
                return object()
            raise TypeError("session arguments cannot be serialized")

        create_session = property(lambda self: _FakeRemoteMethod(self._create))

    gateway = _SubmissionFailureGateway()
    manager = _new_manager(gateway)
    with pytest.raises(TypeError):
        await manager.create_session("failed")
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]
    assert gateway.abort_calls == ["failed"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_concurrent_closes_release_load_slot_once():
    class _ConcurrentCloseGateway(_LifecycleGateway):
        async def _finalize(self, session_id, **kwargs):
            await asyncio.sleep(0)
            return await super()._finalize(session_id, **kwargs)

        async def _abort(self, session_id, **kwargs):
            await asyncio.sleep(0)
            await super()._abort(session_id, **kwargs)

    gateway = _ConcurrentCloseGateway()
    manager = _new_manager(gateway, session_ids=("session-close",))
    await asyncio.gather(manager.finalize_session("session-close"), manager.abort_session("session-close"))
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]


class _RouteClient:
    """Return non-coroutine awaitables, as Ray-backed routing clients do."""

    def __init__(self, bound_version=7, *, bind_error=None, release_errors=()):
        self.bound_version = bound_version
        self.bind_error = bind_error
        self.release_errors = list(release_errors)
        self.bind_calls = []
        self.release_calls = []
        self.bind_started = asyncio.Event()
        self.release_started = asyncio.Event()
        self.allow_bind = asyncio.Event()
        self.allow_release = asyncio.Event()
        self.allow_bind.set()
        self.allow_release.set()

    def bind_route(self, **kwargs):
        async def bind():
            self.bind_calls.append(kwargs)
            self.bind_started.set()
            await self.allow_bind.wait()
            if self.bind_error is not None:
                raise self.bind_error
            return self.bound_version

        return asyncio.ensure_future(bind())

    def release_route(self, **kwargs):
        async def release():
            self.release_calls.append(kwargs)
            self.release_started.set()
            await self.allow_release.wait()
            if self.release_errors:
                raise self.release_errors.pop(0)

        return asyncio.ensure_future(release())


class _RouteGateway(_SlowCreateGateway):
    def __init__(self, **kwargs):
        from uni_agent.gateway.config import GatewayActorConfig
        from uni_agent.gateway.gateway import _GatewayActor

        super().__init__(**kwargs)
        self.create_kwargs = []
        self.release_create.set()
        self.actor = _GatewayActor(GatewayActorConfig(tokenizer=FakeTokenizer()), backend=object())
        self.actor._server_base_url = "http://gateway"

    def seed_session(self, session_id, creation_id):
        from uni_agent.gateway.session import GatewaySession, SessionHandle
        from uni_agent.gateway.session.session import SessionPhase

        session = GatewaySession(SessionHandle(session_id=session_id), self.actor._codec)
        session._creation_id = creation_id
        session.phase = SessionPhase.FINALIZED
        self.actor._sessions[session_id] = session

    async def _create(self, session_id, **kwargs):
        self.create_kwargs.append(kwargs)
        await super()._create(session_id, **kwargs)
        return await self.actor.create_session(session_id, **kwargs)

    async def _abort(self, session_id, **kwargs):
        await super()._abort(session_id, **kwargs)
        await self.actor.abort_session(session_id, **kwargs)

    async def _finalize(self, session_id, **kwargs):
        from uni_agent.gateway.session.session import SessionPhase

        self.finalize_calls.append(session_id)
        # Seeded manager tests model a successful finalize ACK, keeping its
        # actor-side FINALIZED state for release retries.
        if self.actor._get_session(session_id).phase != SessionPhase.FINALIZED:
            await self.actor.finalize_session(session_id, **kwargs)
        return ["trajectory"]

    async def _close(self, session_id, **kwargs):
        await self.actor.close_session(session_id, **kwargs)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("requested,bound", [(3, 7), (3, None), (None, 7)])
async def test_gateway_manager_passes_requested_version_to_actor_backend(requested, bound):
    backend = _RouteClient(bound)
    gateway = _RouteGateway()
    manager = _new_manager(gateway, llm_client=backend)
    await manager.create_session("bound", weight_version=requested, metadata={"owner": "runner"}, sampling_params={})
    assert backend.bind_calls == [{"session_id": "bound", "weight_version": requested}]
    assert gateway.create_kwargs == [
        {
            "weight_version": requested,
            "creation_id": manager._session_owners["bound"].creation_id,
            "metadata": {"owner": "runner"},
            "sampling_params": {},
        }
    ]
    await manager.abort_session("bound")
    assert backend.release_calls == [{"session_id": "bound"}]
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_plain_backend_does_not_pin_batch_version():
    gateway = _RouteGateway()
    manager = _new_manager(gateway)
    await manager.create_session("plain", weight_version=3)
    assert gateway.create_kwargs == [{"weight_version": 3, "creation_id": manager._session_owners["plain"].creation_id}]
    await manager.abort_session("plain")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["bind", "actor"])
@pytest.mark.parametrize("release_fails", [False, True])
async def test_gateway_manager_failed_create_keeps_original_error_and_retry_owner(stage, release_fails):
    original = RuntimeError("original create failure")
    backend = _RouteClient(
        bind_error=original if stage == "bind" else None,
        release_errors=[RuntimeError("cleanup failed")] * 2 if release_fails else [],
    )
    gateway = _RouteGateway(create_error=original if stage == "actor" else None)
    manager = _new_manager(gateway, llm_client=backend)
    with pytest.raises(RuntimeError) as raised:
        await manager.create_session("failed", weight_version=3)
    assert raised.value is original
    expected_releases = [{"session_id": "failed"}] * (2 if release_fails else 1) if stage == "bind" else []
    assert backend.release_calls == expected_releases
    if release_fails and stage == "bind":
        assert "failed" in manager._session_owners
        await manager.abort_session("failed")
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [1, 2])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_gateway_manager_release_failure_keeps_finalized_trajectories_and_retry_owner(failures, error_type):
    backend = _RouteClient(release_errors=[error_type("release failed") for _ in range(failures)])
    gateway = _RouteGateway()
    manager = _new_manager(gateway, session_ids=("finished",), llm_client=backend)
    assert await manager.finalize_session("finished") == ["trajectory"]
    assert len(backend.release_calls) == 2
    assert gateway.abort_calls == []
    if failures == 2:
        assert "finished" in gateway.actor._sessions
        assert manager.active_sessions_per_gateway == [1]
        await manager.abort_session("finished")
        assert gateway.abort_calls == ["finished"]
    assert manager.active_sessions_per_gateway == [0]
    assert manager._session_owners == {}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["bind", "create"])
async def test_gateway_manager_cancelled_route_creation_settles_before_owned_abort(phase):
    backend = _RouteClient()
    gateway = _RouteGateway()
    if phase == "bind":
        backend.allow_bind.clear()
        started, allow = backend.bind_started, backend.allow_bind
    else:
        gateway.release_create.clear()
        started, allow = gateway.create_started, gateway.release_create
    manager = _new_manager(gateway, llm_client=backend)
    create = asyncio.create_task(manager.create_session("cancelled", weight_version=3))
    await started.wait()
    for _ in range(3):
        create.cancel()
        await asyncio.sleep(0)
    assert not create.done()
    allow.set()
    with pytest.raises(asyncio.CancelledError):
        await create
    assert gateway.abort_calls == ["cancelled"]
    assert backend.release_calls == [{"session_id": "cancelled"}]
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("close", ["finalize_session", "abort_session"])
async def test_gateway_manager_repeated_cancellation_settles_route_release_then_propagates(close):
    backend = _RouteClient()
    backend.allow_release.clear()
    gateway = _RouteGateway()
    manager = _new_manager(gateway, session_ids=("closing",), llm_client=backend)
    closing = asyncio.create_task(getattr(manager, close)("closing"))
    await backend.release_started.wait()
    for _ in range(3):
        closing.cancel()
        await asyncio.sleep(0)
    assert not closing.done()
    backend.allow_release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert backend.release_calls == [{"session_id": "closing"}]
    assert manager.active_sessions_per_gateway == [0]
    assert manager._session_owners == {}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [2, True, "backend-specific"])
async def test_gateway_ignores_backend_bind_result(bound):
    backend = _RouteClient(bound)
    gateway = _RouteGateway()
    manager = _new_manager(gateway, llm_client=backend)
    await manager.create_session("opaque", weight_version=3)
    await manager.abort_session("opaque")
    assert backend.bind_calls == [{"session_id": "opaque", "weight_version": 3}]
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_bind_error_survives_cancelled_release():
    original = RuntimeError("bind failed")
    backend = _RouteClient(bind_error=original, release_errors=[asyncio.CancelledError()] * 2)
    manager = _new_manager(_RouteGateway(), llm_client=backend)
    with pytest.raises(RuntimeError) as raised:
        await manager.create_session("failed", weight_version=3)
    assert raised.value is original
    assert "failed" in manager._session_owners
    await manager.abort_session("failed")
    assert manager._session_owners == {}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [1, 2])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_gateway_manager_cancelled_finalize_propagates_even_when_release_fails(failures, error_type):
    backend = _RouteClient(release_errors=[error_type("release failed") for _ in range(failures)])
    backend.allow_release.clear()
    manager = _new_manager(_RouteGateway(), session_ids=("closing",), llm_client=backend)
    closing = asyncio.create_task(manager.finalize_session("closing"))
    await backend.release_started.wait()
    closing.cancel()
    await asyncio.sleep(0)
    backend.allow_release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    if failures == 2:
        assert "closing" in manager._session_owners
        await manager.abort_session("closing")
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_gateway_manager_cancelled_finalize_settles_actor_before_route_release():
    started, finish = asyncio.Event(), asyncio.Event()

    class _SlowFinalizeGateway(_RouteGateway):
        async def _finalize(self, session_id, **kwargs):
            started.set()
            await finish.wait()
            return await super()._finalize(session_id, **kwargs)

    backend = _RouteClient()
    gateway = _SlowFinalizeGateway()
    manager = _new_manager(gateway, session_ids=("closing",), llm_client=backend)
    closing = asyncio.create_task(manager.finalize_session("closing"))
    await started.wait()
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done() and backend.release_calls == []
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert backend.release_calls == [{"session_id": "closing"}]
    assert manager._session_owners == {}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_failed_duplicate_actor_create_never_closes_preexisting_session():
    backend = _RouteClient()
    gateway = _RouteGateway()
    manager = _new_manager(gateway, llm_client=backend)
    await gateway.actor.create_session("duplicate", creation_id="original", weight_version=3)
    original = gateway.actor._sessions["duplicate"]
    with pytest.raises(RuntimeError, match="already exists"):
        await manager.create_session("duplicate", weight_version=3)
    assert gateway.actor._sessions["duplicate"] is original
    assert backend.release_calls == []
    assert manager._session_owners == {}
    assert manager.active_sessions_per_gateway == [0]
    await gateway.actor.abort_session("duplicate", creation_id="original")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_delayed_actor_cleanup_cannot_close_replacement_with_same_session_id():
    backend = _RouteClient()
    gateway = _RouteGateway()
    manager = _new_manager(gateway, llm_client=backend)
    await manager.create_session("reused", weight_version=3)
    old_creation = manager._session_owners["reused"].creation_id
    await manager.finalize_session("reused")
    await manager.create_session("reused", weight_version=4)
    replacement = gateway.actor._sessions["reused"]
    await gateway.actor.close_session("reused", creation_id=old_creation)
    await gateway.actor.abort_session("reused", creation_id=old_creation)
    with pytest.raises(KeyError, match="different creation"):
        await gateway.actor.finalize_session("reused", creation_id=old_creation)
    assert gateway.actor._sessions["reused"] is replacement
    assert backend.release_calls == [{"session_id": "reused"}]
    await manager.abort_session("reused")
    assert manager.active_sessions_per_gateway == [0]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("creation_id", ["original", None], ids=["manager-nonce", "direct-legacy"])
@pytest.mark.parametrize("terminal", ["finalized", "removed", "replaced"])
async def test_actor_rejects_late_bind_result_after_its_session_closes(creation_id, terminal):
    from uni_agent.gateway.session.session import SessionPhase

    class _LateBindClient(_RouteClient):
        def bind_route(self, **kwargs):
            async def bind():
                self.bind_calls.append(kwargs)
                if kwargs["weight_version"] == 3:
                    self.bind_started.set()
                    await self.allow_bind.wait()

            return asyncio.ensure_future(bind())

    backend = _LateBindClient()
    backend.allow_bind.clear()
    gateway = _RouteGateway()
    gateway.actor._backend = backend
    creating = asyncio.create_task(gateway.actor.create_session("reused", weight_version=3, creation_id=creation_id))
    await asyncio.wait_for(backend.bind_started.wait(), 2)
    if terminal == "finalized":
        await gateway.actor.finalize_session("reused", creation_id=creation_id, close=False)
    else:
        await gateway.actor.abort_session("reused", creation_id=creation_id)
    replacement = None
    if terminal == "replaced":
        await gateway.actor.create_session("reused", weight_version=4, creation_id="replacement")
        replacement = gateway.actor._sessions["reused"]

    backend.allow_bind.set()
    with pytest.raises(RuntimeError, match="closed during route binding"):
        await asyncio.wait_for(creating, 2)
    assert backend.release_calls == [{"session_id": "reused"}]
    if replacement is None:
        assert gateway.actor._sessions == {}
    else:
        assert gateway.actor._sessions["reused"] is replacement
        assert replacement.phase == SessionPhase.ACTIVE
        await gateway.actor.abort_session("reused", creation_id="replacement")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_legacy_create_failure_cleanup_cannot_abort_a_queued_replacement():
    from uni_agent.gateway.session.session import SessionPhase

    original_error = RuntimeError("original bind failed")
    gateway = _RouteGateway()

    class _BindOnlyClient:
        async def bind_route(self, *, session_id, weight_version):
            if weight_version == 3:
                self.original_creation = gateway.actor._sessions[session_id]._creation_id
                # This task runs when failed creation yields to its cleanup.
                self.replacing = asyncio.create_task(self.replace())
                raise original_error

        async def replace(self):
            await gateway.actor.finalize_session("reused", close=False)
            await gateway.actor.close_session("reused")
            await gateway.actor.create_session("reused", weight_version=4)
            self.replacement = gateway.actor._sessions["reused"]

    backend = _BindOnlyClient()
    gateway.actor._backend = backend
    with pytest.raises(RuntimeError) as raised:
        await gateway.actor.create_session("reused", weight_version=3)
    await asyncio.wait_for(backend.replacing, 2)
    assert raised.value is original_error
    assert gateway.actor._sessions["reused"] is backend.replacement
    assert backend.replacement.phase == SessionPhase.ACTIVE
    assert backend.original_creation != backend.replacement._creation_id
    await gateway.actor.abort_session("reused")
    assert gateway.actor._sessions == {}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent_abort", [False, True])
async def test_actor_release_settles_repeated_cancellation_and_concurrent_close(concurrent_abort):
    backend = _RouteClient()
    gateway = _RouteGateway()
    gateway.actor._backend = backend
    await gateway.actor.create_session("closing", creation_id="one")
    await gateway.actor.finalize_session("closing", creation_id="one", close=False)
    backend.allow_release.clear()
    close = asyncio.create_task(gateway.actor.close_session("closing", creation_id="one"))
    await backend.release_started.wait()
    other_close = (
        asyncio.create_task(gateway.actor.abort_session("closing", creation_id="one")) if concurrent_abort else None
    )
    for _ in range(3):
        close.cancel()
        await asyncio.sleep(0)
    assert not close.done()
    backend.allow_release.set()
    with pytest.raises(asyncio.CancelledError):
        await close
    if other_close is not None:
        await other_close
    assert backend.release_calls == [{"session_id": "closing"}]
    assert gateway.actor._sessions == {}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_actor_releases_after_session_abort_reports_deferred_cancellation():
    from uni_agent.gateway.session.session import SessionPhase

    backend = _RouteClient()
    gateway = _RouteGateway()
    gateway.actor._backend = backend
    await gateway.actor.create_session("cancelled", creation_id="one")
    session = gateway.actor._sessions["cancelled"]

    async def abort():
        session.phase = SessionPhase.ABORTED
        raise asyncio.CancelledError

    session.abort = abort
    with pytest.raises(asyncio.CancelledError):
        await gateway.actor.abort_session("cancelled", creation_id="one")
    assert backend.release_calls == [{"session_id": "cancelled"}]
    assert gateway.actor._sessions == {}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_manager_cancels_actor_bind_wait_without_waiting_for_readiness(ray_runtime):
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager

    @ray.remote
    class _Calls:
        def __init__(self):
            self.binds = 0
            self.releases = 0

        def bound(self):
            self.binds += 1

        def released(self):
            self.releases += 1

        def counts(self):
            return self.binds, self.releases

    class _WaitingBackend:
        def __init__(self, calls):
            self.calls = calls

        async def bind_route(self, **kwargs):
            await self.calls.bound.remote()
            await asyncio.Event().wait()

        def release_route(self, **kwargs):
            return self.calls.released.remote()

    calls = _Calls.remote()
    manager = GatewayManager(
        _WaitingBackend(calls), gateway_count=1, gateway_actor_config=GatewayActorConfig(tokenizer=FakeTokenizer())
    )
    try:
        creation = asyncio.create_task(manager.create_session("pending"))

        async def _wait_bound():
            while (await calls.counts.remote())[0] == 0:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_wait_bound(), timeout=10)
        for _ in range(3):
            creation.cancel()
            await asyncio.sleep(0)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(creation, timeout=10)
        assert await calls.counts.remote() == (1, 1)
        assert manager._session_owners == {}
        assert manager.active_sessions_per_gateway == [0]
        with pytest.raises(ray.exceptions.RayTaskError):
            await manager.gateways[0].get_session_state.remote("pending")
    finally:
        await manager.shutdown()
        ray.kill(calls)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pinned", "conflicting", "latest"])
async def test_gateway_manager_route_authority_is_shared_across_ray_client_copies(ray_runtime, mode):
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.manager import GatewayManager
    from verl.workers.rollout.replica import TokenOutput

    # Local classes are serialized by value; pytest's short module name is not
    # importable in Ray workers that use the repository-root Python path.
    class _SharedRoutingClient:
        def __init__(self, authority):
            self.authority = authority
            self.bound_versions = {}

        async def bind_route(self, **kwargs):
            version = await self.authority.bind.remote(**kwargs)
            self.bound_versions[kwargs["session_id"]] = version
            return version

        async def release_route(self, **kwargs):
            await self.authority.release.remote(**kwargs)
            self.bound_versions.pop(kwargs["session_id"], None)

        async def generate(
            self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None, **kwargs
        ):
            assert "weight_version" not in kwargs
            assert request_id in self.bound_versions
            output = await self.authority.generate.remote(session_id=request_id)
            expected = self.bound_versions[request_id]
            if expected is not None and output.extra_fields["min_global_steps"] != expected:
                raise RuntimeError("backend generation version conflicts with bound route")
            return output

    @ray.remote
    class _Authority:
        def __init__(self):
            self.routes = {}
            self.served_version = 8 if mode == "conflicting" else 7

        def bind(self, session_id, weight_version):
            self.routes[session_id] = 7
            return None if mode == "latest" else 7

        def generate(self, session_id):
            assert session_id in self.routes
            version = self.served_version
            if mode == "latest":
                self.served_version += 1
            return TokenOutput(
                token_ids=[ord("O"), ord("K")],
                log_probs=[-0.1, -0.1],
                stop_reason="completed",
                extra_fields={"min_global_steps": version, "max_global_steps": version},
            )

        def release(self, session_id):
            self.routes.pop(session_id, None)

        def remaining(self):
            return list(self.routes)

    authority = _Authority.remote()
    backend = _SharedRoutingClient(authority)
    manager = GatewayManager(
        backend,
        gateway_count=1,
        gateway_actor_config=GatewayActorConfig(tokenizer=FakeTokenizer()),
    )
    try:
        session = await manager.create_session("shared", weight_version=3)
        messages = [{"role": "user", "content": "prompt"}]
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            response = await client.post(f"{session.base_url}/chat/completions", json={"messages": messages})
            if mode == "conflicting":
                assert response.status_code == 500
                await manager.abort_session("shared")
            else:
                assert response.status_code == 200
                if mode == "latest":
                    messages += [{"role": "assistant", "content": "OK"}, {"role": "user", "content": "again"}]
                    response = await client.post(f"{session.base_url}/chat/completions", json={"messages": messages})
                    assert response.status_code == 200
                [trajectory] = await manager.finalize_session("shared")
                assert trajectory.extra_fields["min_global_steps"] == 7
                assert trajectory.extra_fields["max_global_steps"] == (8 if mode == "latest" else 7)
        assert await authority.remaining.remote() == []
        assert backend.bound_versions == {}  # Driver copy never owns the actor's bindings.
        assert manager.active_sessions_per_gateway == [0]
    finally:
        await manager.shutdown()
        ray.kill(authority)
