"""Gateway address selection and real HTTP listening, without model downloads."""

import pickle
import socket
from urllib.parse import urlsplit

import httpx
import pytest

from tests.uni_agent.support import FakeTokenizer, SequencedBackend
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor

pytestmark = [pytest.mark.cpu, pytest.mark.level0]

BIND_ADDRESS_ENV = "UNI_AGENT_GATEWAY_BIND_ADDRESS"


@pytest.fixture(autouse=True)
def clear_bind_address(monkeypatch):
    monkeypatch.delenv(BIND_ADDRESS_ENV, raising=False)


@pytest.mark.parametrize("ray_address", ["192.0.2.1", "[2001:0db8:0:0::1]"])
@pytest.mark.parametrize("empty_override", [False, True])
def test_default_address_tracks_the_actor_node(monkeypatch, ray_address, empty_override):
    if empty_override:
        monkeypatch.setenv(BIND_ADDRESS_ENV, "")
    config = GatewayActorConfig(tokenizer=FakeTokenizer())
    monkeypatch.setattr("uni_agent.gateway.gateway.ray.util.get_node_ip_address", lambda: ray_address)

    actor = _GatewayActor(config, SequencedBackend([]))

    assert actor._server_address == ray_address.strip("[]")


@pytest.mark.parametrize("address", ["192.0.2.10", "2001:0DB8:0000:0000::10", "::ffff:127.0.0.1"])
def test_actor_environment_overrides_ray_node_address_verbatim(monkeypatch, address):
    monkeypatch.setenv(BIND_ADDRESS_ENV, address)
    monkeypatch.setattr(
        "uni_agent.gateway.gateway.ray.util.get_node_ip_address",
        lambda: pytest.fail("An explicit Gateway HTTP address must not require a Ray address lookup"),
    )

    actor = _GatewayActor(GatewayActorConfig(tokenizer=FakeTokenizer()), SequencedBackend([]))

    assert actor._server_address == address


@pytest.mark.parametrize("actor_address", [None, "198.51.100.10"])
def test_serialized_config_does_not_capture_driver_environment(monkeypatch, actor_address):
    monkeypatch.setenv(BIND_ADDRESS_ENV, "192.0.2.10")
    serialized_config = pickle.dumps(GatewayActorConfig(tokenizer=FakeTokenizer()))
    if actor_address is None:
        monkeypatch.delenv(BIND_ADDRESS_ENV)
    else:
        monkeypatch.setenv(BIND_ADDRESS_ENV, actor_address)
    monkeypatch.setattr("uni_agent.gateway.gateway.ray.util.get_node_ip_address", lambda: "192.0.2.1")

    actor = _GatewayActor(pickle.loads(serialized_config), SequencedBackend([]))

    assert actor._server_address == (actor_address or "192.0.2.1")


def test_each_actor_reads_its_current_environment(monkeypatch):
    config = GatewayActorConfig(tokenizer=FakeTokenizer())
    actors = []
    for address in ("192.0.2.10", "198.51.100.10"):
        monkeypatch.setenv(BIND_ADDRESS_ENV, address)
        actors.append(_GatewayActor(config, SequencedBackend([])))

    assert [actor._server_address for actor in actors] == ["192.0.2.10", "198.51.100.10"]


def test_policies_share_the_actor_environment_address(monkeypatch):
    monkeypatch.setenv(BIND_ADDRESS_ENV, "2001:db8::10")
    configs = {name: GatewayActorConfig(tokenizer=FakeTokenizer()) for name in ("one", "two")}
    backends = {name: SequencedBackend([]) for name in configs}

    actor = _GatewayActor(configs, backends)

    assert actor._server_address == "2001:db8::10"


@pytest.mark.parametrize("address", ["127.0.0.1", "0:0:0:0:0:0:0:1"])
@pytest.mark.parametrize("use_override", [False, True])
@pytest.mark.asyncio
async def test_http_uses_selected_address_and_returns_reachable_session_url(monkeypatch, address, use_override):
    if ":" in address:
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
                probe.bind((address, 0))
        except OSError:
            pytest.skip("IPv6 loopback is unavailable on this host")
    if use_override:
        monkeypatch.setenv(BIND_ADDRESS_ENV, address)
        ray_node_address = "192.0.2.1"  # Deliberately not a local interface address.
    else:
        ray_node_address = f"[{address}]" if ":" in address else address
    monkeypatch.setattr("uni_agent.gateway.gateway.ray.util.get_node_ip_address", lambda: ray_node_address)
    actor = _GatewayActor(GatewayActorConfig(tokenizer=FakeTokenizer()), SequencedBackend(["OK"]))
    # The actor reads its environment once, before the server or sessions start.
    monkeypatch.setenv(BIND_ADDRESS_ENV, "198.51.100.10")
    try:
        await actor.start()
        handle = await actor.create_session("address-probe")
        assert urlsplit(handle.base_url).hostname == address
        if ":" in address:
            assert handle.base_url.startswith(f"http://[{address}]:")
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            response = await client.post(
                handle.base_url + "/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "probe"}]},
            )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "OK"
        assert len(await actor.finalize_session("address-probe")) == 1
    finally:
        await actor.shutdown()
    assert actor._server_task is None
