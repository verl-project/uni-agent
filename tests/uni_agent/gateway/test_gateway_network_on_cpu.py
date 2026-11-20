"""Gateway address selection and real HTTP listening, without model downloads."""

import socket
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import pytest

from tests.uni_agent.support import FakeTokenizer, SequencedBackend
from uni_agent.gateway.config import GatewayActorConfig

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.mark.parametrize("ray_address", ["192.0.2.1", "[2001:db8::1]"])
def test_default_address_tracks_the_actor_node(ray_address):
    assert GatewayActorConfig(tokenizer=None).resolve_bind_address(ray_address) == ray_address.strip("[]")


@pytest.mark.parametrize("address", ["127.0.0.1", "::1", "2001:db8::10"])
def test_explicit_address_overrides_ray_node_address(address):
    config = GatewayActorConfig(tokenizer=None, bind_address=address)
    assert config.resolve_bind_address("192.0.2.1") == address


def test_node_mapping_selects_each_actor_address_and_copies_input():
    addresses = {"192.0.2.1": "2001:db8::10", "2001:0db8:0:0::2": "198.51.100.11"}
    config = GatewayActorConfig(tokenizer=None, bind_address_by_node=addresses)
    addresses["192.0.2.1"] = "198.51.100.99"

    assert config.resolve_bind_address("192.0.2.1") == "2001:db8::10"
    assert config.resolve_bind_address("[2001:db8::2]") == "198.51.100.11"
    with pytest.raises(ValueError, match="no address for Ray node 192.0.2.3"):
        config.resolve_bind_address("192.0.2.3")


def test_node_mapping_rejects_duplicate_normalized_keys():
    with pytest.raises(ValueError, match="duplicate normalized node IP"):
        GatewayActorConfig(
            tokenizer=None,
            bind_address_by_node={"2001:0db8:0:0::1": "192.0.2.10", "2001:db8::1": "192.0.2.11"},
        )


def test_single_address_and_node_mapping_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        GatewayActorConfig(tokenizer=None, bind_address="127.0.0.1", bind_address_by_node={"192.0.2.1": "::1"})


@pytest.mark.parametrize("mapping", [{}, [], "192.0.2.1", 1])
def test_node_mapping_requires_nonempty_mapping(mapping):
    with pytest.raises(ValueError, match="nonempty mapping"):
        GatewayActorConfig(tokenizer=None, bind_address_by_node=mapping)


@pytest.mark.parametrize(
    "value", ["", "gateway.example", "0.0.0.0", "::", "224.0.0.1", "ff02::1", "fe80::1", "fe80::1%eth0", 1]
)
@pytest.mark.parametrize("field", ["bind_address", "node_key", "node_value"])
def test_rejects_addresses_unsuitable_for_listener_urls(value, field):
    kwargs = {
        "bind_address": {"bind_address": value},
        "node_key": {"bind_address_by_node": {value: "127.0.0.1"}},
        "node_value": {"bind_address_by_node": {"192.0.2.1": value}},
    }[field]
    with pytest.raises(ValueError, match="concrete, unscoped IPv4 or IPv6"):
        GatewayActorConfig(tokenizer=None, **kwargs)


def test_policies_sharing_an_actor_must_agree_on_address(monkeypatch):
    from uni_agent.gateway.gateway import _GatewayActor

    monkeypatch.setattr("uni_agent.gateway.gateway.ray.util.get_node_ip_address", lambda: "192.0.2.1")
    configs = {
        "one": GatewayActorConfig(tokenizer=FakeTokenizer(), bind_address="127.0.0.1"),
        "two": GatewayActorConfig(tokenizer=FakeTokenizer(), bind_address="::1"),
    }
    with pytest.raises(ValueError, match="same bind address"):
        _GatewayActor(configs, {"one": SequencedBackend([]), "two": SequencedBackend([])})


def test_policies_accept_equivalent_default_and_explicit_ipv6_addresses(monkeypatch):
    from uni_agent.gateway.gateway import _GatewayActor

    node_address = "2001:0db8:0:0::1"
    monkeypatch.setattr("uni_agent.gateway.gateway.ray.util.get_node_ip_address", lambda: node_address)
    configs = {
        "one": GatewayActorConfig(tokenizer=FakeTokenizer()),
        "two": GatewayActorConfig(tokenizer=FakeTokenizer(), bind_address="2001:db8::1"),
    }
    actor = _GatewayActor(configs, {"one": SequencedBackend([]), "two": SequencedBackend([])})
    assert actor._server_address == node_address


@pytest.mark.parametrize(
    "options, expected",
    [
        ({}, "192.0.2.1"),
        ({"gateway_bind_address": "127.0.0.1"}, "127.0.0.1"),
        ({"gateway_bind_address_by_node": {"192.0.2.1": "2001:db8::1"}}, "2001:db8::1"),
    ],
)
def test_multi_policy_entry_applies_shared_network_options(monkeypatch, options, expected):
    from omegaconf import OmegaConf

    from examples.multi_policy.framework import multi_agent_entry

    model = SimpleNamespace(tokenizer=FakeTokenizer(), processor=None, hf_config=SimpleNamespace(model_type=None))
    rollout = SimpleNamespace(
        name="vllm", prompt_length=128, response_length=64, multi_turn=SimpleNamespace(format=None)
    )
    monkeypatch.setattr(
        multi_agent_entry, "omega_conf_to_dataclass", lambda cfg: model if cfg.kind == "model" else rollout
    )
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"custom": {"agent_framework": options}}}})
    policy = OmegaConf.create(
        {"data": {}, "actor_rollout_ref": {"model": {"kind": "model"}, "rollout": {"kind": "rollout"}}}
    )
    policies = multi_agent_entry.build_policy_gateway_configs(config, {"one": policy, "two": policy})

    assert {item.resolve_bind_address("192.0.2.1") for item in policies.values()} == {expected}


@pytest.mark.parametrize("address", ["127.0.0.1", "::1"])
@pytest.mark.parametrize("use_mapping", [False, True])
@pytest.mark.asyncio
async def test_http_uses_selected_address_and_returns_reachable_session_url(monkeypatch, address, use_mapping):
    from uni_agent.gateway.gateway import _GatewayActor

    if address == "::1":
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
                probe.bind((address, 0))
        except OSError:
            pytest.skip("IPv6 loopback is unavailable on this host")
    ray_node_address = "192.0.2.1"  # Deliberately not a local interface address.
    monkeypatch.setattr("uni_agent.gateway.gateway.ray.util.get_node_ip_address", lambda: ray_node_address)
    kwargs = {"bind_address_by_node": {ray_node_address: address}} if use_mapping else {"bind_address": address}
    actor = _GatewayActor(GatewayActorConfig(tokenizer=FakeTokenizer(), **kwargs), SequencedBackend(["OK"]))
    await actor.start()
    try:
        handle = await actor.create_session("address-probe")
        assert urlsplit(handle.base_url).hostname == address
        if address == "::1":
            assert handle.base_url.startswith("http://[::1]:")
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
