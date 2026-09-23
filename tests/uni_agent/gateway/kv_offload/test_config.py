import pytest

from uni_agent.gateway.config import GatewayActorConfig, KVCacheHintConfig

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0, -1, True, "300", None])
def test_config_rejects_invalid_lease_duration(value):
    with pytest.raises(ValueError, match="must be finite and positive"):
        KVCacheHintConfig(lease_seconds=value)


@pytest.mark.parametrize("value", [1, 0.5, 300.0])
def test_config_accepts_finite_positive_lease_duration(value):
    config = KVCacheHintConfig(lease_seconds=value)
    assert config.lease_seconds == value


@pytest.mark.parametrize("value", [1, "true", None])
def test_hint_config_rejects_non_boolean_enabled(value):
    with pytest.raises(ValueError, match="enabled must be a bool"):
        KVCacheHintConfig(enabled=value)


@pytest.mark.parametrize("value", ["adaptive", None, []])
def test_hint_config_rejects_invalid_mode(value):
    with pytest.raises(ValueError, match="priority_mode"):
        KVCacheHintConfig(priority_mode=value)


@pytest.mark.parametrize("field", ["priority", "tool_priority"])
@pytest.mark.parametrize("value", [-1, 101, True, "50", 1.5])
def test_hint_config_rejects_invalid_priority(field, value):
    with pytest.raises(ValueError, match="must be an integer"):
        KVCacheHintConfig(**{field: value})


def test_gateway_uses_immutable_hint_config():
    from dataclasses import FrozenInstanceError

    config = GatewayActorConfig(tokenizer=None)
    assert config.kv_cache_offload_config == KVCacheHintConfig()
    assert not config.kv_cache_offload_config.enabled
    with pytest.raises(FrozenInstanceError):
        config.kv_cache_offload_config.priority = 60
    hints = KVCacheHintConfig(enabled=True, priority_mode="dynamic", lease_seconds=120, priority=40)
    assert GatewayActorConfig(tokenizer=None, kv_cache_offload_config=hints).kv_cache_offload_config is hints


@pytest.mark.parametrize("value", [None, {}, True])
def test_gateway_rejects_untyped_hint_config(value):
    with pytest.raises(ValueError, match="must be a KVCacheHintConfig"):
        GatewayActorConfig(tokenizer=None, kv_cache_offload_config=value)
