from dataclasses import FrozenInstanceError

import pytest

from uni_agent.gateway.kv_offload.hints import AgentHintConfig

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0, -1, True, "300", None])
def test_config_rejects_invalid_lease_duration(value):
    with pytest.raises(ValueError, match="must be finite and positive"):
        AgentHintConfig(active_lease_seconds=value)


@pytest.mark.parametrize("value", [1, 0.5, 300.0])
def test_config_accepts_finite_positive_lease_duration(value):
    assert AgentHintConfig(active_lease_seconds=value).active_lease_seconds == value


@pytest.mark.parametrize("value", [True, False, 1.9, "50", -1, 101, None])
def test_config_rejects_invalid_admission_priority(value):
    with pytest.raises(ValueError, match="min_offload_priority"):
        AgentHintConfig(min_offload_priority=value)


@pytest.mark.parametrize("value", ["false", "0", 0, 1, None])
def test_config_rejects_invalid_admission_boolean(value):
    with pytest.raises(ValueError, match="admit_without_hint"):
        AgentHintConfig(admit_without_hint=value)


@pytest.mark.parametrize("field", ["priority_mode", "priority", "tool_priority", "enabled"])
def test_removed_static_and_gateway_options_are_rejected(field):
    with pytest.raises(ValueError, match="Unsupported.*scoring is always dynamic"):
        AgentHintConfig.from_dict({field: 50})


@pytest.mark.parametrize("raw", [None, [], True, "{}"])
def test_config_requires_object(raw):
    with pytest.raises(ValueError, match="agent_hint_config must be an object"):
        AgentHintConfig.from_dict(raw)


def test_config_is_frozen_and_defaults_are_available():
    config = AgentHintConfig.from_dict({})
    assert config == AgentHintConfig(active_lease_seconds=300.0, min_offload_priority=0, admit_without_hint=True)
    with pytest.raises(FrozenInstanceError):
        config.active_lease_seconds = 60
