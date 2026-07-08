import pytest

from uni_agent.gateway.config import GatewayActorConfig


def test_gateway_actor_config_accepts_real_bool_m2_flags():
    config = GatewayActorConfig(
        tokenizer=object(),
        enable_parallel_session_generation=True,
    )

    assert config.enable_parallel_session_generation is True


@pytest.mark.parametrize(
    ("flag_name", "bad_value"),
    [
        ("enable_parallel_session_generation", "true"),
        ("enable_parallel_session_generation", 1),
    ],
)
def test_gateway_actor_config_rejects_non_bool_m2_flags(flag_name, bad_value):
    with pytest.raises(ValueError, match=f"{flag_name} must be a bool"):
        GatewayActorConfig(tokenizer=object(), **{flag_name: bad_value})
