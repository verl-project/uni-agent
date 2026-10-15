import copy

import pytest

from tests.uni_agent.support import FakeTokenizer, SequencedBackend
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.gateway import _GatewayActor


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("eos", [248046, [248046, 248047], (248046, 248047), None])
@pytest.mark.parametrize(
    "engine,skip_tokenizer_init,ignore_eos,adds_stops",
    [
        ("sglang", True, False, True),
        ("sglang", True, True, False),
        ("sglang", False, False, False),
        ("vllm", True, False, False),
        (None, True, False, False),
    ],
)
async def test_gateway_supplies_tokenizer_stops_only_for_sglang_token_input(
    eos, engine, skip_tokenizer_init, ignore_eos, adds_stops
):
    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = eos
    tokenizer.additional_stop_token_ids = [18]
    backend = SequencedBackend(["done"])
    actor = _GatewayActor(
        GatewayActorConfig(tokenizer=tokenizer, rollout_backend=engine, skip_tokenizer_init=skip_tokenizer_init),
        backend,
    )
    actor._server_base_url = "http://test"
    params = {"stop_token_ids": [17], "stop": ["user-stop"], "ignore_eos": ignore_eos}
    original = copy.deepcopy(params)
    await actor.create_session("stops", sampling_params=params)
    await actor._handle_openai_chat_completions(
        "stops", {"model": "test", "messages": [{"role": "user", "content": "hello"}]}
    )

    actual = backend.calls[0]["sampling_params"]
    expected = {17}
    if adds_stops:
        expected.add(18)
        expected.update([eos] if isinstance(eos, int) else eos or [])
    assert set(actual["stop_token_ids"]) == expected
    assert actual["stop"] == ["user-stop"]
    assert params == original
    assert actor._sessions["stops"].sampling_params == original


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_request_ignore_eos_does_not_retain_implicit_session_stops():
    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = 248046
    backend = SequencedBackend(["done"])
    actor = _GatewayActor(
        GatewayActorConfig(
            tokenizer=tokenizer,
            rollout_backend="sglang",
            skip_tokenizer_init=True,
            allowed_request_sampling_param_keys={"ignore_eos"},
        ),
        backend,
    )
    actor._server_base_url = "http://test"
    await actor.create_session("ignore")
    await actor._handle_openai_chat_completions(
        "ignore",
        {"model": "test", "messages": [{"role": "user", "content": "hello"}], "ignore_eos": True},
    )

    assert backend.calls[0]["sampling_params"]["ignore_eos"] is True
    assert "stop_token_ids" not in backend.calls[0]["sampling_params"]
    assert "stop" not in backend.calls[0]["sampling_params"]


@pytest.mark.cpu
@pytest.mark.level0
def test_gateway_config_preserves_existing_positional_arguments():
    config = GatewayActorConfig(FakeTokenizer(), None, None, "vllm", False)
    assert config.enable_tool_parser_cache is False
    assert config.skip_tokenizer_init is False
    config = GatewayActorConfig(FakeTokenizer(), None, None, "sglang", False, skip_tokenizer_init=True)
    assert config.enable_tool_parser_cache is False
    assert config.skip_tokenizer_init is True
