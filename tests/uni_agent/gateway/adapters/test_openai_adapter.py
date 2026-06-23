def test_internal_generation_request_shape():
    from uni_agent.gateway.session.request import InternalGenerationRequest

    assert set(InternalGenerationRequest.__annotations__) == {
        "messages",
        "tools",
        "chat_template_kwargs",
        "sampling_params",
    }

    req: InternalGenerationRequest = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": None,
        "chat_template_kwargs": {},
        "sampling_params": {"max_tokens": 16},
    }
    assert req["messages"][0]["role"] == "user"
    assert req["sampling_params"]["max_tokens"] == 16


def test_openai_to_internal_normalizes_messages_and_sampling():
    from uni_agent.gateway.adapters.openai import (
        OPENAI_ALLOWED_SAMPLING_KEYS,
        openai_to_internal,
    )

    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "max_tokens": 32,
        "temperature": 0.7,
        "stop": ["</s>"],
        "ignored_field": 1,
    }
    req = openai_to_internal(
        payload,
        base_sampling_params={"top_p": 0.9},
        allowed_sampling_keys=OPENAI_ALLOWED_SAMPLING_KEYS,
    )
    assert req["messages"] == [{"role": "user", "content": "hi"}]
    assert req["tools"][0]["function"]["name"] == "f"
    assert req["sampling_params"]["max_tokens"] == 32
    assert req["sampling_params"]["temperature"] == 0.7
    assert req["sampling_params"]["top_p"] == 0.9
    assert req["sampling_params"]["stop"] == ["</s>"]
    assert "ignored_field" not in req["sampling_params"]


def test_openai_to_internal_tool_choice_none_drops_tools():
    from uni_agent.gateway.adapters.openai import OPENAI_ALLOWED_SAMPLING_KEYS, openai_to_internal

    req = openai_to_internal(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": "none",
        },
        base_sampling_params={},
        allowed_sampling_keys=OPENAI_ALLOWED_SAMPLING_KEYS,
    )
    assert req["tools"] is None
