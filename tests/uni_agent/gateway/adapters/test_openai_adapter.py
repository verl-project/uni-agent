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
