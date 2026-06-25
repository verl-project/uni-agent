def test_gateway_public_exports_match_package_contracts():
    import uni_agent.gateway.adapters as adapters
    import uni_agent.gateway.gateway as gateway
    import uni_agent.gateway.session as session
    from uni_agent.gateway.adapters.types import (
        AnthropicRequest,
        MalformedRequestError,
        OpenAIChatCompletionRequest,
        OpenAIChatCompletionResponse,
    )
    from uni_agent.gateway.session.types import InternalGenerationRequest

    assert set(adapters.__all__) == {
        "anthropic_build_response",
        "anthropic_error_body",
        "anthropic_stream_response",
        "anthropic_to_internal",
        "MalformedRequestError",
        "openai_build_response",
        "openai_error_body",
        "openai_stream_response",
        "openai_to_internal",
    }
    assert gateway.DEFAULT_ALLOWED_REQUEST_SAMPLING_KEYS == frozenset(
        {"temperature", "top_p", "top_k", "max_tokens", "stop"}
    )
    assert {"messages", "stream", "stop_sequences"} <= set(AnthropicRequest.__annotations__)
    assert issubclass(MalformedRequestError, ValueError)
    assert "stop" in OpenAIChatCompletionRequest.__annotations__
    assert OpenAIChatCompletionResponse.__name__ == "OpenAIChatCompletionResponse"
    assert set(InternalGenerationRequest.__annotations__) == {
        "messages",
        "tools",
        "chat_template_kwargs",
        "sampling_params",
    }
    assert set(session.__all__) == {
        "GatewaySession",
        "InternalGenerationRequest",
        "MessageCodec",
        "SessionHandle",
        "Trajectory",
        "TrajectoryBuffer",
    }
