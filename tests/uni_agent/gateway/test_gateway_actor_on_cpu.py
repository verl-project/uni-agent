import asyncio
import json
import time

import httpx
import pytest
import ray

from tests.uni_agent.support import (
    FailingBackend,
    FakeProcessor,
    FakeTokenizer,
    InspectingBackend,
    InspectingSequencedBackend,
    QueuedBackend,
    RejectConcurrentSessionBackend,
    RejectRequestEnvelopeBackend,
    SequencedBackend,
    SingleUseVisionInfoExtractor,
    fake_vision_info_extractor,
)


@pytest.fixture(scope="session")
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.asyncio
async def test_gateway_actor_max_tokens_clamped_to_remaining_response_budget():
    """When a session already has ``response_mask`` tokens accumulated,
    the ``max_tokens`` sampling parameter is clamped to the remaining
    ``response_length`` budget so the backend never receives a request
    that would exceed the session-level length limit."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor
    from uni_agent.gateway.session import TrajectoryBuffer

    actor = _GatewayActor(
        GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            prompt_length=2048,
            response_length=100,
        ),
        InspectingBackend(),
    )
    await actor.start()
    try:
        await actor.create_session("s1")
        actor._sessions["s1"].active_trajectory = TrajectoryBuffer(
            prompt_ids=[1, 2, 3],
            response_ids=[10] * 60,
            response_mask=[1] * 60,
        )

        payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 200}
        actor._sessions["s1"].message_history = list(payload["messages"])
        await actor._handle_chat_completions("s1", payload)

        assert actor._backend.calls[-1]["sampling_params"]["max_tokens"] == 40
    finally:
        await actor.shutdown()


@pytest.mark.asyncio
async def test_gateway_actor_continuation_budget_exhausted_materializes_length_stop():
    """When a continuation request would push the total response tokens past
    ``response_length``, the gateway skips the backend call, commits the
    active trajectory with ``finish_reason="length"``, and returns an
    empty assistant message without incrementing ``completion_tokens``."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor
    from uni_agent.gateway.session import TrajectoryBuffer

    backend = InspectingBackend()
    actor = _GatewayActor(
        GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            response_length=50,
        ),
        backend,
    )
    await actor.start()
    try:
        await actor.create_session("s1")
        session = actor._sessions["s1"]
        session.active_trajectory = TrajectoryBuffer(
            prompt_ids=[1, 2, 3, 4, 5],
            response_ids=[10] * 45,
            response_mask=[1] * 45,
        )
        prefix_messages = [
            {"role": "user", "content": "search"},
            {"role": "assistant", "content": "calling tool"},
        ]
        session.message_history = list(prefix_messages)
        payload = {
            "messages": prefix_messages
            + [
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "x" * 200,
                }
            ]
        }
        backend.calls.clear()

        response = await actor._handle_chat_completions("s1", payload)

        body = json.loads(response.body)
        assert body["choices"][0]["finish_reason"] == "length"
        assert body["choices"][0]["message"] == {"role": "assistant", "content": ""}
        assert body["usage"]["completion_tokens"] == 0
        assert backend.calls == []
        assert session.active_trajectory is None
        assert len(session.trajectories) == 1
        assert session.trajectories[0].extra_fields["finish_reason"] == "length"
        assert "length_truncated" not in session.trajectories[0].extra_fields
        assert "traj_exit_reason" not in session.trajectories[0].extra_fields
    finally:
        await actor.shutdown()


@pytest.mark.asyncio
async def test_backend_value_error_raises_400():
    """Backend ``ValueError`` (e.g. prompt-too-long litellm vLLM errors)
    is forwarded as an HTTP 400 with the original error detail, not a
    generic 500."""
    from fastapi import HTTPException

    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor

    backend = InspectingBackend()
    actor = _GatewayActor(GatewayActorConfig(tokenizer=FakeTokenizer()), backend)
    await actor.start()
    try:
        await actor.create_session("s1")
        backend.next_error = ValueError("Prompt length (123456) exceeds the model's maximum context length (8192).")
        with pytest.raises(HTTPException) as exc_info:
            await actor._handle_chat_completions("s1", {"messages": [{"role": "user", "content": "hi"}]})

        assert exc_info.value.status_code == 400
        assert "exceeds the model's maximum context length" in str(exc_info.value.detail)
    finally:
        await actor.shutdown()


@pytest.mark.asyncio
async def test_unknown_session_raises_404():
    """A chat request targeting a session_id that was never created is rejected
    with HTTP 404 (Not Found)."""
    from fastapi import HTTPException

    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor

    actor = _GatewayActor(GatewayActorConfig(tokenizer=FakeTokenizer()), InspectingBackend())
    await actor.start()
    try:
        with pytest.raises(HTTPException) as exc_info:
            await actor._handle_chat_completions("does-not-exist", {"messages": [{"role": "user", "content": "hi"}]})

        assert exc_info.value.status_code == 404
    finally:
        await actor.shutdown()


@pytest.mark.parametrize(
    ("raw_arguments", "expected_arguments"),
    [
        # Valid JSON string is parsed to a dict so Qwen-style chat templates that
        # iterate with ``|items`` receive the expected type.
        ('{"x": 1}', {"x": 1}),
        # Invalid JSON is preserved as the raw string rather than raising or
        # silently corrupting it.
        ("not json", "not json"),
    ],
)
def test_message_normalization_tool_call_arguments(raw_arguments, expected_arguments):
    """``MessageCodec.normalize_request`` parses valid JSON tool-call arguments
    into a dict and leaves invalid JSON as the original string."""
    from uni_agent.gateway.codec import MessageCodec

    result = MessageCodec(FakeTokenizer()).normalize_request(
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {"name": "f", "arguments": raw_arguments},
                        }
                    ],
                }
            ]
        }
    )["messages"][0]

    assert result["tool_calls"][0]["function"]["arguments"] == expected_arguments


@pytest.mark.asyncio
async def test_request_chat_template_kwargs_forwarded(monkeypatch):
    """Per-request ``chat_template_kwargs`` are forwarded to the chat-template
    call alongside the codec-level defaults, and per-request values take
    precedence over matching codec defaults."""
    import uni_agent.gateway.codec as codec_mod
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor

    actor = _GatewayActor(
        GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            apply_chat_template_kwargs={"enable_thinking": False},
        ),
        InspectingBackend(),
    )
    captured_kwargs = {}
    template_fn_name = "_apply_chat" + "_template"
    original_template = getattr(codec_mod, template_fn_name)

    def _spy(tokenizer, messages, **kwargs):
        captured_kwargs.update(kwargs)
        return original_template(tokenizer, messages, **kwargs)

    monkeypatch.setattr(codec_mod, template_fn_name, _spy)
    await actor.start()
    try:
        await actor.create_session("s1")
        await actor._handle_chat_completions(
            "s1",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": True, "extra_flag": "x"},
            },
        )

        assert captured_kwargs["enable_thinking"] is True
        assert captured_kwargs["extra_flag"] == "x"
    finally:
        await actor.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload_extra", "expected_message_substr"),
    [
        ({"n": 2}, "n=2 is not supported"),
        ({"response_format": {"type": "json_object"}}, "response_format is not supported"),
        ({"tool_choice": "required"}, 'tool_choice="required"'),
        (
            {"tool_choice": {"type": "function", "function": {"name": "foo"}}},
            "tool_choice",
        ),
    ],
)
async def test_unsupported_capabilities_rejected_with_400(payload_extra, expected_message_substr):
    """OpenAI capabilities that the gateway does not support (``n > 1``,
    ``response_format``, ``tool_choice="required"``, and per-function
    ``tool_choice``) are rejected with HTTP 400 before reaching the session
    or backend."""
    from fastapi import HTTPException

    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor

    actor = _GatewayActor(GatewayActorConfig(tokenizer=FakeTokenizer()), InspectingBackend())
    await actor.start()
    try:
        await actor.create_session("s1")
        with pytest.raises(HTTPException) as exc_info:
            await actor._handle_chat_completions(
                "s1", {"messages": [{"role": "user", "content": "hi"}], **payload_extra}
            )

        assert exc_info.value.status_code == 400
        assert expected_message_substr in str(exc_info.value.detail)
    finally:
        await actor.shutdown()


@pytest.mark.asyncio
async def test_stream_true_softly_falls_back_to_non_streaming(caplog):
    """``stream=true`` is not supported; the gateway logs a warning and
    returns a non-streaming response (soft fallback)."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor

    actor = _GatewayActor(GatewayActorConfig(tokenizer=FakeTokenizer()), InspectingBackend())
    await actor.start()
    try:
        await actor.create_session("s1")
        with caplog.at_level("WARNING", logger="gateway"):
            response = await actor._handle_chat_completions(
                "s1", {"messages": [{"role": "user", "content": "hi"}], "stream": True}
            )

        assert response.status_code == 200
        assert any("stream=true" in record.getMessage() for record in caplog.records)
    finally:
        await actor.shutdown()


@pytest.mark.asyncio
async def test_chat_completion_response_includes_created_and_model_fields():
    """Response body carries OpenAI-standard ``created`` and ``model`` fields."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor

    actor = _GatewayActor(GatewayActorConfig(tokenizer=FakeTokenizer()), InspectingBackend())
    await actor.start()
    try:
        await actor.create_session("s-model")
        before = int(time.time())
        response = await actor._handle_chat_completions(
            "s-model",
            {"model": "dummy-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        after = int(time.time())

        body = json.loads(response.body)
        assert body["id"].startswith("chatcmpl-")
        assert body["object"] == "chat.completion"
        assert before <= body["created"] <= after
        assert body["model"] == "dummy-model"
        assert body["choices"][0]["index"] == 0
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]

        await actor.create_session("s-fallback")
        fallback_response = await actor._handle_chat_completions(
            "s-fallback",
            {"messages": [{"role": "user", "content": "hi"}]},
        )
        fallback_body = json.loads(fallback_response.body)
        assert isinstance(fallback_body["created"], int)
        assert fallback_body["model"] == "unknown"
    finally:
        await actor.shutdown()


@pytest.mark.asyncio
async def test_tool_choice_none_skips_tool_injection_and_parser(monkeypatch):
    """When ``tool_choice="none"``, tools are cleared before encoding so the
    chat template does not inject tool-call tokens, and the tool parser is
    not used during decode — the response comes back as plain text."""
    import uni_agent.gateway.codec as codec_mod
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor

    actor = _GatewayActor(
        GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            tool_parser_name="hermes",
        ),
        QueuedBackend(['<tool_call>\n{"name": "foo", "arguments": {}}\n</tool_call>']),
    )
    captured_tools = {}
    template_fn_name = "_apply_chat" + "_template"
    original_template = getattr(codec_mod, template_fn_name)

    def _spy(tokenizer, messages, **kwargs):
        captured_tools["tools"] = kwargs.get("tools")
        return original_template(tokenizer, messages, **kwargs)

    monkeypatch.setattr(codec_mod, template_fn_name, _spy)
    await actor.start()
    try:
        await actor.create_session("s1")
        response = await actor._handle_chat_completions(
            "s1",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "foo", "parameters": {}}}],
                "tool_choice": "none",
            },
        )

        assert response.status_code == 200
        assert captured_tools["tools"] is None
        body = json.loads(response.body)
        assert "tool_calls" not in body["choices"][0]["message"]
    finally:
        await actor.shutdown()


@pytest.mark.asyncio
async def test_gateway_actor_forwards_image_data_on_initial_multimodal_request(ray_runtime):
    """On the first turn of a multimodal session, ``image_data`` extracted
    from the request is forwarded to the backend and recorded in the
    resulting ``Trajectory.multi_modal_data``."""
    from uni_agent.gateway.codec import MessageCodec
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import GatewayActor

    processor = FakeProcessor()
    actor = GatewayActor.remote(
        GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            processor=processor,
            vision_info_extractor=fake_vision_info_extractor,
        ),
        InspectingBackend(),
    )
    ray.get(actor.start.remote())

    session = ray.get(actor.create_session.remote("session-mm-initial"))
    payload = {
        "model": "dummy-model",
        "temperature": 0.25,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "image://a.png"}},
                    {"type": "text", "text": "describe this image"},
                ],
            }
        ],
    }

    normalized = MessageCodec(FakeTokenizer()).normalize_request(payload)
    raw_prompt = processor.apply_chat_template(
        normalized["messages"],
        tokenize=False,
        add_generation_prompt=True,
        tools=normalized["tools"],
    )
    expected_prompt_ids = processor(
        text=[raw_prompt],
        images=["image://a.png"],
        videos=None,
        return_tensors="pt",
        do_sample_frames=False,
    )["input_ids"][0].tolist()

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json=payload,
        )

    trajectories = ray.get(actor.finalize_session.remote("session-mm-initial"))
    ray.get(actor.shutdown.remote())

    assert response.status_code == 200
    backend_request = json.loads(response.json()["choices"][0]["message"]["content"])
    assert backend_request["image_data"] == ["image://a.png"]
    assert backend_request["video_data"] is None
    assert backend_request["prompt_ids"] == expected_prompt_ids
    assert backend_request["sampling_params"] == {"temperature": 0.25}
    assert len(trajectories) == 1
    assert trajectories[0].multi_modal_data == {"images": ["image://a.png"]}


@pytest.mark.asyncio
async def test_gateway_actor_complete_wait_and_finalize(ray_runtime):
    """Full lifecycle: create, chat, complete (with reward_info), wait
    for completion, finalize. Verifies the trajectory carries the reward
    and the response mask is all-1 (no incremental interstitial tokens)."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import GatewayActor

    actor = GatewayActor.remote(GatewayActorConfig(tokenizer=FakeTokenizer()), QueuedBackend(["ANSWER: A"]))
    ray.get(actor.start.remote())

    session = ray.get(actor.create_session.remote("session-0"))
    wait_ref = actor.wait_for_completion.remote("session-0", timeout=2.0)

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "Pick label A"}],
            },
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "ANSWER: A"

        complete = await client.post(
            f"{session.base_url.removesuffix('/v1')}/complete",
            json={"reward_info": {"score": 1.0, "label": "A"}},
        )
        assert complete.status_code == 200

    ray.get(wait_ref)
    trajectories = ray.get(actor.finalize_session.remote("session-0"))
    ray.get(actor.shutdown.remote())

    assert len(trajectories) == 1
    assert trajectories[0].reward_info == {"score": 1.0, "label": "A"}
    assert trajectories[0].response_ids
    assert all(mask == 1 for mask in trajectories[0].response_mask)


@pytest.mark.asyncio
async def test_gateway_actor_continuation_reuses_accumulated_media_context(ray_runtime):
    """On a prefix-continuation turn, the accumulated media from the initial
    request is reused so the backend sees the full media context without
    the gateway re-extracting it from the full message history."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import GatewayActor

    actor = GatewayActor.remote(
        GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            processor=FakeProcessor(),
            vision_info_extractor=SingleUseVisionInfoExtractor(),
        ),
        InspectingBackend(),
    )
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-mm-continuation"))

    initial_message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "image://a.png"}},
            {"type": "text", "text": "describe this image"},
        ],
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [initial_message],
            },
        )
        assert first.status_code == 200
        assistant_message = first.json()["choices"][0]["message"]

        second = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [
                    initial_message,
                    assistant_message,
                    {"role": "user", "content": "follow up"},
                ],
            },
        )

    trajectories = ray.get(actor.finalize_session.remote("session-mm-continuation"))
    ray.get(actor.shutdown.remote())

    assert second.status_code == 200
    first_call = json.loads(first.json()["choices"][0]["message"]["content"])
    second_call = json.loads(second.json()["choices"][0]["message"]["content"])
    assert first_call["image_data"] == ["image://a.png"]
    assert second_call["image_data"] == ["image://a.png"]
    assert len(trajectories) == 1
    assert trajectories[0].multi_modal_data == {"images": ["image://a.png"]}


@pytest.mark.asyncio
async def test_gateway_actor_multimodal_reference_change_splits_trajectory(ray_runtime):
    """When a follow-up request changes the image reference (different URL),
    the prefix no longer matches and the gateway splits the trajectory
    — the old active trajectory is materialized and a new one begins."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import GatewayActor

    actor = GatewayActor.remote(
        GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            processor=FakeProcessor(),
            vision_info_extractor=fake_vision_info_extractor,
        ),
        InspectingBackend(),
    )
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-mm-split"))

    first_payload = {
        "model": "dummy-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "image://a.png"}},
                    {"type": "text", "text": "describe image a"},
                ],
            }
        ],
    }
    second_payload = {
        "model": "dummy-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "image://b.png"}},
                    {"type": "text", "text": "describe image b"},
                ],
            }
        ],
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(f"{session.base_url}/chat/completions", json=first_payload)
        second = await client.post(f"{session.base_url}/chat/completions", json=second_payload)

    trajectories = ray.get(actor.finalize_session.remote("session-mm-split"))
    ray.get(actor.shutdown.remote())

    assert first.status_code == 200
    assert second.status_code == 200
    first_call = json.loads(first.json()["choices"][0]["message"]["content"])
    second_call = json.loads(second.json()["choices"][0]["message"]["content"])
    assert first_call["image_data"] == ["image://a.png"]
    assert second_call["image_data"] == ["image://b.png"]
    assert len(trajectories) == 2


@pytest.mark.asyncio
async def test_gateway_actor_continuation_with_tool_returned_image_appends_media(ray_runtime):
    """When a tool-call continuation brings a new image (e.g. a zoomed crop),
    the new image is appended to the session media accumulator. The full
    ``prompt_ids`` sequence (initial prompt + tool-call tokens + incremental
    prompt) is verified token-by-token."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import GatewayActor
    from verl.utils.chat_template import apply_chat_template, initialize_system_prompt

    processor = FakeProcessor()
    tool_call_text = '<tool_call>\n{"name": "search", "arguments": {"query": "crop"}}\n</tool_call>'
    actor = GatewayActor.remote(
        GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            processor=processor,
            tool_parser_name="hermes",
            vision_info_extractor=fake_vision_info_extractor,
        ),
        InspectingSequencedBackend([tool_call_text, "__inspect__"]),
    )
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-mm-tool-image"))

    tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    initial_message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "image://a.png"}},
            {"type": "text", "text": "find a crop"},
        ],
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": tools,
                "messages": [initial_message],
            },
        )
        assert first.status_code == 200
        assistant_message = first.json()["choices"][0]["message"]
        tool_message = {
            "role": "tool",
            "tool_call_id": assistant_message["tool_calls"][0]["id"],
            "content": [
                {"type": "image_url", "image_url": {"url": "image://tool-b.png"}},
                {"type": "text", "text": "zoomed crop"},
            ],
        }

        second = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": tools,
                "messages": [initial_message, assistant_message, tool_message],
            },
        )

    trajectories = ray.get(actor.finalize_session.remote("session-mm-tool-image"))
    ray.get(actor.shutdown.remote())

    assert second.status_code == 200
    second_call = json.loads(second.json()["choices"][0]["message"]["content"])
    assert second_call["image_data"] == ["image://a.png", "image://tool-b.png"]
    assert len(trajectories) == 1
    assert trajectories[0].multi_modal_data == {
        "images": ["image://a.png", "image://tool-b.png"],
    }

    initial_raw_prompt = apply_chat_template(
        processor,
        [initial_message],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    initial_prompt_ids = processor(
        text=[initial_raw_prompt],
        images=["image://a.png"],
        videos=None,
        return_tensors="pt",
        do_sample_frames=False,
    )["input_ids"][0].tolist()

    incremental_raw_prompt = apply_chat_template(
        processor,
        [tool_message],
        tokenize=False,
        add_generation_prompt=True,
    )
    incremental_prompt_ids = processor(
        text=[incremental_raw_prompt],
        images=["image://tool-b.png"],
        videos=None,
        return_tensors="pt",
        do_sample_frames=False,
    )["input_ids"][0].tolist()
    system_prompt = initialize_system_prompt(processor)
    expected_incremental_ids = incremental_prompt_ids[len(system_prompt) :]
    expected_prompt_ids = initial_prompt_ids + [ord(char) for char in tool_call_text] + expected_incremental_ids
    assert second_call["prompt_ids"] == expected_prompt_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("session_id", "first_payload", "second_payload"),
    [
        # Prefix mismatch: second request has a completely different context.
        (
            "session-prefix-mismatch",
            {
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "first turn"}],
            },
            {
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "replacement context"}],
            },
        ),
        # Tool context change: tool set changes between turns.
        (
            "session-tool-context-change",
            {
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
                "messages": [{"role": "user", "content": "first turn"}],
            },
            {
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
                "messages": [
                    {"role": "user", "content": "first turn"},
                    {"role": "assistant", "content": "FIRST"},
                    {"role": "user", "content": "follow up"},
                ],
            },
        ),
    ],
)
async def test_gateway_actor_context_change_splits_trajectory(ray_runtime, session_id, first_payload, second_payload):
    """When the incoming messages are not a prefix of ``session.message_history``
    (context change or tool-set change), the gateway materializes the active
    trajectory and starts a new one. Two parametrized cases: prefix mismatch
    and tool-context change."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import GatewayActor

    actor = GatewayActor.remote(GatewayActorConfig(tokenizer=FakeTokenizer()), QueuedBackend(["FIRST", "SECOND"]))
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote(session_id))

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(f"{session.base_url}/chat/completions", json=first_payload)
        assert first.status_code == 200
        second = await client.post(f"{session.base_url}/chat/completions", json=second_payload)
        assert second.status_code == 200

    trajectories = ray.get(actor.finalize_session.remote(session_id))
    ray.get(actor.shutdown.remote())

    assert len(trajectories) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend_kwargs", "session_id", "request_extra"),
    [
        # Whitelisted keys (temperature, max_tokens) are forwarded; non-whitelisted
        # keys (presence_penalty) and envelope fields (model, tools, messages) are stripped.
        (
            {
                "backend": RejectRequestEnvelopeBackend(
                    "SAFE",
                    expected_sampling_params={"temperature": 0.25, "top_p": 0.8, "max_tokens": 128},
                ),
                "base_sampling_params": {"temperature": 0.1, "top_p": 0.8, "max_tokens": 64},
                "allowed_request_sampling_param_keys": {"temperature", "max_tokens"},
            },
            "session-envelope-boundary",
            {
                "temperature": 0.25,
                "max_tokens": 128,
                "presence_penalty": 1.5,
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
            },
        ),
        # Non-whitelisted key (top_p) in request is ignored; base_sampling_params used as-is.
        (
            {
                "backend": RejectRequestEnvelopeBackend(
                    "SAFE",
                    expected_sampling_params={"temperature": 0.1, "top_p": 0.9},
                ),
                "base_sampling_params": {"temperature": 0.1, "top_p": 0.9},
                "allowed_request_sampling_param_keys": {"temperature"},
            },
            "session-non-whitelist",
            {"presence_penalty": 1.5},
        ),
    ],
)
async def test_gateway_actor_allowlist_filters_sampling_params(ray_runtime, backend_kwargs, session_id, request_extra):
    """The sampling-param allowlist filters request keys: non-whitelisted keys
    (e.g. ``presence_penalty``) are stripped, and envelope fields (model,
    tools, messages) never leak into sampling params. Base sampling params
    supply defaults when the request omits a key."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import GatewayActor

    backend = backend_kwargs["backend"]
    config_kwargs = {key: value for key, value in backend_kwargs.items() if key != "backend"}
    actor = GatewayActor.remote(GatewayActorConfig(tokenizer=FakeTokenizer(), **config_kwargs), backend)
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote(session_id))

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [{"role": "user", "content": "first turn"}],
                **request_extra,
            },
        )

    ray.get(actor.shutdown.remote())

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_gateway_actor_continuation_preserves_prompt_and_generation_masks(ray_runtime):
    """Token-truth: on a continuation turn, the incremental interstitial tokens
    (tool results, chat-template glue) get ``response_mask=0``, while the
    newly generated assistant tokens get ``response_mask=1``."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import GatewayActor

    actor = GatewayActor.remote(GatewayActorConfig(tokenizer=FakeTokenizer()), QueuedBackend(["FIRST", "SECOND"]))
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-continuation-mask"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [
                    {
                        "role": "user",
                        "content": "first turn",
                    }
                ],
            },
        )
        assert first.status_code == 200

        second = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "messages": [
                    {"role": "user", "content": "first turn"},
                    {"role": "assistant", "content": "FIRST"},
                    {"role": "user", "content": "follow up"},
                ],
            },
        )
        assert second.status_code == 200

    trajectories = ray.get(actor.finalize_session.remote("session-continuation-mask"))
    ray.get(actor.shutdown.remote())

    assert len(trajectories) == 1
    assert 0 in trajectories[0].response_mask
    assert trajectories[0].response_mask[-len("SECOND") :] == [1] * len("SECOND")


@pytest.mark.parametrize(
    ("arguments_a", "arguments_b", "expect_equal"),
    [
        # Valid JSON: a dict and an equivalent JSON string (same keys, different
        # order) canonicalize equal, so tool-argument JSON round-trip drift
        # between turns does not spuriously split the trajectory.
        ({"b": 2, "a": 1}, '{"a": 1, "b": 2}', True),
        # Invalid JSON (unquoted keys): comparison falls back to raw string
        # comparison, which is order-sensitive — the same keys in a different
        # order stay not-equal (contrast with the JSON path above).
        ("{b: 2, a: 1}", "{a: 1, b: 2}", False),
    ],
)
def test_canonicalize_tool_call_arguments_for_prefix_comparison(arguments_a, arguments_b, expect_equal):
    """``MessageCodec.canonicalize_message_for_prefix_comparison`` normalizes
    tool-call arguments so that JSON-equivalent values match, and falls back to
    raw string comparison when the arguments are not valid JSON."""
    from uni_agent.gateway.codec import MessageCodec

    def _message(arguments):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "search", "arguments": arguments}}
            ],
        }

    codec = MessageCodec(FakeTokenizer())
    canonical_a = codec.canonicalize_message_for_prefix_comparison(_message(arguments_a))
    canonical_b = codec.canonicalize_message_for_prefix_comparison(_message(arguments_b))

    assert (canonical_a == canonical_b) is expect_equal


@pytest.mark.asyncio
async def test_gateway_actor_serializes_same_session_concurrent_requests(ray_runtime):
    """Two concurrent requests to the same session are serialized by
    ``generation_lock``, each producing its own trajectory with correct
    response tokens and masks."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import GatewayActor

    actor = GatewayActor.remote(
        GatewayActorConfig(tokenizer=FakeTokenizer()),
        RejectConcurrentSessionBackend(["FIRST", "SECOND"]),
    )
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-concurrent"))

    async with httpx.AsyncClient(timeout=5.0) as client:

        async def send_request():
            return await client.post(
                f"{session.base_url}/chat/completions",
                json={
                    "model": "dummy-model",
                    "messages": [{"role": "user", "content": "same session prompt"}],
                },
            )

        first, second = await asyncio.gather(send_request(), send_request())

    trajectories = ray.get(actor.finalize_session.remote("session-concurrent"))
    ray.get(actor.shutdown.remote())

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(trajectories) == 2
    assert trajectories[0].response_ids == [ord(char) for char in "FIRST"]
    assert trajectories[1].response_ids == [ord(char) for char in "SECOND"]
    assert trajectories[0].response_mask == [1] * len("FIRST")
    assert trajectories[1].response_mask == [1] * len("SECOND")


@pytest.mark.asyncio
async def test_gateway_actor_rejects_chat_after_complete(ray_runtime):
    """After ``/complete`` is called, further chat requests are rejected
    with HTTP 409 (Conflict)."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import GatewayActor

    actor = GatewayActor.remote(GatewayActorConfig(tokenizer=FakeTokenizer()), QueuedBackend(["DONE"]))
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-completed-chat"))
    ray.get(actor.complete_session.remote("session-completed-chat"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "after complete"}]},
        )

    ray.get(actor.shutdown.remote())

    assert response.status_code == 409


@pytest.mark.parametrize(
    ("payload", "detail_fragment"),
    [
        ({"model": "dummy-model", "messages": []}, "messages must be non-empty"),
        (
            {"model": "dummy-model", "messages": [{"role": "user", "name": 123, "content": "hello"}]},
            "message.name must be a string",
        ),
        (
            {"model": "dummy-model", "messages": [{"role": "user", "content": 123}]},
            "Unsupported content type",
        ),
        (
            {
                "model": "dummy-model",
                "messages": [{"role": "assistant", "content": "", "tool_calls": {"id": "call-1"}}],
            },
            "tool_calls must be a list",
        ),
        (
            {
                "model": "dummy-model",
                "tools": {"type": "function"},
                "messages": [{"role": "user", "content": "hello"}],
            },
            "tools must be a list",
        ),
    ],
)
@pytest.mark.asyncio
async def test_gateway_actor_rejects_malformed_requests_with_bad_request(ray_runtime, payload, detail_fragment):
    """Malformed request payloads (empty messages, bad types, invalid
    tool_calls/tools structure) are rejected with HTTP 400 and an
    OpenAI-style error envelope."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import GatewayActor

    actor = GatewayActor.remote(GatewayActorConfig(tokenizer=FakeTokenizer()), QueuedBackend(["DONE"]))
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-validation"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json=payload,
        )

    ray.get(actor.shutdown.remote())

    assert response.status_code == 400
    assert detail_fragment in response.text


@pytest.mark.asyncio
async def test_gateway_actor_backend_failure_does_not_commit_partial_state(ray_runtime):
    """Commit-on-success isolation: when the backend raises an error, the
    session state (trajectories, active_trajectory, message_history) is
    *not* mutated. The session reports zero trajectories and no active
    trajectory after the failure."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import GatewayActor

    actor = GatewayActor.remote(GatewayActorConfig(tokenizer=FakeTokenizer()), FailingBackend("boom"))
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-backend-failure"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{session.base_url}/chat/completions",
            json={"model": "dummy-model", "messages": [{"role": "user", "content": "first turn"}]},
        )

    state = ray.get(actor.get_session_state.remote("session-backend-failure"))
    ray.get(actor.shutdown.remote())

    assert response.status_code == 500
    assert state["num_trajectories"] == 0
    assert state["has_active_trajectory"] is False


@pytest.mark.asyncio
async def test_gateway_actor_backend_failure_after_tool_mismatch_does_not_split(ray_runtime):
    """When the first turn succeeds but the second turn causes a backend
    failure, the first turn's trajectory is still preserved at finalization
    (materialized correctly), and the pre-failure session state shows zero
    trajectories (because the active one was not yet committed)."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import GatewayActor

    actor = GatewayActor.remote(
        GatewayActorConfig(tokenizer=FakeTokenizer()),
        SequencedBackend(["FIRST", RuntimeError("boom")]),
    )
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-failure-mismatch"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        first = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
                "messages": [{"role": "user", "content": "first turn"}],
            },
        )
        assert first.status_code == 200

    async with httpx.AsyncClient(timeout=5.0) as client:
        second = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
                "messages": [
                    {"role": "user", "content": "first turn"},
                    {"role": "assistant", "content": "FIRST"},
                    {"role": "user", "content": "follow up"},
                ],
            },
        )
        assert second.status_code == 500

    state = ray.get(actor.get_session_state.remote("session-failure-mismatch"))
    trajectories = ray.get(actor.finalize_session.remote("session-failure-mismatch"))
    ray.get(actor.shutdown.remote())

    assert state["num_trajectories"] == 0
    assert len(trajectories) == 1
    assert trajectories[0].response_ids == [ord(char) for char in "FIRST"]


@pytest.mark.asyncio
async def test_gateway_actor_tool_call_decode_returns_openai_format(ray_runtime):
    """When tool_parser_name is set and model outputs tool call tokens,
    the HTTP response should contain tool_calls in OpenAI format."""
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import GatewayActor

    tool_call_text = '<tool_call>\n{"name": "search", "arguments": {"query": "weather"}}\n</tool_call>'
    actor = GatewayActor.remote(
        GatewayActorConfig(
            tokenizer=FakeTokenizer(),
            tool_parser_name="hermes",
        ),
        QueuedBackend([tool_call_text, "sunny today"]),
    )
    ray.get(actor.start.remote())
    session = ray.get(actor.create_session.remote("session-tool-call"))

    async with httpx.AsyncClient(timeout=5.0) as client:
        # First request: model returns a tool call
        first = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
                "messages": [{"role": "user", "content": "what is the weather?"}],
            },
        )
        assert first.status_code == 200
        first_data = first.json()
        assert first_data["choices"][0]["finish_reason"] == "tool_calls"
        tool_calls = first_data["choices"][0]["message"].get("tool_calls")
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "search"
        assert tool_calls[0]["type"] == "function"
        assert "id" in tool_calls[0]
        # HTTP response arguments should be a JSON string (OpenAI compatible)
        assert isinstance(tool_calls[0]["function"]["arguments"], str)

        # Second request: agent sends back tool result as continuation
        second = await client.post(
            f"{session.base_url}/chat/completions",
            json={
                "model": "dummy-model",
                "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
                "messages": [
                    {"role": "user", "content": "what is the weather?"},
                    {"role": "assistant", "content": None, "tool_calls": tool_calls},
                    {"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "sunny and warm"},
                ],
            },
        )
        assert second.status_code == 200
        assert second.json()["choices"][0]["message"]["content"] == "sunny today"

    trajectories = ray.get(actor.finalize_session.remote("session-tool-call"))
    ray.get(actor.shutdown.remote())

    assert len(trajectories) == 1
    # Should have both mask=0 (incremental) and mask=1 (model output) tokens
    assert 0 in trajectories[0].response_mask
    assert 1 in trajectories[0].response_mask
