"""Complete-response parsing must not compute unused streaming argument deltas."""

import json
from types import SimpleNamespace

import pytest

from tests.uni_agent.gateway.test_message_codec_tool_dispatch import TOOLS
from tests.uni_agent.support import FakeTokenizer
from uni_agent.gateway.session.codec import MessageCodec


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("initial_flag", [True, False])
@pytest.mark.parametrize("has_converter", [True, False])
def test_cached_parser_restores_streaming_policy_after_failure(monkeypatch, initial_flag, has_converter):
    from vllm.tool_parsers import ToolParserManager

    instances = []

    class Parser:
        def __init__(self, tokenizer, *, tools):
            self._parser_engine = SimpleNamespace(
                _stream_arg_deltas=initial_flag,
                _arg_converter=(lambda x, y: x) if has_converter else None,
            )
            instances.append(self)

        def extract_tool_calls(self, text, request):
            expected = False if has_converter else initial_flag
            assert self._parser_engine._stream_arg_deltas is expected
            if text == "failure":
                raise ValueError("invalid response")
            return SimpleNamespace(tools_called=False, content=text, tool_calls=[])

    monkeypatch.setattr(ToolParserManager, "get_tool_parser", lambda name: Parser)
    codec = MessageCodec(FakeTokenizer())
    with pytest.raises(ValueError, match="invalid response"):
        codec._process_tool_calls_vllm("failure", TOOLS, "custom")
    assert instances[0]._parser_engine._stream_arg_deltas is initial_flag
    assert codec._process_tool_calls_vllm("recovered", TOOLS, "custom") == ("recovered", [])
    assert len(instances) == 1
    assert instances[0]._parser_engine._stream_arg_deltas is initial_flag


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "engine",
    [None, SimpleNamespace(_stream_arg_deltas=True), SimpleNamespace(_arg_converter=lambda x, y: x)],
    ids=["legacy-parser", "missing-converter", "missing-delta-flag"],
)
def test_parser_without_delta_capabilities_uses_standard_path(monkeypatch, engine):
    from vllm.tool_parsers import ToolParserManager

    initial_state = vars(engine).copy() if engine is not None else None

    class Parser:
        def __init__(self, tokenizer, *, tools):
            if engine is not None:
                self._parser_engine = engine

        def extract_tool_calls(self, text, request):
            if engine is not None:
                assert vars(engine) == initial_state
            return SimpleNamespace(tools_called=False, content=text, tool_calls=[])

    monkeypatch.setattr(ToolParserManager, "get_tool_parser", lambda name: Parser)
    assert MessageCodec(FakeTokenizer())._process_tool_calls_vllm("plain text", TOOLS, "custom") == ("plain text", [])


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("parser_name", ["qwen3_coder", "qwen3_xml", "qwen3"])
@pytest.mark.parametrize(
    "text",
    [
        "<tool_call><function=search><parameter=query>docs</parameter>"
        "<parameter=limit>2</parameter></function></tool_call>",
        "<tool_call><function=search><parameter=query>" + "x<z>" * 512 + "</parameter></function></tool_call>",
    ],
)
async def test_runtime_qwen3_complete_result_matches_streaming_enabled_parser(text, parser_name, monkeypatch):
    pytest.importorskip("vllm.parser.qwen3")
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
    from vllm.tool_parsers import ToolParserManager

    class Tokenizer(FakeTokenizer):
        def get_vocab(self):
            return {"<tool_call>": 1, "</tool_call>": 2}

        def get_added_vocab(self):
            return {}

    tokenizer = Tokenizer()
    tools = [ChatCompletionToolsParam(**tool) for tool in TOOLS]
    vllm_name = "qwen3_xml" if parser_name == "qwen3" else parser_name
    parser = ToolParserManager.get_tool_parser(vllm_name)(tokenizer, tools=tools)
    assert parser._parser_engine._stream_arg_deltas is True
    request = SimpleNamespace(tools=tools, tool_choice="auto", skip_special_tokens=True)
    reference = parser.extract_tool_calls(text, request)
    assert reference.tools_called
    expected_content = (reference.content or "") if reference.tools_called else text
    expected_calls = [(call.function.name, json.loads(call.function.arguments)) for call in reference.tool_calls]

    codec = MessageCodec(tokenizer, rollout_backend="vllm")
    partial_conversions = []
    engine_cls = type(parser._parser_engine)
    original_delta = engine_cls._compute_arg_delta

    def checked_delta(engine, idx, raw_delta):
        partial_conversions.append(engine._stream_arg_deltas)
        return original_delta(engine, idx, raw_delta)

    monkeypatch.setattr(engine_cls, "_compute_arg_delta", checked_delta)
    # Reuse the same cached parser for a second response to cover state reset.
    for response in [text, "next response"]:
        content, calls = await codec._extract_tool_calls(tokenizer.encode(response), TOOLS, parser_name)
        if response == text:
            assert content == expected_content
            assert [(call.name, json.loads(call.arguments)) for call in calls] == expected_calls
        else:
            assert (content, calls) == ("next response", [])
    engine = next(iter(codec._tool_parser_cache.values()))._parser_engine
    assert engine._stream_arg_deltas is True
    assert not any(partial_conversions)
    if "x<z>" in text:
        assert partial_conversions  # Positive control: the pathological path ran.
