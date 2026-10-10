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
@pytest.mark.parametrize("parser_name", ["qwen3_coder", "other"])
def test_cached_parser_restores_streaming_policy_after_failure(monkeypatch, initial_flag, parser_name):
    from vllm.tool_parsers import ToolParserManager

    instances = []

    class Parser:
        def __init__(self, tokenizer, *, tools):
            self._parser_engine = SimpleNamespace(_stream_arg_deltas=initial_flag)
            instances.append(self)

        def extract_tool_calls(self, text, request):
            expected = False if parser_name == "qwen3_coder" else initial_flag
            assert self._parser_engine._stream_arg_deltas is expected
            if text == "failure":
                raise ValueError("invalid response")
            return SimpleNamespace(tools_called=False, content=text, tool_calls=[])

    monkeypatch.setattr(ToolParserManager, "get_tool_parser", lambda name: Parser)
    codec = MessageCodec(FakeTokenizer())
    with pytest.raises(ValueError, match="invalid response"):
        codec._process_tool_calls_vllm("failure", TOOLS, parser_name)
    assert instances[0]._parser_engine._stream_arg_deltas is initial_flag
    assert codec._process_tool_calls_vllm("recovered", TOOLS, parser_name) == ("recovered", [])
    assert len(instances) == 1
    assert instances[0]._parser_engine._stream_arg_deltas is initial_flag


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "text",
    [
        "<tool_call><function=search><parameter=query>docs</parameter>"
        "<parameter=limit>2</parameter></function></tool_call>",
        "<tool_call><function=search><parameter=query>" + "x<z>" * 512 + "</parameter></function></tool_call>",
    ],
)
def test_runtime_qwen3_complete_result_matches_streaming_enabled_parser(text, monkeypatch):
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
    parser = ToolParserManager.get_tool_parser("qwen3_coder")(tokenizer, tools=tools)
    assert parser._parser_engine._stream_arg_deltas is True
    request = SimpleNamespace(tools=tools, tool_choice="auto", skip_special_tokens=True)
    reference = parser.extract_tool_calls(text, request)
    expected_content = (reference.content or "") if reference.tools_called else text
    expected_calls = [(call.function.name, json.loads(call.function.arguments)) for call in reference.tool_calls]

    codec = MessageCodec(tokenizer)
    partial_conversions = []
    engine_cls = type(parser._parser_engine)
    original_delta = engine_cls._compute_arg_delta

    def checked_delta(engine, idx, raw_delta):
        partial_conversions.append(engine._stream_arg_deltas)
        return original_delta(engine, idx, raw_delta)

    monkeypatch.setattr(engine_cls, "_compute_arg_delta", checked_delta)
    # Reuse the same cached parser for a second response to cover state reset.
    for response in [text, "next response"]:
        content, calls = codec._process_tool_calls_vllm(response, TOOLS, "qwen3_coder")
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
