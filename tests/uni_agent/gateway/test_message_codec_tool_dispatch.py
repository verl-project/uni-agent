import json
from types import SimpleNamespace

import pytest

from tests.uni_agent.support import FakeTokenizer

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search docs",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    }
]


def _ids(text: str) -> list[int]:
    return [ord(char) for char in text]


@pytest.mark.cpu
@pytest.mark.level0
def test_qwen_vllm_parser_uses_tool_schema_for_argument_types():
    from uni_agent.gateway.session.codec import MessageCodec

    class QwenTokenizer(FakeTokenizer):
        def get_vocab(self):
            return {"<tool_call>": 1, "</tool_call>": 2}

    text = (
        "<tool_call>\n"
        "<function=search>\n"
        "<parameter=query>docs</parameter>\n"
        "<parameter=limit>2</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    content, calls = MessageCodec(QwenTokenizer())._process_tool_calls_vllm(text, TOOLS, "qwen3_coder")

    assert content == ""
    assert json.loads(calls[0].arguments) == {"query": "docs", "limit": 2}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "constructor_accepts_tools",
    [False, True],
    ids=["tokenizer-only", "tokenizer-and-tools"],
)
def test_vllm_parser_supports_tool_schema_constructor_contracts(monkeypatch, constructor_accepts_tools):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
    from vllm.tool_parsers import ToolParserManager

    from uni_agent.gateway.session.codec import MessageCodec

    seen = {}

    class ParserBase:
        def extract_tool_calls(self, text, request):
            seen["request"] = request
            return SimpleNamespace(
                tools_called=True,
                content="visible",
                tool_calls=[SimpleNamespace(function=SimpleNamespace(name="search", arguments='{"query":"x"}'))],
            )

    class ParserWithoutConstructorTools(ParserBase):
        def __init__(self, tokenizer):
            seen["tokenizer"] = tokenizer

    class ParserWithConstructorTools(ParserBase):
        def __init__(self, tokenizer, *, tools):
            seen["tokenizer"] = tokenizer
            seen["tools"] = tools

    parser_cls = ParserWithConstructorTools if constructor_accepts_tools else ParserWithoutConstructorTools

    monkeypatch.setattr(
        ToolParserManager,
        "get_tool_parser",
        classmethod(lambda cls, name: parser_cls),
    )

    tokenizer = FakeTokenizer()
    content, calls = MessageCodec(tokenizer)._process_tool_calls_vllm("raw", TOOLS, "qwen3_coder")

    assert content == "visible"
    assert calls[0].name == "search"
    assert seen["tokenizer"] is tokenizer
    assert len(seen["request"].tools) == 1
    assert isinstance(seen["request"].tools[0], ChatCompletionToolsParam)
    if constructor_accepts_tools:
        assert seen["tools"] is seen["request"].tools
    else:
        assert "tools" not in seen


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_tool_call_dispatch_uses_sglang_for_sglang_rollout(monkeypatch):
    from uni_agent.gateway.session.codec import MessageCodec

    seen = {}

    def fake_sglang(text, tools, parser_name):
        seen["sglang"] = (text, tools, parser_name)
        return "visible", [SimpleNamespace(name="search", arguments='{"query":"x"}')]

    def fail_vllm(*args, **kwargs):
        raise AssertionError("vLLM should not run when SGLang succeeds")

    async def fail_verl(*args, **kwargs):
        raise AssertionError("verl should not run when an engine succeeds")

    codec = MessageCodec(FakeTokenizer(), rollout_backend="sglang")
    monkeypatch.setattr(codec, "_process_tool_calls_sglang", fake_sglang)
    monkeypatch.setattr(codec, "_process_tool_calls_vllm", fail_vllm)
    monkeypatch.setattr(codec, "_process_tool_calls_verl", fail_verl)

    content, calls = await codec._extract_tool_calls(_ids("raw"), TOOLS, "hermes")

    assert content == "visible"
    assert calls[0].name == "search"
    assert seen["sglang"] == ("raw", TOOLS, "hermes")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_tool_call_dispatch_uses_vllm_for_vllm_rollout_with_name_mapping(monkeypatch):
    from uni_agent.gateway.session.codec import MessageCodec

    seen = {}

    def fail_sglang(*args, **kwargs):
        raise AssertionError("SGLang should not run for a vLLM rollout")

    def fake_vllm(text, tools, parser_name):
        seen["vllm"] = (text, tools, parser_name)
        return "", [SimpleNamespace(name="search", arguments='{"query":"x"}')]

    async def fail_verl(*args, **kwargs):
        raise AssertionError("verl should not run when an engine succeeds")

    codec = MessageCodec(FakeTokenizer(), rollout_backend="vllm")
    monkeypatch.setattr(codec, "_process_tool_calls_sglang", fail_sglang)
    monkeypatch.setattr(codec, "_process_tool_calls_vllm", fake_vllm)
    monkeypatch.setattr(codec, "_process_tool_calls_verl", fail_verl)

    content, calls = await codec._extract_tool_calls(_ids("raw"), TOOLS, "qwen25")

    assert content == ""
    assert calls[0].arguments == '{"query":"x"}'
    assert seen["vllm"] == ("raw", TOOLS, "qwen3_xml")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_tool_call_dispatch_uses_verl_for_other_rollout_backends(monkeypatch):
    from uni_agent.gateway.session.codec import MessageCodec

    seen = {}

    def fail_engine(*args, **kwargs):
        raise AssertionError("engine parser should not run for another rollout backend")

    async def fake_verl(response_ids, tools, parser_name):
        seen["verl"] = (response_ids, tools, parser_name)
        return "thinking", [SimpleNamespace(name="search", arguments='{"query":"docs"}')]

    codec = MessageCodec(FakeTokenizer(), rollout_backend="hf")
    monkeypatch.setattr(codec, "_process_tool_calls_sglang", fail_engine)
    monkeypatch.setattr(codec, "_process_tool_calls_vllm", fail_engine)
    monkeypatch.setattr(codec, "_process_tool_calls_verl", fake_verl)

    text = 'thinking\n<tool_call>\n{"name": "search", "arguments": {"query": "docs"}}\n</tool_call>'
    content, calls = await codec._extract_tool_calls(_ids(text), TOOLS, "hermes")

    assert content == "thinking"
    assert calls[0].name == "search"
    assert seen["verl"] == (_ids(text), TOOLS, "hermes")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_tool_call_dispatch_surfaces_selected_parser_failure_without_fallback(monkeypatch):
    from uni_agent.gateway.session.codec import MessageCodec

    def broken_vllm(*args, **kwargs):
        raise ModuleNotFoundError("vllm")

    async def fail_verl(*args, **kwargs):
        raise AssertionError("verl must not hide a selected vLLM parser failure")

    codec = MessageCodec(FakeTokenizer(), rollout_backend="vllm")
    monkeypatch.setattr(codec, "_process_tool_calls_vllm", broken_vllm)
    monkeypatch.setattr(codec, "_process_tool_calls_verl", fail_verl)

    with pytest.raises(RuntimeError, match="vllm tool parser 'hermes' failed") as exc_info:
        await codec._extract_tool_calls(_ids("plain text"), TOOLS, "hermes")

    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_selected_parser_empty_result_is_not_an_error(monkeypatch):
    from uni_agent.gateway.session.codec import MessageCodec

    codec = MessageCodec(FakeTokenizer(), rollout_backend="sglang")
    monkeypatch.setattr(
        codec,
        "_process_tool_calls_sglang",
        lambda text, tools, parser_name: (text, []),
    )

    async def fail_verl(*args, **kwargs):
        raise AssertionError("verl should not run when an engine already answered")

    def fail_vllm(*args, **kwargs):
        raise AssertionError("vLLM should not run when SGLang already answered")

    monkeypatch.setattr(codec, "_process_tool_calls_vllm", fail_vllm)
    monkeypatch.setattr(codec, "_process_tool_calls_verl", fail_verl)

    text = '<tool_call>\n{"name": "search", "arguments": {"query": "docs"}}\n</tool_call>'
    content, calls = await codec._extract_tool_calls(_ids(text), TOOLS, "hermes")

    assert content == text
    assert calls == []


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_verl_parser_parses_hermes_envelope():
    from uni_agent.gateway.session.codec import MessageCodec

    text = 'thinking\n<tool_call>\n{"name": "search", "arguments": {"query": "docs", "limit": 2}}\n</tool_call>'
    content, calls = await MessageCodec(FakeTokenizer())._process_tool_calls_verl(_ids(text), TOOLS, "hermes")

    assert content == "thinking\n"
    assert calls[0].name == "search"
    assert json.loads(calls[0].arguments) == {"query": "docs", "limit": 2}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_decode_response_uses_gateway_dispatcher_for_tool_calls(monkeypatch):
    from uni_agent.gateway.session.codec import MessageCodec

    seen = {}

    async def fake_dispatch(response_ids, tools, parser_name):
        seen["dispatch"] = (response_ids, tools, parser_name)
        return "", [SimpleNamespace(name="search", arguments='{"query":"weather"}')]

    tokenizer = FakeTokenizer()
    codec = MessageCodec(tokenizer, tool_parser_name="qwen3_xml")
    monkeypatch.setattr(codec, "_extract_tool_calls", fake_dispatch)
    response_ids = [ord(char) for char in "<tool_call>ignored</tool_call>"]
    message, finish_reason = await codec.decode_response(
        response_ids,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "search docs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {"anyOf": [{"const": "file"}, {"type": "string"}]},
                        },
                    },
                },
            }
        ],
        stop_reason="stop",
    )

    assert finish_reason == "tool_calls"
    assert message["content"] == ""
    assert message["tool_calls"][0]["type"] == "function"
    assert message["tool_calls"][0]["function"] == {"name": "search", "arguments": '{"query":"weather"}'}
    assert seen["dispatch"][0] == response_ids
    assert seen["dispatch"][2] == "qwen3_xml"
