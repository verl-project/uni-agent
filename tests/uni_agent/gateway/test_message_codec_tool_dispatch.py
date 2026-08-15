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


def test_qwen_vllm_parser_uses_tool_schema_for_argument_types():
    import uni_agent.gateway.session.codec as codec_mod

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
    content, calls = codec_mod._process_tool_calls_vllm(text, TOOLS, "qwen3_coder", QwenTokenizer())

    assert content == ""
    assert json.loads(calls[0].arguments) == {"query": "docs", "limit": 2}


@pytest.mark.parametrize(
    "constructor_accepts_tools",
    [False, True],
    ids=["tokenizer-only", "tokenizer-and-tools"],
)
def test_vllm_parser_supports_tool_schema_constructor_contracts(monkeypatch, constructor_accepts_tools):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
    from vllm.tool_parsers import ToolParserManager

    import uni_agent.gateway.session.codec as codec_mod

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
    content, calls = codec_mod._process_tool_calls_vllm("raw", TOOLS, "qwen3_coder", tokenizer)

    assert content == "visible"
    assert calls[0].name == "search"
    assert seen["tokenizer"] is tokenizer
    assert len(seen["request"].tools) == 1
    assert isinstance(seen["request"].tools[0], ChatCompletionToolsParam)
    if constructor_accepts_tools:
        assert seen["tools"] is seen["request"].tools
    else:
        assert "tools" not in seen


@pytest.mark.asyncio
async def test_tool_call_dispatch_prefers_sglang(monkeypatch):
    import uni_agent.gateway.session.codec as codec_mod

    seen = {}

    def fake_sglang(text, tools, parser_name):
        seen["sglang"] = (text, tools, parser_name)
        return "visible", [SimpleNamespace(name="search", arguments='{"query":"x"}')]

    def fail_vllm(*args, **kwargs):
        raise AssertionError("vLLM should not run when SGLang succeeds")

    async def fail_verl(*args, **kwargs):
        raise AssertionError("verl should not run when an engine succeeds")

    monkeypatch.setattr(codec_mod, "_process_tool_calls_sglang", fake_sglang, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_vllm", fail_vllm, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_verl", fail_verl, raising=False)

    content, calls = await codec_mod._extract_tool_calls(_ids("raw"), TOOLS, "hermes", FakeTokenizer())

    assert content == "visible"
    assert calls[0].name == "search"
    assert seen["sglang"] == ("raw", TOOLS, "hermes")


@pytest.mark.asyncio
async def test_tool_call_dispatch_falls_back_to_vllm_with_name_mapping(monkeypatch):
    import uni_agent.gateway.session.codec as codec_mod

    seen = {}

    def missing_sglang(*args, **kwargs):
        raise ModuleNotFoundError("sglang")

    def fake_vllm(text, tools, parser_name, tokenizer):
        seen["vllm"] = (text, tools, parser_name, tokenizer)
        return "", [SimpleNamespace(name="search", arguments='{"query":"x"}')]

    async def fail_verl(*args, **kwargs):
        raise AssertionError("verl should not run when an engine succeeds")

    monkeypatch.setattr(codec_mod, "_process_tool_calls_sglang", missing_sglang, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_vllm", fake_vllm, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_verl", fail_verl, raising=False)

    tokenizer = FakeTokenizer()
    content, calls = await codec_mod._extract_tool_calls(_ids("raw"), TOOLS, "qwen25", tokenizer)

    assert content == ""
    assert calls[0].arguments == '{"query":"x"}'
    assert seen["vllm"] == ("raw", TOOLS, "qwen3_xml", tokenizer)


@pytest.mark.asyncio
async def test_tool_call_dispatch_falls_back_to_verl_when_engines_unavailable(monkeypatch):
    """With neither SGLang nor vLLM importable, the dispatcher hands the response token ids
    to verl's tool-parser registry, which needs no inference engine."""
    import uni_agent.gateway.session.codec as codec_mod

    seen = {}

    def missing_engine(*args, **kwargs):
        raise ModuleNotFoundError("tool parser engine")

    async def fake_verl(response_ids, tools, parser_name, tokenizer):
        seen["verl"] = (response_ids, tools, parser_name, tokenizer)
        return "thinking", [SimpleNamespace(name="search", arguments='{"query":"docs"}')]

    monkeypatch.setattr(codec_mod, "_process_tool_calls_sglang", missing_engine, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_vllm", missing_engine, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_verl", fake_verl, raising=False)

    text = 'thinking\n<tool_call>\n{"name": "search", "arguments": {"query": "docs"}}\n</tool_call>'
    tokenizer = FakeTokenizer()
    content, calls = await codec_mod._extract_tool_calls(_ids(text), TOOLS, "hermes", tokenizer)

    assert content == "thinking"
    assert calls[0].name == "search"
    assert seen["verl"] == (_ids(text), TOOLS, "hermes", tokenizer)


@pytest.mark.asyncio
async def test_tool_call_dispatch_returns_text_when_verl_does_not_register_the_parser(monkeypatch):
    """verl's registry raises ValueError for a parser name it does not know; the dispatcher
    then returns the raw text unchanged."""
    import uni_agent.gateway.session.codec as codec_mod

    def missing_engine(*args, **kwargs):
        raise ModuleNotFoundError("tool parser engine")

    async def unknown_parser(*args, **kwargs):
        raise ValueError("Unknown tool parser: qwen3_xml")

    monkeypatch.setattr(codec_mod, "_process_tool_calls_sglang", missing_engine, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_vllm", missing_engine, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_verl", unknown_parser, raising=False)

    text = '<tool_call>\n{"name": "search", "arguments": {"query": "docs"}}\n</tool_call>'
    content, calls = await codec_mod._extract_tool_calls(_ids(text), TOOLS, "qwen3_xml", FakeTokenizer())

    assert content == text
    assert calls == []


@pytest.mark.asyncio
async def test_tool_call_dispatch_returns_text_when_verl_parsing_fails(monkeypatch):
    import uni_agent.gateway.session.codec as codec_mod

    def missing_engine(*args, **kwargs):
        raise ModuleNotFoundError("tool parser engine")

    async def broken_verl(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(codec_mod, "_process_tool_calls_sglang", missing_engine, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_vllm", missing_engine, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_verl", broken_verl, raising=False)

    content, calls = await codec_mod._extract_tool_calls(_ids("plain text"), TOOLS, "hermes", FakeTokenizer())

    assert content == "plain text"
    assert calls == []


@pytest.mark.asyncio
async def test_tool_call_dispatch_prefers_sglang_empty_result_over_fallback(monkeypatch):
    """An installed engine that reports no tool call is authoritative; the fallbacks exist for
    hosts where no engine can run at all, not to second-guess one that did."""
    import uni_agent.gateway.session.codec as codec_mod

    monkeypatch.setattr(
        codec_mod,
        "_process_tool_calls_sglang",
        lambda text, tools, parser_name: (text, []),
        raising=False,
    )

    async def fail_verl(*args, **kwargs):
        raise AssertionError("verl should not run when an engine already answered")

    monkeypatch.setattr(codec_mod, "_process_tool_calls_verl", fail_verl, raising=False)

    text = '<tool_call>\n{"name": "search", "arguments": {"query": "docs"}}\n</tool_call>'
    content, calls = await codec_mod._extract_tool_calls(_ids(text), TOOLS, "hermes", FakeTokenizer())

    assert content == text
    assert calls == []


@pytest.mark.asyncio
async def test_verl_fallback_parses_hermes_envelope():
    """The engine-less path uses verl's own registry, so a Hermes envelope is parsed even on
    a host with neither SGLang nor vLLM installed."""
    import uni_agent.gateway.session.codec as codec_mod

    text = 'thinking\n<tool_call>\n{"name": "search", "arguments": {"query": "docs", "limit": 2}}\n</tool_call>'
    content, calls = await codec_mod._process_tool_calls_verl(_ids(text), TOOLS, "hermes", FakeTokenizer())

    assert content == "thinking\n"
    assert calls[0].name == "search"
    assert json.loads(calls[0].arguments) == {"query": "docs", "limit": 2}


@pytest.mark.asyncio
async def test_decode_response_uses_gateway_dispatcher_for_tool_calls(monkeypatch):
    import uni_agent.gateway.session.codec as codec_mod
    from uni_agent.gateway.session.codec import MessageCodec

    seen = {}

    async def fake_dispatch(response_ids, tools, parser_name, tokenizer):
        seen["dispatch"] = (response_ids, tools, parser_name, tokenizer)
        return "", [SimpleNamespace(name="search", arguments='{"query":"weather"}')]

    monkeypatch.setattr(codec_mod, "_extract_tool_calls", fake_dispatch, raising=False)

    tokenizer = FakeTokenizer()
    codec = MessageCodec(tokenizer, tool_parser_name="qwen3_xml")
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
