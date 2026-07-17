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
                "properties": {"query": {"type": "string"}},
            },
        },
    }
]


def test_tool_call_dispatch_prefers_sglang(monkeypatch):
    import uni_agent.gateway.session.codec as codec_mod

    seen = {}

    def fake_sglang(text, tools, parser_name):
        seen["sglang"] = (text, tools, parser_name)
        return "visible", [SimpleNamespace(name="search", arguments='{"query":"x"}')]

    def fail_vllm(*args, **kwargs):
        raise AssertionError("vLLM should not run when SGLang succeeds")

    monkeypatch.setattr(codec_mod, "_process_tool_calls_sglang", fake_sglang, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_vllm", fail_vllm, raising=False)

    content, calls = codec_mod._extract_tool_calls_with_sglang_or_vllm("raw", TOOLS, "hermes", FakeTokenizer())

    assert content == "visible"
    assert calls[0].name == "search"
    assert seen["sglang"] == ("raw", TOOLS, "hermes")


def test_tool_call_dispatch_falls_back_to_vllm_with_name_mapping(monkeypatch):
    import uni_agent.gateway.session.codec as codec_mod

    seen = {}

    def missing_sglang(*args, **kwargs):
        raise ModuleNotFoundError("sglang")

    def fake_vllm(text, tools, parser_name, tokenizer):
        seen["vllm"] = (text, tools, parser_name, tokenizer)
        return "", [SimpleNamespace(name="search", arguments='{"query":"x"}')]

    monkeypatch.setattr(codec_mod, "_process_tool_calls_sglang", missing_sglang, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_vllm", fake_vllm, raising=False)

    tokenizer = FakeTokenizer()
    content, calls = codec_mod._extract_tool_calls_with_sglang_or_vllm("raw", TOOLS, "qwen25", tokenizer)

    assert content == ""
    assert calls[0].arguments == '{"query":"x"}'
    assert seen["vllm"] == ("raw", TOOLS, "qwen3_xml", tokenizer)


def test_tool_call_dispatch_returns_text_when_backends_unavailable(monkeypatch):
    import uni_agent.gateway.session.codec as codec_mod

    def missing_backend(*args, **kwargs):
        raise ModuleNotFoundError("tool parser backend")

    monkeypatch.setattr(codec_mod, "_process_tool_calls_sglang", missing_backend, raising=False)
    monkeypatch.setattr(codec_mod, "_process_tool_calls_vllm", missing_backend, raising=False)

    content, calls = codec_mod._extract_tool_calls_with_sglang_or_vllm("plain text", TOOLS, "hermes", FakeTokenizer())

    assert content == "plain text"
    assert calls == []


@pytest.mark.asyncio
async def test_decode_response_uses_gateway_dispatcher_for_tool_calls(monkeypatch):
    import uni_agent.gateway.session.codec as codec_mod
    from uni_agent.gateway.session.codec import MessageCodec

    seen = {}

    def fake_dispatch(text, tools, parser_name, tokenizer):
        seen["dispatch"] = (text, tools, parser_name, tokenizer)
        return "", [SimpleNamespace(name="search", arguments='{"query":"weather"}')]

    monkeypatch.setattr(codec_mod, "_extract_tool_calls_with_sglang_or_vllm", fake_dispatch, raising=False)

    tokenizer = FakeTokenizer()
    codec = MessageCodec(tokenizer, tool_parser_name="qwen3_xml")
    message, finish_reason = await codec.decode_response(
        [ord(char) for char in "<tool_call>ignored</tool_call>"],
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
    assert seen["dispatch"][2] == "qwen3_xml"
