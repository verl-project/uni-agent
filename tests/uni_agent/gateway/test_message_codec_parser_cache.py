import sys
import types
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


def _equivalent_tools():
    return [
        {
            "function": {
                "parameters": {
                    "properties": {"query": {"type": "string"}},
                    "type": "object",
                },
                "description": "search docs",
                "name": "search",
            },
            "type": "function",
        }
    ]


def _install_fake_sglang(monkeypatch, constructions):
    protocol = types.ModuleType("sglang.srt.entrypoints.openai.protocol")
    function_call_parser = types.ModuleType("sglang.srt.function_call.function_call_parser")

    class Function:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Tool:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FunctionCallParser:
        def __init__(self, tools, parser_name):
            constructions.append((tools, parser_name))

        def has_tool_call(self, text):
            return False

    protocol.Function = Function
    protocol.Tool = Tool
    function_call_parser.FunctionCallParser = FunctionCallParser
    for name, module in {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.entrypoints": types.ModuleType("sglang.srt.entrypoints"),
        "sglang.srt.entrypoints.openai": types.ModuleType("sglang.srt.entrypoints.openai"),
        "sglang.srt.entrypoints.openai.protocol": protocol,
        "sglang.srt.function_call": types.ModuleType("sglang.srt.function_call"),
        "sglang.srt.function_call.function_call_parser": function_call_parser,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _install_fake_vllm(monkeypatch, constructions, lookups=None):
    protocol = types.ModuleType("vllm.entrypoints.openai.chat_completion.protocol")
    tool_parsers = types.ModuleType("vllm.tool_parsers")

    class ChatCompletionToolsParam:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Parser:
        def __init__(self, tokenizer, *, tools):
            del tokenizer
            constructions.append(tools)

        def extract_tool_calls(self, text, request):
            del request
            return SimpleNamespace(tools_called=False, content=text, tool_calls=[])

    class ToolParserManager:
        @classmethod
        def get_tool_parser(cls, name):
            del cls
            if lookups is not None:
                lookups.append(name)
            return Parser

    protocol.ChatCompletionToolsParam = ChatCompletionToolsParam
    tool_parsers.ToolParserManager = ToolParserManager
    for name, module in {
        "vllm": types.ModuleType("vllm"),
        "vllm.entrypoints": types.ModuleType("vllm.entrypoints"),
        "vllm.entrypoints.openai": types.ModuleType("vllm.entrypoints.openai"),
        "vllm.entrypoints.openai.chat_completion": types.ModuleType("vllm.entrypoints.openai.chat_completion"),
        "vllm.entrypoints.openai.chat_completion.protocol": protocol,
        "vllm.tool_parsers": tool_parsers,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _install_fake_verl(monkeypatch, lookups):
    tool_parser_module = types.ModuleType("verl.experimental.agent_loop.tool_parser")
    schemas_module = types.ModuleType("verl.tools.schemas")

    class Parser:
        async def extract_tool_calls(self, response_ids, tool_schemas):
            del response_ids, tool_schemas
            return "plain", []

    class ToolParser:
        @classmethod
        def get_tool_parser(cls, name, tokenizer):
            del cls, tokenizer
            lookups.append(name)
            return Parser()

    class OpenAIFunctionToolSchema:
        @classmethod
        def model_validate(cls, tool):
            return tool

    tool_parser_module.ToolParser = ToolParser
    schemas_module.OpenAIFunctionToolSchema = OpenAIFunctionToolSchema
    monkeypatch.setitem(sys.modules, "verl.experimental.agent_loop.tool_parser", tool_parser_module)
    monkeypatch.setitem(sys.modules, "verl.tools.schemas", schemas_module)


@pytest.mark.cpu
@pytest.mark.level0
def test_sglang_parser_cache_reuses_equivalent_tool_schemas(monkeypatch):
    from uni_agent.gateway.session.codec import MessageCodec

    constructions = []
    _install_fake_sglang(monkeypatch, constructions)
    codec = MessageCodec(FakeTokenizer())

    assert codec._process_tool_calls_sglang("plain", TOOLS, "qwen3_coder") == (
        "plain",
        [],
    )
    assert codec._process_tool_calls_sglang("plain", _equivalent_tools(), "qwen3_coder") == (
        "plain",
        [],
    )

    assert len(constructions) == 1


@pytest.mark.cpu
@pytest.mark.level0
def test_vllm_parser_cache_separates_tool_schemas(monkeypatch):
    from uni_agent.gateway.session.codec import MessageCodec

    constructions = []
    _install_fake_vllm(monkeypatch, constructions)
    codec = MessageCodec(FakeTokenizer())
    codec._process_tool_calls_vllm("plain", TOOLS, "qwen3_coder")
    codec._process_tool_calls_vllm("plain", _equivalent_tools(), "qwen3_coder")
    different_tools = [{**TOOLS[0], "function": {**TOOLS[0]["function"], "name": "lookup"}}]
    codec._process_tool_calls_vllm("plain", different_tools, "qwen3_coder")

    assert len(constructions) == 2


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enable_tool_parser_cache", "expected_constructions"),
    [(True, 1), (False, 2)],
)
async def test_message_codec_applies_parser_cache_policy_across_decode_calls(
    monkeypatch,
    enable_tool_parser_cache,
    expected_constructions,
):
    from uni_agent.gateway.session.codec import MessageCodec

    constructions = []
    _install_fake_vllm(monkeypatch, constructions)
    codec = MessageCodec(
        FakeTokenizer(),
        tool_parser_name="qwen3_coder",
        rollout_backend="vllm",
        enable_tool_parser_cache=enable_tool_parser_cache,
    )

    await codec.decode_response([ord("x")], tools=TOOLS)
    await codec.decode_response([ord("x")], tools=_equivalent_tools())

    assert len(constructions) == expected_constructions


@pytest.mark.cpu
@pytest.mark.level0
def test_parser_cache_is_scoped_to_message_codec(monkeypatch):
    from uni_agent.gateway.session.codec import MessageCodec

    constructions = []
    _install_fake_vllm(monkeypatch, constructions)

    MessageCodec(FakeTokenizer())._process_tool_calls_vllm("plain", TOOLS, "qwen3_coder")
    MessageCodec(FakeTokenizer())._process_tool_calls_vllm("plain", TOOLS, "qwen3_coder")

    assert len(constructions) == 2


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_verl_parser_cache_ignores_tool_schema(monkeypatch):
    from uni_agent.gateway.session.codec import MessageCodec

    lookups = []
    _install_fake_verl(monkeypatch, lookups)
    codec = MessageCodec(FakeTokenizer())

    await codec._process_tool_calls_verl([ord("x")], TOOLS, "hermes")
    await codec._process_tool_calls_verl([ord("y")], _equivalent_tools(), "hermes")

    assert lookups == ["hermes"]
