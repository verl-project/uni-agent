"""Verify complete-response delta optimization applies to all ParserEngine-based parsers."""

from types import SimpleNamespace

import pytest

from tests.uni_agent.gateway.test_message_codec_tool_dispatch import TOOLS
from tests.uni_agent.support import FakeTokenizer
from uni_agent.gateway.session.codec import MessageCodec


class MinimalQwenTokenizer(FakeTokenizer):
    """Tokenizer surface required by vLLM's Qwen3 parser."""

    def get_vocab(self):
        return {
            "<tool_call>": 1,
            "</tool_call>": 2,
            "<think>": 3,
            "</think>": 4,
        }

    def get_added_vocab(self):
        return {}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("parser_name", ["qwen3_coder", "qwen3_xml", "qwen3"])
def test_delta_optimization_applies_to_all_qwen3_aliases(parser_name, monkeypatch):
    """All qwen3 variants (qwen3_coder, qwen3_xml, qwen3) should skip delta computation."""
    pytest.importorskip("vllm.parser.qwen3")
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
    from vllm.tool_parsers import ToolParserManager

    tokenizer = MinimalQwenTokenizer()
    tools = [ChatCompletionToolsParam(**tool) for tool in TOOLS]

    # Create a real parser to verify it has the expected structure
    parser = ToolParserManager.get_tool_parser(parser_name)(tokenizer, tools=tools)
    assert hasattr(parser, "_parser_engine")
    assert hasattr(parser._parser_engine, "_arg_converter")
    assert parser._parser_engine._arg_converter is not None
    assert hasattr(parser._parser_engine, "_stream_arg_deltas")

    codec = MessageCodec(tokenizer)
    engine_cls = type(parser._parser_engine)
    original_delta = engine_cls._compute_arg_delta
    delta_calls = []

    def tracked_delta(engine, idx, raw_delta):
        delta_calls.append(engine._stream_arg_deltas)
        return original_delta(engine, idx, raw_delta)

    monkeypatch.setattr(engine_cls, "_compute_arg_delta", tracked_delta)

    # A response with a long parameter value that would trigger many delta calls
    text = (
        "<tool_call>\n"
        "<function=search>\n"
        "<parameter=query>\n" + "x<z>" * 200 + "\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    content, calls = codec._process_tool_calls_vllm(text, TOOLS, parser_name)

    # Verify the tool call was correctly parsed
    assert len(calls) == 1
    assert calls[0].name == "search"

    # Verify _compute_arg_delta was never called with _stream_arg_deltas=True
    assert not any(delta_calls), (
        f"Parser {parser_name} called _compute_arg_delta with _stream_arg_deltas=True "
        f"during complete-response parsing, indicating delta optimization did not apply"
    )

    # Verify the cached parser's flag was restored
    cached_parser = next(iter(codec._tool_parser_cache.values()))
    assert cached_parser._parser_engine._stream_arg_deltas is True


@pytest.mark.cpu
@pytest.mark.level0
def test_delta_optimization_skips_parsers_without_converter(monkeypatch):
    """Parsers without arg_converter should skip the optimization and use standard path."""
    from vllm.tool_parsers import ToolParserManager

    class ParserWithoutConverter:
        """Mock parser that looks like a ParserEngine but has no arg_converter."""

        def __init__(self, tokenizer, *, tools):
            self._parser_engine = SimpleNamespace(
                _stream_arg_deltas=True,
                _arg_converter=None,  # No converter
            )

        def extract_tool_calls(self, text, request):
            # Standard path should be used
            return SimpleNamespace(
                tools_called=False,
                content=text,
                tool_calls=[],
            )

    monkeypatch.setattr(ToolParserManager, "get_tool_parser", lambda name: ParserWithoutConverter)
    codec = MessageCodec(FakeTokenizer())

    # Should not raise; should use standard path
    content, calls = codec._process_tool_calls_vllm("plain text", TOOLS, "test_parser")
    assert content == "plain text"
    assert calls == []


@pytest.mark.cpu
@pytest.mark.level0
def test_delta_optimization_handles_legacy_parsers(monkeypatch):
    """Legacy parsers without _parser_engine should work via standard path."""
    from vllm.tool_parsers import ToolParserManager

    class LegacyParser:
        """Mock parser without _parser_engine (like old Hermes implementation)."""

        def __init__(self, tokenizer, *, tools):
            pass  # No _parser_engine attribute

        def extract_tool_calls(self, text, request):
            return SimpleNamespace(
                tools_called=False,
                content=text,
                tool_calls=[],
            )

    monkeypatch.setattr(ToolParserManager, "get_tool_parser", lambda name: LegacyParser)
    codec = MessageCodec(FakeTokenizer())

    # Should not raise; should use standard path
    content, calls = codec._process_tool_calls_vllm("plain text", TOOLS, "legacy")
    assert content == "plain text"
    assert calls == []


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("has_converter_attr", [True, False])
@pytest.mark.parametrize("has_deltas_attr", [True, False])
def test_delta_optimization_version_compatibility(has_converter_attr, has_deltas_attr, monkeypatch):
    """The optimization should gracefully degrade across vLLM versions with different APIs."""
    from vllm.tool_parsers import ToolParserManager

    flag_during_parse = []

    class VersionedParser:
        def __init__(self, tokenizer, *, tools):
            engine = SimpleNamespace()
            if has_converter_attr:
                engine._arg_converter = lambda x, y: x
            if has_deltas_attr:
                engine._stream_arg_deltas = True
            self._parser_engine = engine

        def extract_tool_calls(self, text, request):
            # Record the flag state during parse to verify optimization behavior
            flag_during_parse.append(
                getattr(self._parser_engine, "_stream_arg_deltas", None)
            )
            return SimpleNamespace(tools_called=False, content=text, tool_calls=[])

    monkeypatch.setattr(ToolParserManager, "get_tool_parser", lambda name: VersionedParser)
    codec = MessageCodec(FakeTokenizer())

    # All combinations should work without raising
    content, calls = codec._process_tool_calls_vllm("test", TOOLS, "versioned")
    assert content == "test"
    assert calls == []

    # Verify optimization behavior based on attribute presence
    if has_converter_attr and has_deltas_attr:
        # Optimization should apply: flag was False during parse
        assert flag_during_parse == [False], (
            "When both _arg_converter and _stream_arg_deltas exist, "
            "the flag should be False during parse (optimization applied)"
        )
        # Flag should be restored after parse
        cached = next(iter(codec._tool_parser_cache.values()))
        assert cached._parser_engine._stream_arg_deltas is True
    else:
        # Standard path: flag unchanged (True if attr exists, None if not)
        expected = True if has_deltas_attr else None
        assert flag_during_parse == [expected], (
            f"Without both attrs, flag should stay at original value "
            f"(expected {expected}, got {flag_during_parse})"
        )

