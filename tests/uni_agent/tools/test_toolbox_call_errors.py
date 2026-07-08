"""Error-path tests for `Toolbox.call`, the tools layer's single dispatch point.

It parses every tool call and turns malformed calls / tool failures into a result
the policy sees (skipped, never raised), classified via :attr:`ToolResult.status`.
These pin down those returns plus :meth:`ToolResult.to_observation` rendering. No
sandbox / network -- the tools are tiny fakes, so it runs fast under ``pytest`` (or
``python`` on this file).
"""

from __future__ import annotations

import asyncio

import pytest

from uni_agent.tools import Tool, Toolbox, ToolError, ToolResult


class _Echo(Tool):
    """Succeeds, echoing the parsed args + forwarded timeout back as text."""

    name = "echo"

    async def run(self, args, *, timeout=None):
        return ToolResult(text=f"args={args} timeout={timeout}")


class _Boom(Tool):
    """Raises a :class:`ToolError` -- an expected runtime failure."""

    name = "boom"

    async def run(self, args, *, timeout=None):
        raise ToolError("kaboom")


class _Slow(Tool):
    """Reports a self-timeout the way the shell does (``status="timeout"``)."""

    name = "slow"

    async def run(self, args, *, timeout=None):
        return ToolResult(text="partial output", status="timeout")


class _Kaboom(Tool):
    """Raises a non-:class:`ToolError` -- a tool bug or an infra fault (dead sandbox).

    Unlike :class:`_Boom` (a *ToolError*, caught and returned), this is meant to
    propagate out of :meth:`Toolbox.call` so the caller can end/bucket the episode.
    """

    name = "kaboom"

    async def run(self, args, *, timeout=None):
        raise RuntimeError("modal sandbox is not alive")


def _toolbox() -> Toolbox:
    sandbox = object()  # the fakes above never touch the sandbox
    return Toolbox([_Echo(sandbox), _Boom(sandbox), _Slow(sandbox), _Kaboom(sandbox)])


# --------------------------- Toolbox.call: error returns ---------------------------


def test_unknown_tool_returns_format_hint():
    result = asyncio.run(_toolbox().call("does_not_exist", "{}"))
    assert result.status == "format_error"
    assert "Invalid action: function 'does_not_exist' is not defined" in result.text
    assert "Allowed functions should be one of:" in result.text


@pytest.mark.parametrize(
    "raw, needle",
    [
        ("{not valid json", "could not parse arguments"),  # malformed JSON
        ("[1, 2, 3]", "must be a JSON object"),  # valid JSON, but a list
        ("42", "must be a JSON object"),  # valid JSON, but a scalar
        ('"just a string"', "must be a JSON object"),  # valid JSON, but a string
    ],
)
def test_bad_arguments_return_format_hint(raw, needle):
    result = asyncio.run(_toolbox().call("echo", raw))
    assert result.status == "format_error"
    assert needle in result.text


def test_tool_error_is_caught_and_tagged():
    result = asyncio.run(_toolbox().call("boom", "{}"))
    assert result.status == "error"
    assert result.text == "Error: kaboom"


def test_unexpected_exception_propagates():
    # A non-ToolError (tool bug / infra fault) is NOT swallowed into an observation;
    # it propagates so the caller ends/buckets the episode instead of feeding it back.
    with pytest.raises(RuntimeError, match="modal sandbox is not alive"):
        asyncio.run(_toolbox().call("kaboom", "{}"))


@pytest.mark.parametrize("raw", ["{}", "", None, '{"a": 1}', {"a": 1}])
def test_valid_arguments_run_the_tool(raw):
    result = asyncio.run(_toolbox().call("echo", raw))
    assert result.status == "ok"
    assert result.text.startswith("args=")


def test_timeout_is_forwarded_to_the_tool():
    result = asyncio.run(_toolbox().call("echo", "{}", timeout=12.5))
    assert "timeout=12.5" in result.text


# --------------------------- status classification ---------------------------


@pytest.mark.parametrize(
    "name, raw, expected_status",
    [
        ("echo", '{"a": 1}', "ok"),
        ("does_not_exist", "{}", "format_error"),
        ("echo", "{not valid json", "format_error"),
        ("echo", "[1]", "format_error"),
        ("boom", "{}", "error"),
        ("slow", "{}", "timeout"),
    ],
)
def test_call_status(name, raw, expected_status):
    result = asyncio.run(_toolbox().call(name, raw, timeout=30.0))
    assert result.status == expected_status


# --------------------------- to_observation() ---------------------------


def test_to_observation_prepends_the_label():
    assert ToolResult(text="hello").to_observation() == "Observation:\nhello"
    assert ToolResult().to_observation() == "Observation:\n"  # no text channel -> just the label


def test_to_observation_clips_over_max_length():
    text = "x" * 50
    clipped = ToolResult(text=text).to_observation(max_length=10)
    assert clipped.startswith("Observation:\n" + "x" * 10)
    assert "<response clipped>" in clipped
    assert "40" in clipped  # the elided-character count is reported
    assert (
        ToolResult(text=text).to_observation(max_length=1000) == f"Observation:\n{text}"
    )  # under cap -> verbatim body


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
