"""GUI tools against a fake sandbox: argv construction, screenshots, failure paths."""

from __future__ import annotations

import base64

import pytest

from uni_agent.sandbox.base import ExecResult
from uni_agent.tools import Toolbox

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes"


class _FakeSandbox:
    def __init__(self, *, exit_code: int = 0, stderr: str = "", file_bytes: bytes = PNG, geometry: str = "1024 768"):
        self.geometry = geometry
        self.exit_code = exit_code
        self.stderr = stderr
        self.file_bytes = file_bytes
        self.execs: list[tuple[list[str], dict | None]] = []

    async def exec(self, argv, *, timeout=None, workdir=None, env=None):
        self.execs.append((list(argv), env))
        stdout = self.geometry if argv[:2] == ["xdotool", "getdisplaygeometry"] else ""
        return ExecResult(exit_code=self.exit_code, stdout=stdout, stderr=self.stderr)

    async def read_file(self, path):
        return self.file_bytes


def _toolbox(sandbox: _FakeSandbox, **kwargs) -> Toolbox:
    names = ["gui_screenshot", "gui_click", "gui_type", "gui_key", "gui_scroll"]
    return Toolbox.from_specs([{"name": n, "settle_seconds": 0, **kwargs} for n in names], sandbox=sandbox)


@pytest.mark.cpu
@pytest.mark.level0
def test_gui_tools_are_registered_under_model_facing_names():
    assert _toolbox(_FakeSandbox()).names() == ["screenshot", "click", "type_text", "press_key", "scroll"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_screenshot_returns_png_data_url_and_uses_configured_display():
    sandbox = _FakeSandbox()

    result = await _toolbox(sandbox, display=":7").call("screenshot", {})

    assert result.status == "ok"
    assert result.images == ("data:image/png;base64," + base64.b64encode(PNG).decode(),)
    argv, env = sandbox.execs[0]
    assert argv[0] == "import" and argv[1:3] == ["-window", "root"]
    assert env == {"DISPLAY": ":7"}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "args", "expected_argv"),
    [
        (
            "click",
            {"x": 10, "y": 20},
            ["xdotool", "mousemove", "10", "20", "click", "--repeat", "1", "--delay", "100", "1"],
        ),
        (
            "click",
            {"x": 1, "y": 2, "button": "right", "double": True},
            ["xdotool", "mousemove", "1", "2", "click", "--repeat", "2", "--delay", "100", "3"],
        ),
        ("type_text", {"text": "-rf hello"}, ["xdotool", "type", "--delay", "12", "--", "-rf hello"]),
        ("press_key", {"keys": "ctrl+a Return"}, ["xdotool", "key", "--", "ctrl+a", "Return"]),
        (
            "scroll",
            {"x": 5, "y": 6, "direction": "down", "amount": 4},
            ["xdotool", "mousemove", "5", "6", "click", "--repeat", "4", "5"],
        ),
    ],
)
async def test_actions_build_expected_argv_and_attach_a_screenshot(tool, args, expected_argv):
    sandbox = _FakeSandbox()

    result = await _toolbox(sandbox).call(tool, args)

    assert result.status == "ok"
    assert sandbox.execs[0][0] == expected_argv
    assert sandbox.execs[1][0][0] == "import"  # the follow-up screenshot
    assert len(result.images) == 1


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_screenshot_after_action_can_be_disabled():
    sandbox = _FakeSandbox()

    result = await _toolbox(sandbox, screenshot_after_action=False).call("click", {"x": 1, "y": 1})

    assert result.images == ()
    assert len(sandbox.execs) == 1


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("click", {"x": -1, "y": 0}),
        ("click", {"x": 1}),
        ("click", {"x": 1, "y": 1, "button": "back"}),
        ("type_text", {"text": ""}),
        ("press_key", {"keys": ""}),
        ("scroll", {"x": 1, "y": 1, "direction": "left"}),
        ("scroll", {"x": 1, "y": 1, "direction": "up", "amount": 0}),
    ],
)
async def test_invalid_arguments_are_format_errors_and_run_nothing(tool, args):
    sandbox = _FakeSandbox()

    result = await _toolbox(sandbox).call(tool, args)

    assert result.status == "format_error"
    assert sandbox.execs == []


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_failed_command_is_a_tool_error_with_stderr():
    sandbox = _FakeSandbox(exit_code=1, stderr="Can't open display :0")

    result = await _toolbox(sandbox).call("click", {"x": 1, "y": 1})

    assert result.status == "error"
    assert "Can't open display" in result.text
    assert result.images == ()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_non_png_screenshot_is_a_tool_error():
    result = await _toolbox(_FakeSandbox(file_bytes=b"not a png")).call("screenshot", {})

    assert result.status == "error"
    assert "not a PNG" in result.text


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_relative_1000_coordinates_are_scaled_to_the_screen():
    sandbox = _FakeSandbox(geometry="1024 768")

    result = await _toolbox(sandbox, coordinate_space="relative_1000").call("click", {"x": 292, "y": 328})

    assert result.status == "ok"
    clicks = [argv for argv, _ in sandbox.execs if argv[:2] == ["xdotool", "mousemove"]]
    assert clicks == [["xdotool", "mousemove", "299", "252", "click", "--repeat", "1", "--delay", "100", "1"]]
    assert "Clicked left at (299, 252)" in result.text


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_relative_1000_clamps_the_far_edge_inside_the_screen_and_reads_size_once_per_tool():
    sandbox = _FakeSandbox(geometry="1024 768")
    toolbox = _toolbox(sandbox, coordinate_space="relative_1000", screenshot_after_action=False)

    await toolbox.call("click", {"x": 1000, "y": 1000})
    await toolbox.call("click", {"x": 0, "y": 0})

    moves = [argv[2:4] for argv, _ in sandbox.execs if argv[:2] == ["xdotool", "mousemove"]]
    assert moves == [["1023", "767"], ["0", "0"]]
    assert sum(argv[:2] == ["xdotool", "getdisplaygeometry"] for argv, _ in sandbox.execs) == 1


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_relative_1000_rejects_out_of_range_coordinates():
    sandbox = _FakeSandbox()

    result = await _toolbox(sandbox, coordinate_space="relative_1000").call("click", {"x": 1001, "y": 5})

    assert result.status == "error"
    assert all(argv[0] != "xdotool" or argv[1] != "mousemove" for argv, _ in sandbox.execs)


@pytest.mark.cpu
@pytest.mark.level0
def test_relative_mode_is_announced_in_the_tool_description():
    description = _toolbox(_FakeSandbox(), coordinate_space="relative_1000").schemas()[1]["function"]["description"]

    assert "0-1000" in description
    assert "0-1000" not in _toolbox(_FakeSandbox()).schemas()[1]["function"]["description"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["screenshot", "click"])
async def test_command_timeout_is_a_timeout_result_not_an_error(tool):
    sandbox = _FakeSandbox(exit_code=-1, stderr="exec timed out after 5s")

    result = await _toolbox(sandbox).call(tool, {"x": 1, "y": 1} if tool == "click" else {}, timeout=5)

    assert result.status == "timeout"
    assert "did not finish within 5" in result.text
    assert result.images == ()
