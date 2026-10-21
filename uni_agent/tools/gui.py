"""``gui`` toolset: drive an X11 desktop inside the sandbox from the host.

Each tool is one model-visible action (screenshot, click, type, key, scroll) run
through ``sandbox.exec`` with ``DISPLAY`` set: ``xdotool`` for input and ImageMagick
``import`` for the screenshot. The sandbox image needs a running X server plus those
two binaries (see ``examples/gui/Dockerfile``). Screenshots come back as
``ToolResult.images`` so the agent loop can show them to a vision model.

The screenshot is never rescaled. Coordinates are screen pixels by default; models that
ground on a normalized grid (Qwen-VL family: 0-1000 over the image) use
``coordinate_space="relative_1000"``, and the tools convert to pixels.
"""

from __future__ import annotations

import abc
import asyncio
import base64
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from .base import Tool, ToolError, ToolResult, register_tool

_SCREENSHOT_PATH = "/tmp/uni_agent_gui_screenshot.png"
#: ``Sandbox.exec`` reports a command that hit its timeout with this exit code.
_TIMEOUT_EXIT_CODE = -1
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _CommandTimeout(Exception):
    """A sandbox command exceeded its timeout; :meth:`_GuiTool.run` turns it into a ``timeout`` result."""


class GuiConfig(BaseModel):
    display: str = Field(default=":0", description="X display the desktop runs on.")
    coordinate_space: Literal["pixel", "relative_1000"] = Field(
        default="pixel",
        description="Space of x / y the model emits: screen pixels, or a 0-1000 grid over the screen "
        "(the grounding convention of Qwen-VL models).",
    )
    screenshot_after_action: bool = Field(
        default=True, description="Attach a fresh screenshot to every click / type / key / scroll result."
    )
    settle_seconds: float = Field(
        default=0.5, ge=0, description="Pause after an action so the UI can redraw before the screenshot."
    )


class _GuiTool(Tool):
    """Shared plumbing: run an X command, grab a screenshot, wrap the result."""

    config_model: ClassVar[type[BaseModel] | None] = GuiConfig
    _screen_size: tuple[int, int] | None = None

    @property
    def _cfg(self) -> GuiConfig:
        assert isinstance(self.config, GuiConfig)
        return self.config

    def schema(self) -> dict:
        schema = super().schema()
        if self._cfg.coordinate_space == "relative_1000":
            schema["function"]["description"] += (
                " Coordinates are on a 0-1000 grid: (0, 0) is the top-left of the screenshot,"
                " (1000, 1000) the bottom-right."
            )
        return schema

    async def _pixels(self, x: int, y: int) -> tuple[int, int]:
        """Convert a model coordinate to screen pixels per ``coordinate_space``."""
        if self._cfg.coordinate_space == "pixel":
            return x, y
        if x > 1000 or y > 1000:
            raise ToolError(f"coordinates must be within 0-1000, got ({x}, {y})")
        if self._screen_size is None:
            result = await self.sandbox.exec(["xdotool", "getdisplaygeometry"], env={"DISPLAY": self._cfg.display})
            try:
                width, height = (int(v) for v in result.stdout.split())
            except ValueError:
                detail = (result.stderr or result.stdout).strip()[:200]
                raise ToolError(f"could not read the screen size: {detail}") from None
            self._screen_size = (width, height)
        width, height = self._screen_size
        return min(round(x / 1000 * width), width - 1), min(round(y / 1000 * height), height - 1)

    async def run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        try:
            return await self._run(args, timeout=timeout)
        except _CommandTimeout as exc:
            return ToolResult(text=f"Timed out: {exc}", status="timeout")

    @abc.abstractmethod
    async def _run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult: ...

    async def _x(self, argv: list[str], *, timeout: float | None = None) -> None:
        result = await self.sandbox.exec(argv, timeout=timeout, env={"DISPLAY": self._cfg.display})
        if result.exit_code == _TIMEOUT_EXIT_CODE:
            raise _CommandTimeout(f"`{argv[0]}` did not finish within {timeout}s")
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout).strip()[:500]
            raise ToolError(f"`{argv[0]}` failed (exit {result.exit_code}): {detail}")

    async def _screenshot(self, *, timeout: float | None = None) -> str:
        await self._x(["import", "-window", "root", _SCREENSHOT_PATH], timeout=timeout)
        data = await self.sandbox.read_file(_SCREENSHOT_PATH)
        if not data.startswith(_PNG_MAGIC):
            raise ToolError("screenshot is not a PNG (is the display up?)")
        return "data:image/png;base64," + base64.b64encode(data).decode()

    async def _done(self, text: str, *, timeout: float | None = None) -> ToolResult:
        if not self._cfg.screenshot_after_action:
            return ToolResult(text=text)
        await asyncio.sleep(self._cfg.settle_seconds)
        return ToolResult(text=text, images=(await self._screenshot(timeout=timeout),))


class _NoArguments(BaseModel):
    pass


@register_tool("gui_screenshot")
class ScreenshotTool(_GuiTool):
    name = "screenshot"
    description = "Capture the current screen. Returns the image."
    args_model = _NoArguments

    async def _run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        return ToolResult(text="Screenshot of the current screen.", images=(await self._screenshot(timeout=timeout),))


class ClickArguments(BaseModel):
    x: int = Field(ge=0, description="Horizontal position of the target (0 is the left edge).")
    y: int = Field(ge=0, description="Vertical position of the target (0 is the top edge).")
    button: Literal["left", "middle", "right"] = Field(default="left", description="Mouse button.")
    double: bool = Field(default=False, description="Double-click instead of a single click.")


_BUTTONS = {"left": "1", "middle": "2", "right": "3"}


@register_tool("gui_click")
class ClickTool(_GuiTool):
    name = "click"
    description = "Move the mouse to (x, y) in the screenshot and click."
    args_model = ClickArguments

    async def _run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        x, y = await self._pixels(int(args["x"]), int(args["y"]))
        button = args.get("button", "left")
        repeat = "2" if args.get("double") else "1"
        await self._x(
            ["xdotool", "mousemove", str(x), str(y), "click", "--repeat", repeat, "--delay", "100", _BUTTONS[button]],
            timeout=timeout,
        )
        return await self._done(f"Clicked {button} at ({x}, {y}).", timeout=timeout)


class TypeArguments(BaseModel):
    text: str = Field(min_length=1, description="Text to type into the focused window.")


@register_tool("gui_type")
class TypeTool(_GuiTool):
    name = "type_text"
    description = "Type text into the focused window. Use press_key for Enter, Tab and shortcuts."
    args_model = TypeArguments

    async def _run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        text = str(args["text"])
        # `--` keeps text that starts with `-` from being read as an xdotool option.
        await self._x(["xdotool", "type", "--delay", "12", "--", text], timeout=timeout)
        return await self._done(f"Typed {len(text)} characters.", timeout=timeout)


class KeyArguments(BaseModel):
    keys: str = Field(
        min_length=1,
        description="One key or chord in xdotool syntax, e.g. 'Return', 'ctrl+c', 'alt+Tab'. "
        "Separate several presses with spaces.",
    )


@register_tool("gui_key")
class KeyTool(_GuiTool):
    name = "press_key"
    description = "Press a key or key chord such as Return, Tab, Escape, ctrl+a or alt+F4."
    args_model = KeyArguments

    async def _run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        keys = str(args["keys"]).split()
        await self._x(["xdotool", "key", "--", *keys], timeout=timeout)
        return await self._done(f"Pressed {' '.join(keys)}.", timeout=timeout)


class ScrollArguments(BaseModel):
    x: int = Field(ge=0, description="Horizontal position to scroll at.")
    y: int = Field(ge=0, description="Vertical position to scroll at.")
    direction: Literal["up", "down"] = Field(description="Scroll direction.")
    amount: int = Field(default=3, ge=1, le=50, description="Number of wheel clicks.")


_WHEEL = {"up": "4", "down": "5"}


@register_tool("gui_scroll")
class ScrollTool(_GuiTool):
    name = "scroll"
    description = "Scroll the mouse wheel at (x, y)."
    args_model = ScrollArguments

    async def _run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        x, y = await self._pixels(int(args["x"]), int(args["y"]))
        direction, amount = args["direction"], int(args.get("amount", 3))
        await self._x(
            ["xdotool", "mousemove", str(x), str(y), "click", "--repeat", str(amount), _WHEEL[direction]],
            timeout=timeout,
        )
        return await self._done(f"Scrolled {direction} {amount} at ({x}, {y}).", timeout=timeout)
