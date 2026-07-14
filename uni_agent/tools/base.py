"""Tool layer: host-side tools, each a schema plus a (possibly stateful) ``run``.

The agent runs *outside* the task image. A :class:`Tool` pairs a schema (what the
model sees) with an async :meth:`run` that drives the container through the
:class:`~uni_agent.sandbox.SandboxBackend` data plane. A tool is built with its
sandbox and owns whatever state it needs -- the editor keeps undo history and the
shell holds a live channel opened lazily and closed in :meth:`close`. Every :meth:`run` returns a normalized :class:`ToolResult`
(``text`` + ``status``); :class:`Toolbox` binds a set of tools to one sandbox.
"""

from __future__ import annotations

import abc
import asyncio
import dataclasses
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from uni_agent.sandbox import SandboxBackend

logger = logging.getLogger(__name__)

ToolStatus = Literal["ok", "format_error", "error", "timeout"]


@dataclasses.dataclass
class ToolResult:
    """One tool call's normalized result: the ``text`` the model sees plus a
    ``status`` for the loop's counters. ``status`` is set by whoever knows it --
    :meth:`Toolbox.call` for bad calls / tool errors, the tool itself for a
    ``"timeout"`` -- and defaults to ``"ok"``. :meth:`to_observation` renders the
    next-turn content (``str(result)`` gives just the text).
    """

    text: str | None = None
    status: ToolStatus = "ok"

    def __str__(self) -> str:
        return self.text if self.text is not None else ""

    def to_observation(self, max_length: int = 100_000) -> str:
        text = self.text or ""
        if len(text) > max_length:
            elided = len(text) - max_length
            text = (
                f"{text[:max_length]}\n<response clipped>\n"
                f"<NOTE>The observation exceeded {max_length} characters, so {elided} were elided. "
                "Retry with something that yields less output (e.g. head/tail/grep, or redirect to "
                "a file); do not use interactive pagers.</NOTE>"
            )
        return f"Observation:\n{text}"


class ToolError(Exception):
    """A user-facing *runtime* failure while executing (bad path, refused command).

    :meth:`Toolbox.call` turns it into an ``"Error: ..."`` observation for the model
    instead of crashing the rollout; let genuine bugs propagate. Malformed *calls*
    raise :class:`ToolCallFormatError` instead.
    """


class ToolCallFormatError(Exception):
    """A malformed tool call, caught *before* the tool runs (unknown function, or
    arguments that don't decode to a JSON object). :meth:`Toolbox.call` returns the
    message to the policy as an observation so it can self-correct; the wording
    follows the ``"Invalid action: ..."`` convention the policy is trained on.
    """


def _normalize_json_schema(value: Any) -> Any:
    """Normalize Pydantic JSON Schema into the shape tool runtimes expect.

    Drops ``title``, collapses ``Optional[...]`` ``anyOf`` down to the non-null
    variant, removes ``default: null`` and applies a stable key order, yielding
    the standard OpenAI function-call parameter schema.
    """
    if isinstance(value, list):
        return [_normalize_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if key == "title":
            continue
        normalized_item = _normalize_json_schema(item)
        if key == "default" and normalized_item is None:
            continue
        normalized[key] = normalized_item

    if "anyOf" in normalized:
        non_null_variants = [
            item
            for item in normalized["anyOf"]
            if not (isinstance(item, dict) and item.get("type") == "null" and len(item) == 1)
        ]
        if len(non_null_variants) == 1 and isinstance(non_null_variants[0], dict):
            merged = dict(non_null_variants[0])
            for key, item in normalized.items():
                if key != "anyOf":
                    merged[key] = item
            normalized = merged

    preferred_order = ("type", "description", "enum", "default", "items", "properties", "required")
    ordered: dict[str, Any] = {}
    for key in preferred_order:
        if key in normalized:
            ordered[key] = normalized.pop(key)
    ordered.update(normalized)
    return ordered


def build_function_schema(name: str, description: str, model: type[BaseModel]) -> dict:
    """Build an OpenAI-compatible function schema from a Pydantic args model."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": _normalize_json_schema(model.model_json_schema()),
        },
    }


class Tool(abc.ABC):
    """A host-side tool: a schema plus an async :meth:`run` over the sandbox.

    Subclasses set :attr:`name` / :attr:`description` / :attr:`args_model` (the
    per-call args, used by the default :meth:`schema`) and implement :meth:`run`.
    Construction kwargs are declared via :attr:`config_model` and auto-parsed into
    ``self.config`` (e.g. the shell's ``command_timeout``). Stateful tools open a
    channel lazily and release it in :meth:`close`.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    args_model: ClassVar[type[BaseModel] | None] = None
    #: Pydantic schema for this tool's construction kwargs (a tools entry's kwargs).
    #: ``None`` means the tool takes no kwargs beyond the sandbox.
    config_model: ClassVar[type[BaseModel] | None] = None

    def __init__(self, sandbox: SandboxBackend, **kwargs: Any):
        self.sandbox = sandbox
        if self.config_model is not None:
            # Auto-parse: raw kwargs -> typed, validated config object.
            self.config: BaseModel | None = self.config_model(**kwargs)
        elif kwargs:
            raise TypeError(
                f"{type(self).__name__} takes no tool kwargs, got {sorted(kwargs)}"
            )
        else:
            self.config = None

    def schema(self) -> dict:
        """Return the OpenAI function schema shown to the model."""
        if self.args_model is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set `args_model` or override schema()"
            )
        return build_function_schema(self.name, self.description, self.args_model)

    @classmethod
    def config_schema(cls) -> dict | None:
        """JSON schema for this tool's construction kwargs, or ``None`` if it has none."""
        return cls.config_model.model_json_schema() if cls.config_model is not None else None

    @abc.abstractmethod
    async def run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        """Execute the call and return a :class:`ToolResult`.

        Drive the container through ``self.sandbox``; raise :class:`ToolError` for
        user-facing failures.
        """
        ...

    async def start(self) -> None:
        """Eagerly set up state (open channels); no-op by default.

        Optional counterpart to :meth:`close` -- tools open channels lazily on first
        :meth:`run`, so this just front-loads that cost (e.g. the shell installing
        ``tmux``) before the first turn.
        """
        return None

    async def close(self) -> None:
        """Release any state the tool holds (open channels). No-op by default."""
        return None


TOOL_REGISTRY: dict[str, type[Tool]] = {}


def register_tool(name: str):
    """Class decorator: register ``cls`` under registry key ``name``.

    The registry key (config / :func:`get_tool`) is independent of the model-facing
    :attr:`Tool.name`: a class without its own ``name`` inherits the key, one that
    sets ``name`` keeps it -- e.g. ``stateful_shell`` is seen by the model as ``shell``.
    """

    def decorator(cls: type[Tool]) -> type[Tool]:
        if name in TOOL_REGISTRY and TOOL_REGISTRY[name] is not cls:
            raise ValueError(
                f"Tool {name!r} already registered: {TOOL_REGISTRY[name]!r} vs {cls!r}"
            )
        if not cls.__dict__.get("name"):
            cls.name = name
        TOOL_REGISTRY[name] = cls
        return cls

    return decorator


def get_tool(name: str, sandbox: SandboxBackend, **kwargs: Any) -> Tool:
    """Instantiate a registered tool by name, bound to ``sandbox``.

    Extra ``kwargs`` are auto-parsed into the tool's ``config_model``, e.g.
    ``get_tool("stateful_shell", sb, command_timeout=120)``.
    """
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {name!r}")
    return TOOL_REGISTRY[name](sandbox, **kwargs)


class Toolbox:
    """A set of tool instances bound to one sandbox for a rollout.

    Holds the instances (so stateful tools keep state across calls) and exposes the
    model-facing :meth:`schemas`, the single :meth:`call` dispatch, and :meth:`close`.
    """

    def __init__(self, tools: list[Tool]):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self._tools[tool.name] = tool

    @classmethod
    def from_specs(cls, specs: list[dict[str, Any]], *, sandbox: SandboxBackend) -> Toolbox:
        """Build a toolbox from ``{name, ...kwargs}`` config entries bound to ``sandbox``.

        Each entry has a ``name`` (a TOOL_REGISTRY key) plus that tool's kwargs, e.g.
        ``{"name": "stateful_shell", "command_timeout": 120}``.
        """
        tools: list[Tool] = []
        for entry in specs:
            if not isinstance(entry, dict) or not entry.get("name"):
                raise ValueError(f"each tools entry must be a mapping with a 'name': {entry!r}")
            kwargs = {k: v for k, v in entry.items() if k != "name"}
            tools.append(get_tool(entry["name"], sandbox, **kwargs))
        return cls(tools)

    @classmethod
    def all(cls, *, sandbox: SandboxBackend) -> Toolbox:
        """Build a toolbox from every registered tool, each bound to ``sandbox``."""
        return cls([t(sandbox) for t in TOOL_REGISTRY.values()])

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        """OpenAI function schemas for every tool (pass straight to the model)."""
        return [tool.schema() for tool in self._tools.values()]

    async def start(self) -> None:
        """Eagerly set up every tool once, front-loading first-use cost (no retry).

        Prefer ``async with toolbox`` for a rollout: it starts with retries and closes
        on exit. This bare version is for callers that manage their own lifecycle.
        """
        for tool in self._tools.values():
            await tool.start()

    async def __aenter__(self, retry: int = 3, timeout: float = 60.0) -> Toolbox:
        """Enter a rollout: start every tool (retrying transient failures) and return
        the ready toolbox. If a tool can't be started, tools already started are closed
        before the error propagates; :meth:`close` runs again on normal exit.

        ``async with`` uses the defaults; call ``__aenter__(retry=..., timeout=...)``
        directly to override. ``timeout`` caps each per-tool start attempt; raise it if a
        tool's first-use setup is slow (e.g. the shell installing tmux).
        """
        retry = max(1, retry)
        try:
            for tool in self._tools.values():
                await self._start_tool(tool, retry, timeout)
        except BaseException:
            await self.close()  # roll back partially-started tools
            raise
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        await self.close()
        return False

    @asynccontextmanager
    async def entered(self, **start_kwargs: Any) -> AsyncIterator[Toolbox]:
        """Parametrized ``async with``: same lifecycle as ``async with toolbox``, but
        forwards ``retry`` / ``timeout`` to :meth:`__aenter__` (the bare ``async with``
        can't pass args)::

            async with toolbox.entered(retry=5, timeout=90) as tb:
                ...
        """
        await self.__aenter__(**start_kwargs)
        try:
            yield self
        finally:
            await self.close()

    async def _start_tool(self, tool: Tool, retry: int, timeout: float) -> None:
        """Start one tool, retrying with backoff on timeout or transient failure.

        Each attempt is bounded by ``asyncio.wait_for(timeout)`` so a hung connect can't
        block forever; after ``retry`` attempts the last error propagates. A genuine
        cancellation of our own task is re-raised immediately, never retried.
        """
        last_exc: BaseException | None = None
        for attempt in range(1, retry + 1):
            try:
                await asyncio.wait_for(tool.start(), timeout=timeout)
                return
            except asyncio.CancelledError as exc:
                task = asyncio.current_task()
                if task is not None and task.cancelling() > 0:
                    raise  # our caller asked to cancel -- honor it
                last_exc = exc  # a dependency (e.g. Modal connect) leaked a cancel
            except Exception as exc:  # incl. asyncio.TimeoutError from wait_for
                last_exc = exc
            logger.warning(
                "tool %r failed to start (attempt %d/%d): %r",
                tool.name, attempt, retry, last_exc,
            )
            if attempt < retry:
                await asyncio.sleep(2 * attempt)
                timeout = timeout * 2
        assert last_exc is not None
        logger.error("tool %r failed to start after %d attempts: %r", tool.name, retry, last_exc)
        raise last_exc

    async def call(
        self, name: str, args: dict[str, Any] | str | None = None, *, timeout: float | None = None
    ) -> ToolResult:
        """Dispatch one tool call, returning the :class:`ToolResult` for the model."""
        try:
            tool = self._tools.get(name)
            if tool is None:
                raise ToolCallFormatError(
                    f"Invalid action: function {name!r} is not defined in the tools list.\n"
                    f"Allowed functions should be one of: {self.names()}."
                )
            result = await tool.run(self._parse_arguments(name, args), timeout=timeout)
        except ToolCallFormatError as exc:
            return ToolResult(text=str(exc), status="format_error")
        except ToolError as exc:
            return ToolResult(text=f"Error: {exc}", status="error")
        except Exception:
            logger.exception("tool %r raised an unexpected error", name)
            raise
        return result if isinstance(result, ToolResult) else ToolResult(text=str(result))

    @staticmethod
    def _parse_arguments(name: str, raw_arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        """Coerce a call's ``arguments`` into a dict, or raise :class:`ToolCallFormatError`.

        The OpenAI API sends ``arguments`` as a JSON *string*; we also accept a
        mapping (local calls / tests) and ``None``. Anything not decoding to a JSON
        object is a format error.
        """
        if raw_arguments is None:
            return {}
        if isinstance(raw_arguments, dict):
            return dict(raw_arguments)
        if isinstance(raw_arguments, str):
            text = raw_arguments.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ToolCallFormatError(
                    f"Invalid action: could not parse arguments for {name!r} as JSON ({exc})."
                ) from None
        else:
            parsed = raw_arguments
        if not isinstance(parsed, dict):
            raise ToolCallFormatError(
                f"Invalid action: arguments for {name!r} must be a JSON object, got {type(parsed).__name__}."
            )
        return parsed

    async def close(self) -> None:
        """Close every tool (release open channels); never raises."""
        for tool in self._tools.values():
            try:
                await tool.close()
            except Exception:
                pass
