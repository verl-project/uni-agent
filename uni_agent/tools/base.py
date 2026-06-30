"""Tool layer: host-side tools, each a schema plus a (possibly stateful) ``run``.

The agent runs *outside* the task image. A :class:`Tool` is the single
agent-facing unit: a schema (what the model sees) plus an async :meth:`run` that
drives the container through the :class:`~uni_agent.sandbox.SandboxBackend` data
plane (``exec`` / ``read_file`` / ``write_file`` / ...). A tool is constructed
with its sandbox and *owns* whatever state it needs:

* **Stateless tools** (e.g. the editor) just read/write the data plane; their
  only state is incidental (the editor's undo history lives on the instance).
* **Stateful tools** (e.g. ``shell``) hold a live channel -- a persistent
  shell/browser/desktop handle -- as a private attribute, opened lazily on first
  use and torn down in :meth:`close`. There is no separate "session" layer: the
  stateful part of a tool *is* the channel it keeps open.

Every :meth:`run` returns a normalized :class:`Observation` (text and/or an
image plus structured extras), so a multimodal agent loop never special-cases a
modality: a shell tool fills ``text``, a browser/desktop tool fills ``image``.
Bind a selection of tools to a sandbox with :class:`Toolbox`.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from ..sandbox import SandboxBackend


@dataclasses.dataclass
class Observation:
    """A normalized result of one tool call.

    One shape for every modality: terminal text, a screenshot, and any
    structured extras (exit code, url, cursor position) funnel through here so
    the agent loop never special-cases the transport. ``str(obs)`` yields the
    text channel for convenience.
    """

    text: str | None = None
    image: bytes | None = None
    structured: dict | None = None
    meta: dict | None = None

    def __str__(self) -> str:
        return self.text if self.text is not None else ""


class ToolError(Exception):
    """Raised by a tool for a user-facing failure.

    :meth:`Toolbox.call` turns this into an ``"Error: ..."`` observation handed
    back to the model, instead of crashing the rollout. Use it for bad arguments
    / expected failures; let genuine bugs propagate.
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
    per-call args the model fills, so the default :meth:`schema` works) and
    implement :meth:`run`. A tool also declares its *construction* options via
    :attr:`config_model`: a tool is built with its sandbox plus keyword args, and
    the base auto-parses those kwargs through ``config_model`` into ``self.config``
    (typed, defaulted, validated). e.g. the shell tool's ``env_vars`` /
    ``command_timeout`` come from its config. Stateful tools open a channel lazily
    and release it in :meth:`close`.
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
    async def run(self, args: dict[str, Any]) -> Observation:
        """Execute the call and return an :class:`Observation` for the model.

        Drive the container through ``self.sandbox`` (and any channel the tool
        holds). Raise :class:`ToolError` for user-facing failures.
        """
        ...

    async def close(self) -> None:
        """Release any state the tool holds (open channels). No-op by default."""
        return None


TOOL_REGISTRY: dict[str, type[Tool]] = {}


def register_tool(name: str):
    """Class decorator: register ``cls`` under the registry key ``name``.

    The registry key (used in config and :func:`get_tool`) is independent of the
    model-facing :attr:`Tool.name`. If the class doesn't set its own ``name``, the
    registry key is stamped as the default; a class that sets ``name`` explicitly
    keeps it -- e.g. ``stateful_shell`` registers a tool the model still sees as
    ``shell``.
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

    Extra ``kwargs`` are forwarded to the tool constructor (auto-parsed into the
    tool's ``config_model``), e.g. ``get_tool("stateful_shell", sb, command_timeout=120)``.
    """
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {name!r}")
    return TOOL_REGISTRY[name](sandbox, **kwargs)


class Toolbox:
    """A set of instantiated tools bound to one sandbox for a rollout.

    Holds tool *instances* (each already bound to the sandbox and owning its own
    state) so stateful tools keep state across calls. Exposes the model-facing
    :meth:`schemas`, a single :meth:`call` dispatch the agent loop drives, and an
    ordered :meth:`close`.
    """

    def __init__(self, tools: list[Tool]):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self._tools[tool.name] = tool

    @classmethod
    def from_specs(cls, specs: list[dict[str, Any]], *, sandbox: SandboxBackend) -> Toolbox:
        """Build a toolbox from ``{name, ...kwargs}`` config entries, bound to ``sandbox``.

        Each entry is a mapping with a ``name`` (a TOOL_REGISTRY key) plus that
        tool's construction kwargs (auto-parsed into its ``config_model``), e.g.
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

    async def call(self, name: str, args: dict[str, Any] | None = None) -> Observation:
        """Dispatch one tool call, returning the :class:`Observation` for the model."""
        tool = self._tools.get(name)
        if tool is None:
            return Observation(text=f"Error: unknown tool {name!r}. Available: {', '.join(self._tools)}")
        try:
            obs = await tool.run(dict(args or {}))
        except ToolError as exc:
            return Observation(text=f"Error: {exc}")
        return obs if isinstance(obs, Observation) else Observation(text=str(obs))

    async def close(self) -> None:
        """Close every tool (release open channels); never raises."""
        for tool in self._tools.values():
            try:
                await tool.close()
            except Exception:
                pass
