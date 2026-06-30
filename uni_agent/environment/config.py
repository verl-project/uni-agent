"""Environment configuration models.

Mirrors the YAML shape::

    environment:
      sandbox:
        provider: local | modal  # SANDBOX_REGISTRY key
        runtime_timeout: 3600
      tools:
        - name: stateful_shell  # registry key (model sees it as `shell`)
          command_timeout: 120  # shell's per-command timeout (seconds)
          env_vars:
            PAGER: cat
        - name: str_replace_editor

The ``sandbox`` block is the sandbox layer's :class:`SandboxConfig` (re-exported
here for convenience). Each ``tools`` entry is a mapping with ``name`` plus that
tool's construction kwargs (co-located, no separate block). Each tool declares
its own kwargs schema (``Tool.config_model``) and the env auto-parses the
entry's kwargs into it. A tool's own timeout (e.g. the shell's
``command_timeout``) lives in its kwargs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..sandbox import SandboxConfig

__all__ = ["SandboxConfig", "EnvironmentConfig"]


class EnvironmentConfig(BaseModel):
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tools to expose; each entry is a mapping with `name` plus that "
        "tool's kwargs, e.g. {'name': 'shell', 'env_vars': {...}}.",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("tools")
    @classmethod
    def _require_name(cls, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for entry in tools:
            if not entry.get("name"):
                raise ValueError(f"each tools entry must have a 'name': {entry!r}")
        return tools
