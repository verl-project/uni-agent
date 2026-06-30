"""Environment configuration models.

Mirrors the YAML shape::

    environment:
      sandbox:
        provider: local | vefaas | modal
        runtime_timeout: 3600
      tools:
        - name: stateful_shell  # registry key (model sees it as `shell`)
          command_timeout: 120  # shell's per-command timeout (seconds)
          env_vars:
            PAGER: cat
        - name: str_replace_editor

Each ``tools`` entry is a mapping with ``name`` plus that tool's construction
kwargs (co-located, no separate block). Each tool declares its own kwargs schema
(``Tool.config_model``) and the env auto-parses the entry's kwargs into it. A
tool's own timeout (e.g. the shell's ``command_timeout``) lives in its kwargs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SandboxConfig(BaseModel):
    provider: Literal["local", "vefaas", "modal"] = Field(
        default="local", description="Which sandbox backend to run the task in."
    )
    runtime_timeout: float = Field(
        default=3600.0,
        description="Max sandbox runtime/lifetime (seconds) before it is killed; used by remote providers.",
    )
    image: str = Field(default="python:3.12", description="Container image for remote providers (modal / vefaas).")
    sandbox_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra provider-specific kwargs forwarded to the sandbox constructor "
        "(e.g. modal: app_name, modal_sandbox_kwargs).",
    )

    model_config = ConfigDict(extra="forbid")


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
