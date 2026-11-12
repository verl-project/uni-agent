"""OpenCode 2.2.5 via a per-invocation standalone server and native v2 config."""

from __future__ import annotations

import json

from ..opencode.agent import OpenCodeAgent, OpenCodeConfig, _runtime_config
from ..registry import register_agent

OPENCODE2_VERSION = "2.2.5"


class OpenCode2Config(OpenCodeConfig):
    name: str = "opencode2"
    state_root: str = "/tmp/uni-agent-opencode2"


@register_agent("opencode2")
class OpenCode2Agent(OpenCodeAgent):
    """Pinned v2 transport; the Task still owns the Sandbox lifecycle.

    Host managed configuration retains OpenCode's precedence. Use a clean
    container for reproducible runs; HOME isolation does not disable MDM.
    """

    config_model = OpenCode2Config

    @staticmethod
    def _accept_version(version: str) -> bool:
        return version == f"opencode v{OPENCODE2_VERSION}"

    @staticmethod
    def _runtime_config(cfg):
        validated = _runtime_config(cfg)
        if "#" in cfg.model.model_name:
            raise ValueError("opencode2 model_name cannot contain a variant separator (#)")
        provider = validated["provider"]["uni-agent"]
        return {
            "providers": {
                "uni-agent": {
                    "package": "@opencode/ai/providers/openai-compatible",
                    "name": "Uni-Agent Gateway",
                    "settings": provider["options"],
                    "models": provider["models"],
                }
            },
            "model": validated["model"],
            "agents": {"title": {"disabled": True}, "summary": {"disabled": True}},
            "policies": [
                {"action": "provider.use", "resource": "*", "effect": "deny"},
                {"action": "provider.use", "resource": "uni-agent", "effect": "allow"},
                {"action": "plugin.use", "resource": "*", "effect": "deny"},
                {"action": "mcp.use", "resource": "*", "effect": "deny"},
                {"action": "skill.use", "resource": "*", "effect": "deny"},
                *[
                    {"action": "tool.use", "resource": f"{name}:*", "effect": "deny"}
                    for name in ("subagent", "question", "webfetch", "websearch", "schedule*", "image_generate")
                ],
            ],
            "permissions": [
                {"action": action, "resource": "*", "effect": effect}
                for action, effect in {
                    "*": "allow",
                    "subagent": "deny",
                    "question": "deny",
                    "webfetch": "deny",
                    "websearch": "deny",
                    "external_directory": "deny",
                }.items()
            ],
            "compaction": {"auto": False},
            "share": "disabled",
            "update": "disable",
            "snapshots": False,
            "warming": False,
        }

    @staticmethod
    def _environment(cfg, root, workspace, runtime_config):
        return {
            "HOME": f"{root}/home",
            "PWD": workspace,
            "TMPDIR": root,
            "XDG_CONFIG_HOME": f"{root}/config",
            "XDG_DATA_HOME": f"{root}/data",
            "XDG_STATE_HOME": f"{root}/state",
            "XDG_CACHE_HOME": f"{root}/cache",
            "OPENCODE_CONFIG": "",
            "OPENCODE_CONFIG_DIR": f"{root}/config/opencode",
            "OPENCODE_CONFIG_CONTENT": json.dumps(runtime_config),
            "OPENCODE_DB": f"{root}/opencode.db",
            "OPENCODE_TEST_HOME": f"{root}/home",
            "OPENCODE_CLI_CONFIG": "",
            "OPENCODE_CLI_CONFIG_CONTENT": "{}",
            "OPENCODE_CONFIG_PROJECT_DISABLE": "1",
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_DISABLE_CLAUDE_CODE": "1",
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
            "OPENCODE_MODELS_PATH": f"{root}/no-model-catalog.json",
            "OPENCODE_SIMULATE": "0",
            "OPENCODE_FILEWATCHER_DISABLE": "1",
            "OPENCODE_DISABLE_FFF": "1",
            "UNI_AGENT_OPENCODE_API_KEY": cfg.model.api_key,
        }

    @staticmethod
    def _argv(cfg, workspace, prompt):
        return [
            cfg.opencode_bin,
            "run",
            "--standalone",
            "--format",
            "json",
            "--model",
            f"uni-agent/{cfg.model.model_name}",
            "--agent",
            "build",
            "--title",
            "Uni-Agent rollout",
            "--",
            prompt,
        ]
