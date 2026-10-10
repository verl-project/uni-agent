"""Backend capability adaptation for request-level agent runtime hints."""

from __future__ import annotations

import json


class AgentHintBackend:
    """Expose the configured connector's runtime-hint capability to Gateway."""

    supports_agent_runtime_hint = True

    def __init__(self, backend):
        self._backend = backend

    async def generate(self, *args, **kwargs):
        return await self._backend.generate(*args, **kwargs)


def configure_agent_hint_backend(backend, rollout_config):
    """Adapt verl clients using their engine config, without changing Gateway config.

    Custom backends may explicitly expose supports_agent_runtime_hint; that
    declaration takes precedence over inference from the managed engine config.
    """
    if type(getattr(backend, "supports_agent_runtime_hint", None)) is bool:
        return backend
    if rollout_config.name != "vllm":
        return backend
    engine_kwargs = rollout_config.get("engine_kwargs") or {}
    vllm_kwargs = engine_kwargs.get("vllm") or {}
    transfer = vllm_kwargs.get("kv_transfer_config") or {}
    if isinstance(transfer, str):
        transfer = json.loads(transfer)
    if not isinstance(transfer, dict) and not hasattr(transfer, "get"):
        raise ValueError("kv_transfer_config must be an object")
    extra = transfer.get("kv_connector_extra_config") or {}
    if (
        transfer.get("kv_connector") == "OffloadingConnector"
        and transfer.get("kv_role") in {"kv_both", "kv_producer", "kv_consumer"}
        and extra.get("spec_name") == "PriorityGPUOffloadingSpec"
        and extra.get("spec_module_path") == "uni_agent.gateway.kv_offload.priority_policy"
    ):
        return AgentHintBackend(backend)
    return backend
