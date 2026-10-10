import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from uni_agent.gateway.backend import AgentHintBackend, configure_agent_hint_backend

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


class Rollout(dict):
    @property
    def name(self):
        return self["name"]


def _rollout(*, name="vllm", connector="OffloadingConnector", spec="PriorityGPUOffloadingSpec", encoded=False):
    transfer = {
        "kv_connector": connector,
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": spec,
            "spec_module_path": "uni_agent.gateway.kv_offload.priority_policy",
        },
    }
    return Rollout(
        name=name, engine_kwargs={"vllm": {"kv_transfer_config": json.dumps(transfer) if encoded else transfer}}
    )


@pytest.mark.parametrize("encoded", [False, True])
def test_matching_connector_exposes_capability_and_preserves_generation(encoded):
    backend = SimpleNamespace(generate=AsyncMock(return_value="output"))
    adapted = configure_agent_hint_backend(backend, _rollout(encoded=encoded))
    assert isinstance(adapted, AgentHintBackend)
    assert adapted.supports_agent_runtime_hint is True
    params = {"temperature": 0.5}
    assert asyncio.run(adapted.generate(request_id="request", sampling_params=params)) == "output"
    backend.generate.assert_awaited_once_with(request_id="request", sampling_params=params)


@pytest.mark.parametrize(
    "overrides", [{"name": "sglang"}, {"connector": "OtherConnector"}, {"spec": "CPUOffloadingSpec"}]
)
def test_unrelated_engine_or_connector_has_no_hint_capability(overrides):
    backend = object()
    assert configure_agent_hint_backend(backend, _rollout(**overrides)) is backend


@pytest.mark.parametrize("capability", [False, True])
def test_explicit_backend_capability_takes_precedence(capability):
    backend = SimpleNamespace(supports_agent_runtime_hint=capability)
    assert configure_agent_hint_backend(backend, _rollout()) is backend


def test_no_connector_keeps_original_backend():
    backend = object()
    assert configure_agent_hint_backend(backend, Rollout(name="vllm")) is backend
