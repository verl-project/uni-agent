"""Check replica registration and the selected server actor."""

import os
import subprocess
import sys
from unittest.mock import Mock

import pytest

from uni_agent.agent_aware_router.server import register_vllm_metadata_replica
from verl.workers.rollout import replica


@pytest.mark.cpu
@pytest.mark.level0
def test_foreign_registration_is_not_overwritten(monkeypatch):
    other = lambda: None
    monkeypatch.setattr(replica.RolloutReplicaRegistry, "_registry", {"vllm": other})
    with pytest.raises(RuntimeError, match="another vllm replica") as error:
        register_vllm_metadata_replica()
    assert repr(other) in str(error.value)
    assert replica.RolloutReplicaRegistry._registry["vllm"] is other


@pytest.mark.cpu
@pytest.mark.level0
def test_fresh_process_registration_is_explicit_lazy_and_idempotent():
    script = """
import sys
from verl.workers.rollout import replica
from uni_agent.agent_aware_router.server import register_vllm_metadata_replica
assert replica.RolloutReplicaRegistry._registry['vllm'] is replica._load_vllm
register_vllm_metadata_replica()
loader = replica.RolloutReplicaRegistry._registry['vllm']
assert callable(loader) and loader is not replica._load_vllm
register_vllm_metadata_replica()
assert replica.RolloutReplicaRegistry._registry['vllm'] is loader
assert 'uni_agent.agent_aware_router.server.http_server' not in sys.modules
assert 'vllm.v1.engine.async_llm' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], env=os.environ.copy(), check=True, timeout=60)


@pytest.mark.cpu
@pytest.mark.level0
def test_metadata_replica_uses_metadata_http_server(monkeypatch):
    from uni_agent.agent_aware_router.server import http_server

    monkeypatch.setattr(http_server.vLLMReplica, "__init__", lambda self: None)
    remote = Mock()
    monkeypatch.setattr(http_server.ray, "remote", remote)
    replica = http_server.KVMetadataReplica()
    remote.assert_called_once_with(http_server.KVMetadataHttpServer)
    assert replica.server_class is remote.return_value
