"""KVConnectorFactory smoke: our connector class is reachable by name."""
from llm_router.connector import register_with_vllm
from llm_router.connector.connector import MooncakeKVConnector


def test_register_idempotent():
    register_with_vllm()
    register_with_vllm()  # second call must not raise


def test_factory_resolves_connector_class():
    register_with_vllm()
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    cls = KVConnectorFactory.get_connector_class_by_name("MooncakeKVConnector")
    assert cls is MooncakeKVConnector
