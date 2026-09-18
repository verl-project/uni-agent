# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""KvEventsHttpServer port-assignment behavior (CPU-only, no engine).

``__init__`` needs ray actors and a live engine, so these bypass it with
``object.__new__`` and set only the attributes the kv-events code reads —
the same trick as verl's test_mtp_hybrid_sleep_acceptance_on_cpu.py.
``run_server``/``run_headless`` need a real vLLM engine and are covered by
e2e runs, not here.
"""

import socket
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.level0, pytest.mark.cpu]

vllm = pytest.importorskip("vllm")  # noqa: F841  (http_server import chain needs it)

from uni_agent.agent_aware_router.server.http_server import KvEventsHttpServer  # noqa: E402


def _make_server(dp_size: int = 2, server_address: str = "10.0.0.1"):
    server = object.__new__(KvEventsHttpServer)
    server.config = SimpleNamespace(data_parallel_size=dp_size)
    server._server_address = server_address
    server.replica_rank = 0
    server._kv_events_socks = []
    server._kv_events_endpoints = None
    return server


def _bind(addr: str, port: int, reuse: bool = False) -> socket.socket:
    s = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
    if reuse:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((addr, port))
    return s


def test_assign_writes_block_back_and_records_endpoints():
    server = _make_server(dp_size=2)
    kv = {"enable_kv_cache_events": True}
    server._assign_kv_events_ports(kv)

    endpoint, replay = kv["endpoint"], kv["replay_endpoint"]
    base = int(endpoint.rsplit(":", 1)[1])
    replay_port = int(replay.rsplit(":", 1)[1])
    # Replay base sits dp_size above the endpoint base; each expands by +dp_rank in vLLM.
    assert replay_port == base + 2
    # The whole 2*dp_size block is held while assigned.
    assert len(server._kv_events_socks) == 4
    # Getter list uses the router-visible server address; publisher/topic unset -> None.
    assert server.get_kv_events_endpoints() == [f"10.0.0.1:{base}", f"10.0.0.1:{replay_port}", None, None]
    # Held block really blocks a fresh bind (incl. REUSEADDR, as vLLM's ZMQ binds).
    with pytest.raises(OSError):
        _bind("0.0.0.0", base, reuse=True).close()


def test_assign_passes_publisher_topic_through():
    server = _make_server(dp_size=1)
    kv = {"enable_kv_cache_events": True, "publisher": "zmq", "topic": "kv-events"}
    server._assign_kv_events_ports(kv)
    endpoints = server.get_kv_events_endpoints()
    assert endpoints[2:] == ["zmq", "kv-events"]
    # Values are reported verbatim and left untouched in the config.
    assert kv["publisher"] == "zmq" and kv["topic"] == "kv-events"


def test_assign_preserves_explicit_host():
    server = _make_server(dp_size=1)
    kv = {"enable_kv_cache_events": True, "endpoint": "tcp://127.0.0.1:1234", "replay_endpoint": "tcp://127.0.0.1:1234"}
    server._assign_kv_events_ports(kv)
    base = int(kv["endpoint"].rsplit(":", 1)[1])
    assert kv["endpoint"] == f"tcp://127.0.0.1:{base}"
    assert kv["replay_endpoint"] == f"tcp://127.0.0.1:{base + 1}"


def test_assign_skips_non_tcp_endpoints():
    server = _make_server(dp_size=2)
    kv = {"enable_kv_cache_events": True, "endpoint": "ipc://@kv", "replay_endpoint": "ipc://@kv-replay"}
    server._assign_kv_events_ports(kv)
    # No tcp endpoint to allocate: nothing recorded, config untouched.
    assert server._kv_events_endpoints is None
    assert server._kv_events_socks == []
    assert kv == {"enable_kv_cache_events": True, "endpoint": "ipc://@kv", "replay_endpoint": "ipc://@kv-replay"}


def test_assign_mixed_tcp_and_ipc_allocates_tcp_only():
    server = _make_server(dp_size=2)
    kv = {"enable_kv_cache_events": True, "endpoint": "ipc://@kv", "replay_endpoint": "tcp://*:0"}
    server._assign_kv_events_ports(kv)
    # ipc endpoint untouched, tcp one rewritten; block is dp_size wide.
    assert kv["endpoint"] == "ipc://@kv"
    assert kv["replay_endpoint"].startswith("tcp://*:")
    assert len(server._kv_events_socks) == 2
    # Only the tcp endpoint lands in the getter list (3 elements, not 4).
    endpoints = server.get_kv_events_endpoints()
    assert len(endpoints) == 3
    assert endpoints[0] == f"10.0.0.1:{kv['replay_endpoint'].rsplit(':', 1)[1]}"


def test_release_is_idempotent_and_unblocks():
    server = _make_server(dp_size=2)
    kv = {"enable_kv_cache_events": True}
    server._assign_kv_events_ports(kv)
    base = int(kv["endpoint"].rsplit(":", 1)[1])
    server._release_kv_events_ports()
    server._release_kv_events_ports()  # second call is a no-op, not an error
    assert server._kv_events_socks == []
    # After release the ports are bindable again, as vLLM's ZMQ binder needs.
    _bind("0.0.0.0", base, reuse=True).close()
    _bind("0.0.0.0", base + 1).close()


def test_getter_is_none_before_assign():
    assert _make_server().get_kv_events_endpoints() is None


def test_preprocess_gates_on_enable_kv_cache_events():
    server = _make_server(dp_size=1)
    engine_kwargs = {"kv-events-config": {"enable_kv_cache_events": False}}
    server._preprocess_engine_kwargs(engine_kwargs)
    assert server._kv_events_endpoints is None
    assert "endpoint" not in engine_kwargs["kv-events-config"]

    engine_kwargs = {"kv-events-config": {"enable_kv_cache_events": True}}
    server._preprocess_engine_kwargs(engine_kwargs)
    assert "endpoint" in engine_kwargs["kv-events-config"]
    assert server.get_kv_events_endpoints() is not None
