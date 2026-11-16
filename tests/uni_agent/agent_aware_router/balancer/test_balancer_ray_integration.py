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

"""Balancer protocol integration across the real veRL and Ray boundary."""

from __future__ import annotations

import os

import pytest
import ray

import uni_agent.agent_aware_router as agent_aware_router_package
from verl.workers.rollout.router import get_router_handle

ROUTER_CONFIG_PATH = os.path.join(
    os.path.dirname(agent_aware_router_package.__file__), "configs", "agent_aware_router.yaml"
)

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


@ray.remote
class _MockServer:
    def get_server_address(self):
        return ("127.0.0.1", 8000)


def _servers(*server_ids: str) -> dict:
    return {server_id: _MockServer.remote() for server_id in server_ids}


@pytest.fixture(scope="session")
def ray_runtime():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_actor_exposes_balancer_protocol(ray_runtime):
    servers = _servers("s0", "s1")
    router = get_router_handle(servers, router_config_path=ROUTER_CONFIG_PATH)

    status = ray.get(router.get_status.remote())
    assert status["provider"] == "CollectorManager"
    assert status["route_calls"] == 0
    assert set(status["servers"]) == set(servers)
    assert ray.get(router.require_acquire_fields.remote()) == ["prompt_ids"]
    assert ray.get(router.require_release_fields.remote()) == ["request_id"]

    server_id, server = ray.get(router.acquire_server.remote("request-1", [1, 2, 3]))
    assert server_id in servers
    assert server in servers.values()
    ray.get(router.release_server.remote(server_id, "request-1"))

    status = ray.get(router.get_status.remote())
    assert status["route_calls"] == 1
    assert status["total_inflight"] == 0


def test_actor_pool_mutations_are_visible(ray_runtime):
    router = get_router_handle(_servers("s0"), router_config_path=ROUTER_CONFIG_PATH)

    ray.get(router.add_servers.remote(_servers("s1")))
    assert set(ray.get(router.get_all_servers.remote())) == {"s0", "s1"}

    ray.get(router.remove_servers.remote(["s0"]))
    assert ray.get(router.get_all_servers.remote()) == ["s1"]
