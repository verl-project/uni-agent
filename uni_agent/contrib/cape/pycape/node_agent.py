# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

class InProcessNodeAgent(object):
    """Small node-agent API facade for the embeddable V0 controller.

    It keeps the node-agent execution boundary visible without requiring a network
    daemon. Production deployments can replace this with an RPC client that
    calls the same controller methods.
    """

    def __init__(self, controller, node_id, pool, resources, capabilities=None, allocation_id=None, allocation_deadline=None, drain_before_sec=0, cleanup_buffer_sec=0, runtime_cache_config="default", join_token=None, heartbeat_token=None, pool_mode="exclusive", gpu_slots=1, placement="pack"):
        self.controller = controller
        self.node_id = node_id
        self.pool = pool
        self.resources = resources
        self.capabilities = capabilities
        self.allocation_id = allocation_id
        self.allocation_deadline = allocation_deadline
        self.drain_before_sec = drain_before_sec
        self.cleanup_buffer_sec = cleanup_buffer_sec
        self.runtime_cache_config = runtime_cache_config
        self.join_token = join_token
        self.heartbeat_token = heartbeat_token
        self.pool_mode = pool_mode
        self.gpu_slots = gpu_slots
        self.placement = placement

    def register(self):
        return self.controller.register_node(
            node_id=self.node_id,
            pool=self.pool,
            resources=self.resources,
            capabilities=self.capabilities,
            allocation_id=self.allocation_id,
            allocation_deadline=self.allocation_deadline,
            drain_before_sec=self.drain_before_sec,
            cleanup_buffer_sec=self.cleanup_buffer_sec,
            runtime_cache_config=self.runtime_cache_config,
            join_token=self.join_token,
            pool_mode=self.pool_mode,
            gpu_slots=self.gpu_slots,
            placement=self.placement,
        )

    def heartbeat(self, state=None):
        return self.controller.heartbeat(self.node_id, state=state, token=self.heartbeat_token)

    def collect_status(self):
        node = self.controller.get_node(self.node_id, token=self.heartbeat_token)
        return {
            "node_id": node.node_id,
            "state": node.state,
            "active_sessions": list(node.active_sessions),
            "available_cpu": node.available_cpu(),
            "available_memory": node.available_memory(),
        }

    def drain(self, reason="node_agent_drain"):
        return self.controller.drain_node(self.node_id, reason=reason, token=self.heartbeat_token)
