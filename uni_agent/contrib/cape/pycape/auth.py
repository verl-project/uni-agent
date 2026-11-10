# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from .models import Role


class JoinToken(object):
    def __init__(self, token, pool_id, allocation_id, expires_at=None, max_nodes=None, allowed_node_prefixes=(), allowed_runtime_configs=()):
        self.token = token
        self.pool_id = pool_id
        self.allocation_id = allocation_id
        self.expires_at = expires_at
        self.max_nodes = max_nodes
        self.allowed_node_prefixes = tuple(allowed_node_prefixes)
        self.allowed_runtime_configs = tuple(allowed_runtime_configs)
        self.used_nodes = 0
        self.used_node_ids = set()

    def allows_node(self, node_id):
        return not self.allowed_node_prefixes or any(node_id.startswith(prefix) for prefix in self.allowed_node_prefixes)

    def allows_runtime_config(self, runtime_cache_config):
        return not self.allowed_runtime_configs or runtime_cache_config in self.allowed_runtime_configs


class UserToken(object):
    def __init__(self, token, user_id, roles_by_pool, expires_at=None):
        self.token = token
        self.user_id = user_id
        self.roles_by_pool = roles_by_pool
        self.expires_at = expires_at


class TokenAuthority(object):
    def __init__(self, clock=None):
        self.clock = clock or time.time
        self.user_tokens = {}
        self.join_tokens = {}

    def add_user_token(self, token, user_id, pool, roles, expires_at=None):
        self.user_tokens[token] = UserToken(token, user_id, {pool: set(roles)}, expires_at)

    def add_join_token(self, join_token):
        self.join_tokens[join_token.token] = join_token

    def authenticate_user(self, token, pool, required_role):
        record = self._get_live_user_token(token)
        roles = record.roles_by_pool.get(pool, set())
        if required_role not in roles and Role.ADMIN not in roles:
            raise PermissionError("missing required role")
        return record.user_id

    def user_has_role(self, token, pool, role):
        record = self._get_live_user_token(token)
        roles = record.roles_by_pool.get(pool, set())
        return role in roles or Role.ADMIN in roles

    def user_id_for_token(self, token):
        return self._get_live_user_token(token).user_id

    def _get_live_user_token(self, token):
        record = self.user_tokens.get(token)
        if record is None:
            raise PermissionError("invalid user token")
        now = self.clock()
        if record.expires_at is not None and now >= record.expires_at:
            raise PermissionError("expired user token")
        return record

    def validate_join_token(self, token, pool, node_id, runtime_cache_config):
        record = self.join_tokens.get(token)
        if record is None:
            raise PermissionError("invalid join token")
        now = self.clock()
        if record.expires_at is not None and now >= record.expires_at:
            raise PermissionError("expired join token")
        if record.pool_id != pool:
            raise PermissionError("join token pool mismatch")
        if node_id not in record.used_node_ids and record.max_nodes is not None and record.used_nodes >= record.max_nodes:
            raise PermissionError("join token node limit reached")
        if not record.allows_node(node_id):
            raise PermissionError("node id not allowed by join token")
        if not record.allows_runtime_config(runtime_cache_config):
            raise PermissionError("runtime cache config not allowed by join token")
        if node_id not in record.used_node_ids:
            record.used_node_ids.add(node_id)
            record.used_nodes += 1
        return record
