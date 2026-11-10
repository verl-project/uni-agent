# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller authorization adapter for explicit persistent instances."""

from .models import Role
from . import identity_nodes


class ManagedAuthority:
    managed = True

    def __init__(self, store):
        self.store = store

    def pool_id(self, value):
        pools = self.store.snapshot()["pools"]
        if value in pools:
            return value
        for key, pool in pools.items():
            if pool["name"] == value:
                return key
        raise ValueError("unknown pool")

    def user_id_for_token(self, token):
        return self.store.authenticate(token)["user_id"]

    def user_has_role(self, token, pool, role):
        if role == Role.NODE_AGENT:
            try:
                identity_nodes.authenticate_agent(self.store, token, pool=pool)
                return True
            except PermissionError:
                self.store.authenticate(token)
                return False
        try:
            actor = self.store.authenticate(token)
        except PermissionError:
            identity_nodes.authenticate_agent(self.store, token, pool=pool)
            return False
        if role == Role.SUBMITTER:
            return self.store.can_submit(token, self.pool_id(pool))
        # Legacy VIEWER/OPERATOR grant private cross-owner access. Public
        # inventory is separately authenticated, never mapped to these roles.
        return actor["is_admin"]

    def authenticate_user(self, token, pool, required_role):
        if required_role == Role.NODE_AGENT:
            return identity_nodes.authenticate_agent(self.store, token, pool=pool)["node_id"]
        if not self.user_has_role(token, pool, required_role):
            raise PermissionError("missing required permission")
        return self.user_id_for_token(token)

    def require_owner(self, token, owner):
        return self.store.require_owner(token, owner)

    def validate_join_token(self, *args, **kwargs):
        raise PermissionError("persistent instances require node enrollment credentials")
