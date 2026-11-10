# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent personal identities; independent of browser identity providers.

Each mutation reads and commits one versioned authority snapshot in a SQLite
transaction. No process-local permission cache can outlive a policy change.
This module does not authorize legacy auth-file controllers.
"""

import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
import stat
import time
import uuid
from contextlib import contextmanager


SCHEMA_VERSION = 2
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_VERIFIER = re.compile(r"[0-9a-f]{64}\Z")


def new_secret():
    return "cap1_" + secrets.token_hex(32)


def verifier(secret):
    if not isinstance(secret, str) or not secret.startswith("cap1_") or not _VERIFIER.fullmatch(secret[5:]):
        raise PermissionError("invalid credential")
    return hashlib.sha256(secret.encode("ascii")).hexdigest()


def _name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError("name must contain 1–64 ASCII letters, digits, dots, underscores or hyphens")
    return value


def _id():
    return str(uuid.uuid4())


def _audit(state, actor, action, target, now):
    state.setdefault("_pending_audit", []).append((_id(), now, actor, action, target))


def _validate_state(state):
    try:
        if "telemetry" in state:
            from .telemetry_control import validate_state
            validate_state(state["telemetry"])
        if type(state["version"]) is not int or state["version"] not in (SCHEMA_VERSION, 3) or "_pending_audit" in state:
            raise ValueError()
        uuid.UUID(state["scheduler_id"])
        for key in ("users", "tokens", "groups", "pools", "operations"):
            if not isinstance(state[key], dict):
                raise ValueError()
        if "default" not in state["groups"]:
            raise ValueError()
        for user in state["users"].values():
            _name(user["name"])
            if user["role"] not in ("admin", "regular") or type(user["disabled"]) is not bool or not isinstance(user["groups"], list):
                raise ValueError()
            if any(not isinstance(group, str) or group not in state["groups"] for group in user["groups"]):
                raise ValueError()
        for token in state["tokens"].values():
            if token["user_id"] not in state["users"] or token["kind"] not in ("admin", "regular") or type(token["revoked"]) is not bool:
                raise ValueError()
            if not isinstance(token["verifier"], str) or not _VERIFIER.fullmatch(token["verifier"]):
                raise ValueError()
            expiry = token["expires_at"]
            if expiry is not None and (type(expiry) not in (int, float) or not math.isfinite(expiry)):
                raise ValueError()
        for group in state["groups"].values():
            if type(group["all_pools"]) is not bool or not isinstance(group["pools"], list) or any(not isinstance(p, str) for p in group["pools"]):
                raise ValueError()
        for pool in state["pools"].values():
            from .models import PoolConfig
            _name(pool["name"])
            if type(pool["draining"]) is not bool:
                raise ValueError()
            config = PoolConfig(**pool.get("config", {}))
            if config.gpu_slots > 2 ** 63 - 1:
                raise ValueError()
            limit = pool.get("max_nodes")
            if limit is not None and (type(limit) is not int or not 1 <= limit <= 2 ** 63 - 1):
                raise ValueError()
        if "nodes" in state and not isinstance(state["nodes"], dict):
            raise ValueError()
        for node in state.get("nodes", {}).values():
            if node["owner_user_id"] not in state["users"] or node["desired"] not in ("active", "draining", "removed", "reconfigure"):
                raise ValueError()
            if type(node["announced"]) is not bool or type(node.get("force", False)) is not bool:
                raise ValueError()
            if type(node["intent_at"]) not in (float, int) or not math.isfinite(node["intent_at"]) or node["intent_at"] < 0:
                raise ValueError()
            if not isinstance(node["verifier"], str) or not _VERIFIER.fullmatch(node["verifier"]):
                raise ValueError()
            from .identity_nodes import resources
            resources(node["resources"])
            if "purge" in node:
                purge = node["purge"]
                if (not isinstance(purge, dict) or purge.get("node_id") != node["node_id"]
                        or node["desired"] != "removed" or purge.get("state") != "REMOVED"
                        or purge.get("purged") is not True or purge.get("cleanup_confirmed") is not False
                        or not isinstance(purge.get("reason"), str) or not purge["reason"].strip()
                        or len(purge["reason"].encode("utf-8")) > 1024):
                    raise ValueError()
                resources(dict(node["resources"], gpu_uuids=purge.get("quarantined_gpu_uuids")))
                if not set(node.get("effective_resources", node["resources"])["gpu_uuids"]).issubset(purge["quarantined_gpu_uuids"]):
                    raise ValueError()
            if "effective_resources" in node:
                resources(node["effective_resources"])
            if "quarantine_release" in node:
                release = node["quarantine_release"]
                if ("purge" not in node or not isinstance(release, dict)
                        or release.get("confirmed_stopped") is not True
                        or release.get("confirmation_source") != "administrator"
                        or not isinstance(release.get("reason"), str) or not release["reason"].strip()
                        or len(release["reason"].encode("utf-8")) > 1024
                        or not isinstance(release.get("actor_user_id"), str)
                        or release["actor_user_id"] not in state["users"]
                        or type(release.get("released_at")) not in (int, float)
                        or not math.isfinite(release["released_at"]) or release["released_at"] < 0):
                    raise ValueError()
            if node["desired"] == "reconfigure":
                resources(node["pending_resources"])
        from .management_sso import validate
        validate(state)
    except (KeyError, TypeError, AttributeError, ValueError):
        raise ValueError("unsupported or invalid authority state") from None


def _authenticate(state, secret, now):
    digest = verifier(secret)
    for token_id, token in state["tokens"].items():
        if secrets.compare_digest(token["verifier"], digest):
            user = state["users"][token["user_id"]]
            if token["revoked"] or user["disabled"] or (
                token["expires_at"] is not None and now >= token["expires_at"]
            ):
                break
            return dict(user, user_id=token["user_id"], token_id=token_id,
                        token_kind=token["kind"],
                        is_admin=token["kind"] == "admin" and user["role"] == "admin")
    from .management_sso import authenticate_session
    return authenticate_session(state, secret, now)


def _admin(state, secret, now):
    actor = _authenticate(state, secret, now)
    if not actor["is_admin"]:
        raise PermissionError("admin credential required")
    return actor


class IdentityStore:
    """Open existing state only. Creation is an explicit, exclusive operation."""

    def __init__(self, path, clock=time.time):
        self.path = os.path.abspath(os.fspath(path))
        self.clock = clock
        info = os.lstat(self.path)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
            raise PermissionError("authority must be an owner-only regular file")
        # mode=rw is important: a missing state must never become a new instance.
        from pathlib import Path
        self.db = sqlite3.connect(Path(self.path).as_uri() + "?mode=rw", uri=True,
                                  isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA synchronous=FULL")
        try:
            self.snapshot()
            self.db.execute("SELECT event_id FROM audit LIMIT 0")
        except BaseException:
            self.db.close()
            raise

    @classmethod
    def create(cls, path, admin_name, admin_secret, clock=time.time):
        _name(admin_name)
        digest = verifier(admin_secret)
        # Parent security is the instance initializer's responsibility. O_EXCL
        # refuses an existing file or symlink, including a partial prior init.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        user_id, token_id = _id(), _id()
        state = {
            "version": SCHEMA_VERSION, "scheduler_id": _id(),
            "users": {user_id: {"name": admin_name, "role": "admin", "disabled": False, "groups": ["default"]}},
            "tokens": {token_id: {"user_id": user_id, "name": "bootstrap", "kind": "admin",
                                  "verifier": digest, "expires_at": None, "revoked": False}},
            "groups": {"default": {"all_pools": True, "pools": []}},
            "pools": {}, "operations": {},
        }
        with sqlite3.connect(path) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("CREATE TABLE authority (singleton INTEGER PRIMARY KEY CHECK(singleton=1), body TEXT NOT NULL)")
            db.execute("CREATE TABLE audit (event_id TEXT PRIMARY KEY, timestamp REAL NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, target TEXT NOT NULL)")
            db.execute("INSERT INTO authority VALUES (1, ?)", (json.dumps(state),))
            db.execute("INSERT INTO audit VALUES (?, ?, ?, ?, ?)", (_id(), clock(), user_id, "controller.init", state["scheduler_id"]))
        parent = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        return cls(path, clock)

    def close(self):
        self.db.close()

    def snapshot(self):
        row = self.db.execute("SELECT body FROM authority WHERE singleton=1").fetchone()
        if row is None:
            raise ValueError("missing authority state")
        state = json.loads(row[0])
        _validate_state(state)
        return state

    @contextmanager
    def _transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            state = self.snapshot()
            yield state
            for event in state.pop("_pending_audit", []):
                self.db.execute("INSERT INTO audit VALUES (?, ?, ?, ?, ?)", event)
            self.db.execute("UPDATE authority SET body=? WHERE singleton=1", (json.dumps(state, sort_keys=True),))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def authenticate(self, secret):
        return _authenticate(self.snapshot(), secret, self.clock())

    def require_admin(self, secret):
        return _admin(self.snapshot(), secret, self.clock())

    def register_token(self, operation_id, digest, name, username=None, bearer=None, expires_at=None, target_user_id=None):
        """The verifier is registration data, never proof of possession.

        An anonymous retry of a committed operation is rejected. The caller
        recovers using its already-saved new token (or its existing identity).
        """
        if not isinstance(operation_id, str) or str(uuid.UUID(operation_id)) != operation_id or not isinstance(digest, str) or not _VERIFIER.fullmatch(digest):
            raise ValueError("invalid registration operation")
        _name(name)
        if username is not None:
            _name(username)
        if expires_at is not None and (isinstance(expires_at, bool) or not isinstance(expires_at, (int, float))
                                       or not math.isfinite(expires_at)):
            raise ValueError("invalid expiry")
        if expires_at is not None:
            expires_at = float(expires_at)
        params = {"verifier": digest, "name": name, "username": username, "expires_at": expires_at}
        if target_user_id is not None:
            if not isinstance(target_user_id, str) or str(uuid.UUID(target_user_id)) != target_user_id:
                raise ValueError("invalid target identity")
            params["user_id"] = target_user_id
        with self._transaction() as state:
            now = self.clock()
            old = state["operations"].get(operation_id)
            actor = _authenticate(state, bearer, now) if bearer is not None else None
            if target_user_id is not None and (actor is None or actor["user_id"] != target_user_id):
                raise PermissionError("token target identity mismatch")
            if old is not None:
                if actor is None or actor["user_id"] != old["result"]["user_id"]:
                    raise PermissionError("registration recovery requires owner credential")
                if params != old["params"]:
                    raise ValueError("operation parameters differ")
                _audit(state, actor["user_id"], "token.recover", old["result"]["token_id"], now)
                return dict(old["result"])
            if expires_at is not None and expires_at <= now:
                raise ValueError("expiry must be in the future")
            from .management_sso import contains_verifier
            if contains_verifier(state, digest):
                raise ValueError("SSO credential cannot be reused as personal credential")
            if any(t["verifier"] == digest for t in state["tokens"].values()):
                raise ValueError("credential already registered")
            if any(node["verifier"] == digest for node in state.get("nodes", {}).values()):
                raise ValueError("node credential cannot be reused as personal credential")
            if actor is None:
                if username is None or any(u["name"] == username for u in state["users"].values()):
                    raise ValueError("registration unavailable")
                user_id = _id()
                state["users"][user_id] = {"name": username, "role": "regular", "disabled": False, "groups": ["default"]}
            else:
                user_id = actor["user_id"]
                if username is not None and username != actor["name"]:
                    raise PermissionError("cannot create another identity")
            token_id = _id()
            state["tokens"][token_id] = {"user_id": user_id, "name": name, "kind": "regular",
                                        "verifier": digest, "expires_at": expires_at, "revoked": False}
            result = {"operation_id": operation_id, "user_id": user_id, "token_id": token_id, "status": "registered"}
            state["operations"][operation_id] = {"params": params, "result": result}
            _audit(state, user_id, "token.create", token_id, now)
            return dict(result)

    def list_tokens(self, bearer):
        state = self.snapshot()
        actor = _authenticate(state, bearer, self.clock())
        return [dict(token_id=key, **{k: v for k, v in token.items() if k != "verifier"})
                for key, token in state["tokens"].items()
                if token["user_id"] == actor["user_id"] and (token["kind"] == "regular" or actor["is_admin"])]

    def revoke_token(self, bearer, token_id):
        with self._transaction() as state:
            actor = _authenticate(state, bearer, self.clock())
            token = state["tokens"].get(token_id)
            if token is None or token["user_id"] != actor["user_id"] or (token["kind"] == "admin" and not actor["is_admin"]):
                raise PermissionError("credential not accessible")
            if token["kind"] == "admin":
                raise ValueError("bootstrap admin replacement must precede revocation")
            token["revoked"] = True
            _audit(state, actor["user_id"], "token.revoke", token_id, self.clock())

    def set_group(self, bearer, name, all_pools=False, pools=()):
        _name(name)
        if type(all_pools) is not bool or not isinstance(pools, (list, tuple)):
            raise ValueError("invalid group policy")
        with self._transaction() as state:
            actor = _admin(state, bearer, self.clock())
            if any(pool not in state["pools"] for pool in pools):
                raise ValueError("unknown pool ID")
            state["groups"][name] = {"all_pools": all_pools, "pools": sorted(set(pools))}
            _audit(state, actor["user_id"], "group.set", name, self.clock())

    def delete_group(self, bearer, name):
        with self._transaction() as state:
            actor = _admin(state, bearer, self.clock())
            if name == "default":
                raise ValueError("default group cannot be deleted")
            if name not in state["groups"]:
                raise ValueError("unknown group")
            del state["groups"][name]
            for user in state["users"].values():
                user["groups"] = [group for group in user["groups"] if group != name]
            _audit(state, actor["user_id"], "group.delete", name, self.clock())

    def edit_user(self, bearer, user_id, groups=None, disabled=None):
        with self._transaction() as state:
            actor = _admin(state, bearer, self.clock())
            user = state["users"][user_id]
            if groups is not None:
                if not isinstance(groups, (list, tuple)) or any(group not in state["groups"] for group in groups):
                    raise ValueError("unknown group")
                user["groups"] = sorted(set(groups))
            if disabled is not None:
                if type(disabled) is not bool or (disabled and user["role"] == "admin"):
                    raise ValueError("cannot disable bootstrap admin")
                user["disabled"] = disabled
            _audit(state, actor["user_id"], "user.edit", user_id, self.clock())

    def create_pool(self, bearer, name, config=None, max_nodes=None):
        from .models import PoolConfig
        _name(name)
        if config is not None and (not isinstance(config, dict) or set(config) - {"pool_mode", "gpu_slots", "placement"}):
            raise ValueError("invalid pool configuration")
        config = PoolConfig(**(config or {})).to_dict()
        if config["gpu_slots"] > 2 ** 63 - 1:
            raise ValueError("GPU slot count is too large")
        if max_nodes is not None and (type(max_nodes) is not int or not 1 <= max_nodes <= 2 ** 63 - 1):
            raise ValueError("max_nodes must be a positive integer or null")
        with self._transaction() as state:
            actor = _admin(state, bearer, self.clock())
            if any(pool["name"] == name for pool in state["pools"].values()):
                raise ValueError("pool already exists")
            pool_id = _id()
            state["pools"][pool_id] = {"name": name, "draining": False, "config": config, "max_nodes": max_nodes}
            _audit(state, actor["user_id"], "pool.create", pool_id, self.clock())
            return pool_id

    def can_submit(self, bearer, pool_id):
        state = self.snapshot()
        actor = _authenticate(state, bearer, self.clock())
        pool = state["pools"].get(pool_id)
        if pool is None or pool["draining"]:
            return False
        return actor["is_admin"] or any(
            state["groups"][group]["all_pools"] or pool_id in state["groups"][group]["pools"]
            for group in actor["groups"] if group in state["groups"]
        )

    def require_owner(self, bearer, owner_user_id):
        actor = self.authenticate(bearer)
        if actor["user_id"] != owner_user_id and not actor["is_admin"]:
            raise PermissionError("resource not accessible")
        if actor["user_id"] != owner_user_id:
            with self._transaction() as state:
                _admin(state, bearer, self.clock())
                _audit(state, actor["user_id"], "admin.owner-access", owner_user_id, self.clock())
        return actor

    def audit_events(self, bearer, limit=100):
        self.require_admin(bearer)
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("audit limit must be between 1 and 1000")
        return [dict(zip(("event_id", "timestamp", "actor", "action", "target"), row)) for row in self.db.execute(
            "SELECT event_id,timestamp,actor,action,target FROM audit ORDER BY rowid DESC LIMIT ?", (limit,))]

    def pending_allowed(self, user_id, pool_id, token_id):
        state = self.snapshot()
        user = state["users"].get(user_id)
        pool = state["pools"].get(pool_id)
        token = state["tokens"].get(token_id) or state.get("web_auth", {}).get("sessions", {}).get(token_id)
        if not user or user["disabled"] or not pool or pool["draining"] or not token or token["user_id"] != user_id:
            return False
        # Accepted work retains its admission credential kind. Expiry/revocation
        # prevents new calls; dispatch still checks live account/group rights.
        return (token["kind"] in ("admin", "session") and user["role"] == "admin") or any(
            state["groups"][group]["all_pools"] or pool_id in state["groups"][group]["pools"]
            for group in user["groups"] if group in state["groups"])
