# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SSO bridges and browser sessions in the native, version-fenced authority."""
import copy
import secrets
import uuid
from .identity import new_secret, verifier, _audit


def guid(value):
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError("expected a canonical UUID")
    return value


def validate(state):
    auth = state.get("web_auth")
    if auth is None:
        return
    if state["version"] != 3 or not isinstance(auth, dict) or set(auth) != {"bridges", "sessions"}:
        raise ValueError("invalid SSO authority")
    for kind, rows in auth.items():
        if not isinstance(rows, dict):
            raise ValueError("invalid SSO records")
        for key, row in rows.items():
            guid(key)
            if row["user_id"] not in state["users"] or type(row["revoked"]) is not bool:
                raise ValueError("invalid SSO owner")
            from .identity import _VERIFIER
            if not isinstance(row["verifier"], str) or not _VERIFIER.fullmatch(row["verifier"]):
                raise ValueError("invalid SSO verifier")
            if kind == "bridges":
                guid(row["tenant_id"]); guid(row["client_id"])
                if type(row["auto_provision"]) is not bool:
                    raise ValueError("invalid SSO provisioning policy")
            else:
                import math
                if row["parent_id"] not in auth["bridges"] or type(row["expires_at"]) not in (int, float) or not math.isfinite(row["expires_at"]):
                    raise ValueError("invalid SSO session")
                user = state["users"][row["user_id"]]
                guid(user["sso_tenant"]); guid(user["sso_subject"])
                if user["sso_tenant"] != auth["bridges"][row["parent_id"]]["tenant_id"]:
                    raise ValueError("invalid SSO session scope")


def contains_verifier(state, digest):
    return any(row["verifier"] == digest for rows in state.get("web_auth", {}).values() for row in rows.values())


def active_bridge(state, row):
    owner = state["users"].get(row["user_id"], {})
    return not row["revoked"] and owner.get("disabled") is False and owner.get("role") == "admin"


def authenticate_session(state, secret, now):
    digest = verifier(secret)
    auth = state.get("web_auth", {"sessions": {}, "bridges": {}})
    for key, row in auth["sessions"].items():
        if not secrets.compare_digest(row["verifier"], digest):
            continue
        user = state["users"][row["user_id"]]
        if row["revoked"] or user["disabled"] or now >= row["expires_at"] or not active_bridge(state, auth["bridges"][row["parent_id"]]):
            break
        return dict(user, user_id=row["user_id"], token_id=key, token_kind="session",
                    is_admin=user["role"] == "admin")
    raise PermissionError("invalid credential")


def provision(state, tenant, subject, allow, target=None):
    matches = [key for key, user in state["users"].items()
               if user.get("sso_tenant") == tenant and user.get("sso_subject") == subject]
    if matches:
        if len(matches) != 1 or (target is not None and target != matches[0]):
            raise ValueError("SSO identity already bound")
        uid = matches[0]
    elif target is not None:
        if target not in state["users"] or state["users"][target].get("sso_subject") is not None:
            raise ValueError("SSO target unavailable")
        uid = target
        state["users"][uid].update(sso_tenant=tenant, sso_subject=subject)
    elif allow:
        uid = str(uuid.uuid4())
        state["users"][uid] = dict(name="sso-" + uid, role="regular", disabled=False, groups=[],
                                   sso_tenant=tenant, sso_subject=subject)
    else:
        raise PermissionError("SSO account not provisioned")
    if state["users"][uid]["disabled"]:
        raise PermissionError("SSO account unavailable")
    return uid


def call(store, action, token, body):
    from .identity import _admin, _authenticate
    from .management import public_token, public_user
    fields = {
        "session.sso": {"tenant_id", "client_id", "subject"},
        "session.revoke": set(),
        "sso.bridge.create": {"tenant_id", "client_id", "auto_provision"},
        "sso.bridge.revoke": {"bridge_id"},
        "sso.users.create": {"tenant_id", "subject"},
        "sso.users.bind": {"tenant_id", "subject", "user_id"},
    }
    if action not in fields or set(body) - fields[action] - {"operation_id"}:
        raise ValueError("unknown SSO parameter or action")
    with store._transaction() as state:
        now = store.clock()
        if action == "session.sso":
            auth = state.get("web_auth", {"bridges": {}, "sessions": {}})
            digest = verifier(token)
            pairs = [(key,row) for key,row in auth["bridges"].items()
                     if secrets.compare_digest(row["verifier"], digest) and active_bridge(state,row)]
            if len(pairs) != 1:
                raise PermissionError("SSO bridge credential required")
            parent, bridge = pairs[0]
            tenant, subject = guid(body.get("tenant_id")), guid(body.get("subject"))
            if tenant != bridge["tenant_id"] or body.get("client_id") != bridge["client_id"]:
                raise PermissionError("SSO identity scope mismatch")
            uid = provision(state, tenant, subject, bridge["auto_provision"])
            actor_id = parent
        else:
            actor = _authenticate(state, token, now)
            actor_id = actor["user_id"]
            if action == "session.revoke":
                if actor["token_kind"] != "session":
                    raise PermissionError("SSO session required")
                state["web_auth"]["sessions"][actor["token_id"]]["revoked"] = True
                _audit(state, actor_id, action, actor["token_id"], now)
                return {"revoked": True}
            _admin(state, token, now)
            if action == "sso.bridge.revoke":
                row = state.get("web_auth", {}).get("bridges", {}).get(body.get("bridge_id"))
                if row is None:
                    raise ValueError("unknown bridge")
                row["revoked"] = True
                _audit(state, actor_id, action, body["bridge_id"], now)
                return {"revoked": True}
            if action not in ("sso.bridge.create", "sso.users.create", "sso.users.bind"):
                raise ValueError("unsupported SSO action")
            tenant = guid(body.get("tenant_id"))
            if action == "sso.bridge.create":
                client = guid(body.get("client_id"))
                if type(body.get("auto_provision")) is not bool:
                    raise ValueError("auto_provision must be explicitly boolean")
            else:
                uid = provision(state, tenant, guid(body.get("subject")), True,
                                guid(body.get("user_id")) if action == "sso.users.bind" else None)
                state["version"] = 3
                _audit(state, actor_id, action, uid, now)
                return {"id": uid, "user": public_user(state, uid)}
        op = body.get("operation_id")
        if not isinstance(op, str) or not 1 <= len(op) <= 128:
            raise ValueError("operation_id required")
        key = "web-auth:" + actor_id + ":" + action + ":" + op
        old = state["operations"].get(key)
        if old:
            if old["params"] != body:
                raise ValueError("operation parameters differ")
            return dict(copy.deepcopy(old["result"]), already_issued=True)
        state["version"] = 3
        auth = state.setdefault("web_auth", {"bridges": {}, "sessions": {}})
        secret, identity = new_secret(), str(uuid.uuid4())
        if action == "sso.bridge.create":
            if sum(not row["revoked"] for row in auth["bridges"].values()) >= 8:
                raise ValueError("SSO bridge limit reached")
            row = dict(user_id=actor_id, verifier=verifier(secret), revoked=False,
                       tenant_id=tenant, client_id=client, auto_provision=body["auto_provision"])
            auth["bridges"][identity] = row
            result = public_token(identity, row)
            secret_key = "token"
        else:
            if sum(row["user_id"] == uid and not row["revoked"] and row["expires_at"] > now for row in auth["sessions"].values()) >= 20:
                raise ValueError("active session limit reached")
            row = dict(user_id=uid, verifier=verifier(secret), revoked=False, parent_id=parent,
                       expires_at=now+3600, auth_method="sso", kind="session")
            auth["sessions"][identity] = row
            result = dict(public_token(identity, row), user=public_user(state, uid, state["users"][uid]["role"] == "admin"))
            secret_key = "session_token"
        state["operations"][key] = {"params":copy.deepcopy(body), "result":copy.deepcopy(result)}
        _audit(state, actor_id, action, identity, now)
        return dict(result, **{secret_key:secret})
