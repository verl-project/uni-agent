# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os

from .credential_file import prepare_registration, read_registration


def add_user_parser(subcommands):
    user = subcommands.add_parser("user")
    commands = user.add_subparsers(dest="user_command", required=True)
    for verb in ("whoami", "permissions"):
        _connection(commands.add_parser(verb))
    _connection(commands.add_parser("pools"))
    nodes = commands.add_parser("nodes")
    _connection(nodes)
    nodes.add_argument("--pool")
    tokens = commands.add_parser("token").add_subparsers(dest="token_command", required=True)
    create = tokens.add_parser("create")
    _connection(create)
    create.add_argument("--user")
    create.add_argument("--name", default="default")
    create.add_argument("--expires-at", type=float)
    create.add_argument("--token-out", required=True)
    resume = tokens.add_parser("resume")
    _connection(resume)
    resume.add_argument("--token-out", required=True)
    _connection(tokens.add_parser("list"))
    revoke = tokens.add_parser("revoke")
    _connection(revoke)
    revoke.add_argument("token_id")


def _connection(parser):
    parser.add_argument("--controller-url")
    parser.add_argument("--token", default=os.environ.get("CAPE_TOKEN"))
    parser.add_argument("--token-file")


def run_user(args, request, resolve_url, http_error, stdout, stderr):
    if args.user_command == "nodes":
        result = request("GET", args.controller_url, "/v1/identity/nodes", token=args.token,
                         query={"pool": args.pool} if args.pool else None)
    elif args.user_command == "pools":
        result = request("GET", args.controller_url, "/v1/identity/pools", token=args.token)
    elif args.user_command != "token":
        result = request("GET", args.controller_url, "/v1/identity/me", token=args.token)
    elif args.token_command == "list":
        result = request("GET", args.controller_url, "/v1/identity/tokens", token=args.token)
    elif args.token_command == "revoke":
        result = request("POST", args.controller_url, "/v1/identity/tokens/{}/revoke".format(args.token_id), token=args.token)
    else:
        resume = args.token_command == "resume"
        if resume:
            envelope = read_registration(args.token_out)
        else:
            actor = request("GET", args.controller_url, "/v1/identity/me", token=args.token) if args.token else None
            envelope = prepare_registration(args.token_out, resolve_url(args.controller_url), args.user, args.name, args.expires_at,
                                            target_user_id=actor["user_id"] if actor else None)
        url = envelope["controller_url"]
        if not url.startswith(("http://", "https://")):
            raise ValueError("this credential is not a resumable user registration")
        if args.controller_url and resolve_url(args.controller_url) != url:
            raise ValueError("registration controller differs from saved operation")
        payload = envelope["registration"]
        try:
            if resume:
                try:
                    result = request("POST", url, "/v1/identity/tokens", payload=payload, token=envelope["secret"])
                except http_error as error:
                    if error.status != 403:
                        raise
                    if payload.get("user_id") is None and payload.get("username") is None:
                        raise ValueError("saved operation has no target identity binding")
                    result = request("POST", url, "/v1/identity/tokens", payload=payload, token=args.token)
            else:
                result = request("POST", url, "/v1/identity/tokens", payload=payload, token=args.token)
            if (result.get("operation_id") != payload["operation_id"] or result.get("status") != "registered"
                    or not result.get("token_id") or not result.get("user_id")):
                raise ValueError("invalid registration acknowledgement")
        except Exception as error:
            # Never emit transport exceptions: some transports include headers
            # or response bodies. Preserve the envelope even on rejection.
            rejected = isinstance(error, http_error) and 400 <= error.status < 500 and not resume
            json.dump({"outcome": "rejected" if rejected else "unknown", "token_file": args.token_out,
                       "operation_id": payload["operation_id"], "recovery": "user token resume"}, stderr)
            stderr.write("\n")
            return 1 if rejected else 3
        result = dict(result, token_file=args.token_out)
    json.dump(result, stdout, sort_keys=True)
    stdout.write("\n")
    return 0


def add_admin_parsers(commands):
    metric = commands.add_parser("metric").add_subparsers(dest="metric_admin_command", required=True)
    for action in ("get", "set"):
        leaf = metric.add_parser(action)
        _connection(leaf)
        leaf.add_argument("--node-id")
        if action == "set":
            leaf.add_argument("--expected-version", type=int)
            leaf.add_argument("--interval-seconds", type=float)
            leaf.add_argument("--stale-periods", type=int)
            for flag in ("enabled", "basic", "profiling"):
                leaf.add_argument("--" + flag, choices=("on", "off"))
    nodes = commands.add_parser("node").add_subparsers(dest="admin_node_command", required=True)
    purge = nodes.add_parser("purge", description="Retire a LOST node from pool inventory; preserve data/history and GPU quarantine. Does not kill remote processes.")
    _connection(purge)
    purge.add_argument("node_id")
    purge.add_argument("--reason", required=True)
    release = nodes.add_parser("release-quarantine", description="Release only this purged node's GPU quarantine source after an administrator confirms its processes stopped. Does not restore old credentials or certify workspace cleanup.")
    _connection(release)
    release.add_argument("node_id")
    release.add_argument("--reason", required=True)
    release.add_argument("--confirm-stopped", action="store_true")
    audit = commands.add_parser("audit")
    _connection(audit)
    audit.add_argument("--limit", type=int, default=100)
    pools = commands.add_parser("pool").add_subparsers(dest="pool_command", required=True)
    create = pools.add_parser("create")
    _connection(create)
    create.add_argument("name")
    create.add_argument("--max-nodes", type=int)
    create.add_argument("--pool-mode", choices=("exclusive", "shared"), default="exclusive")
    create.add_argument("--gpu-slots", type=int, default=1)
    create.add_argument("--placement", choices=("pack", "spread"), default="pack")
    edit_pool = pools.add_parser("edit")
    _connection(edit_pool)
    edit_pool.add_argument("pool")
    edit_pool.add_argument("--name")
    limit = edit_pool.add_mutually_exclusive_group()
    limit.add_argument("--max-nodes", type=int)
    limit.add_argument("--unlimited-nodes", action="store_true")
    edit_pool.add_argument("--pool-mode", choices=("exclusive", "shared"))
    edit_pool.add_argument("--gpu-slots", type=int)
    edit_pool.add_argument("--placement", choices=("pack", "spread"))
    for verb in ("drain", "remove"):
        command = pools.add_parser(verb)
        _connection(command)
        command.add_argument("pool")
        if verb == "drain":
            command.add_argument("--force", action="store_true")
    groups = pools.add_parser("group").add_subparsers(dest="group_command", required=True)
    _connection(groups.add_parser("list"))
    set_group = groups.add_parser("set")
    _connection(set_group)
    set_group.add_argument("name")
    set_group.add_argument("--pool", action="append", default=[])
    set_group.add_argument("--all-pools", action="store_true")
    remove = groups.add_parser("remove")
    _connection(remove)
    remove.add_argument("name")
    users = commands.add_parser("user").add_subparsers(dest="admin_user_command", required=True)
    _connection(users.add_parser("list"))
    edit = users.add_parser("edit")
    _connection(edit)
    edit.add_argument("user_id")
    membership = edit.add_mutually_exclusive_group()
    membership.add_argument("--group", action="append")
    membership.add_argument("--clear-groups", action="store_true")
    status = edit.add_mutually_exclusive_group()
    status.add_argument("--disable", action="store_true")
    status.add_argument("--enable", action="store_true")
    remove = users.add_parser("remove")
    _connection(remove)
    remove.add_argument("user_id")


def run_admin(args, request, stdout):
    if args.admin_command == "metric":
        body = {"action": "metric-" + args.metric_admin_command, "node_id": args.node_id}
        if args.metric_admin_command == "set":
            changes = {key: getattr(args, key) for key in ("interval_seconds", "stale_periods") if getattr(args, key) is not None}
            changes.update({key: getattr(args, key) == "on" for key in ("enabled", "basic", "profiling") if getattr(args, key) is not None})
            body.update(changes=changes, expected_version=args.expected_version)
    elif args.admin_command == "node":
        body = {"action": "node-" + args.admin_node_command, "node_id": args.node_id, "reason": args.reason}
        if args.admin_node_command == "release-quarantine":
            if not args.confirm_stopped:
                raise ValueError("release-quarantine requires --confirm-stopped")
            body["confirm_stopped"] = True
    elif args.admin_command == "audit":
        body = {"action": "audit", "limit": args.limit}
    elif args.admin_command == "pool":
        if args.pool_command == "create":
            body = {"action": "pool-create", "name": args.name, "max_nodes": args.max_nodes,
                    "config": {"pool_mode": args.pool_mode, "gpu_slots": args.gpu_slots, "placement": args.placement}}
        elif args.pool_command == "edit":
            changes = {}
            if args.name is not None:
                changes["name"] = args.name
            if args.max_nodes is not None or args.unlimited_nodes:
                changes["max_nodes"] = args.max_nodes
            if any(value is not None for value in (args.pool_mode, args.gpu_slots, args.placement)):
                pools = request("GET", args.controller_url, "/v1/identity/pools", token=args.token)
                pool = next((p for p in pools if args.pool in (p["pool_id"], p["name"])), None)
                if pool is None:
                    raise ValueError("unknown pool")
                config = dict(pool["config"])
                for key, value in (("pool_mode", args.pool_mode), ("gpu_slots", args.gpu_slots), ("placement", args.placement)):
                    if value is not None:
                        config[key] = value
                changes["config"] = config
            body = {"action": "pool-edit", "pool": args.pool, "changes": changes}
        elif args.pool_command in ("drain", "remove"):
            body = {"action": "pool-" + args.pool_command, "pool": args.pool, "force": getattr(args, "force", False)}
        elif args.group_command == "list":
            body = {"action": "group-list"}
        elif args.group_command == "set":
            body = {"action": "group-set", "name": args.name, "all_pools": args.all_pools, "pools": args.pool}
        else:
            body = {"action": "group-delete", "name": args.name}
    else:
        if args.admin_user_command == "list":
            result = request("POST", args.controller_url, "/v1/identity/admin", token=args.token, payload={"action": "user-list"})
            json.dump(result, stdout, sort_keys=True)
            stdout.write("\n")
            return 0
        body = {"action": "user-edit", "user_id": args.user_id}
        if args.admin_user_command == "remove":
            body["disabled"] = True
        else:
            if args.clear_groups or args.group is not None:
                body["groups"] = [] if args.clear_groups else args.group
            if args.disable or args.enable:
                body["disabled"] = args.disable
    result = request("POST", args.controller_url, "/v1/identity/admin", token=args.token, payload=body)
    json.dump(result, stdout, sort_keys=True)
    stdout.write("\n")
    return 0


def add_node_action_parsers(commands):
    edit = commands.add_parser("edit")
    _connection(edit)
    edit.add_argument("node_id")
    edit.add_argument("--cpu-cores", type=float)
    edit.add_argument("--memory-gb", type=float)
    gpus = edit.add_mutually_exclusive_group()
    gpus.add_argument("--gpu-uuid", action="append")
    gpus.add_argument("--no-gpus", action="store_true")
    for action in ("drain", "exit"):
        parser = commands.add_parser(action)
        _connection(parser)
        parser.add_argument("node_id")
        if action == "drain":
            parser.add_argument("--force", action="store_true")
