# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared image-workspace configuration contract; filesystem operations are node-local."""

import copy
import posixpath
import re


CONFIG_SCHEMA_VERSION = 1
DEFAULT_IMAGE_SIZE_BYTES = 10 * 1024 ** 3
DEFAULT_MAX_IMAGE_SIZE_BYTES = 1024 ** 4
MIN_IMAGE_SIZE_BYTES = 32 * 1024 ** 2
MAX_INTEGER = 2 ** 63 - 1


def workspace_bind_conflict(bind_target, workspace_target):
    target = "/" + posixpath.normpath(bind_target).lstrip("/")
    return target == "/" or target == workspace_target or workspace_target.startswith(target + "/")


def workspace_selection(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - {"id", "target"}:
        raise ValueError("workspace must be an object containing only id and target")
    identity = value.get("id")
    if identity is not None and (not isinstance(identity, str) or re.fullmatch(r"ws-[0-9a-f]{32}", identity) is None):
        raise ValueError("invalid workspace.id")
    try:
        target = WorkspaceConfig(image_root=value.get("target", "/workspace")).image_root
    except ValueError as error:
        raise ValueError("workspace target: " + str(error))
    return {"id": identity, "target": target}


class WorkspaceConfig:
    FIELDS = frozenset((
        "version", "image_root", "image_size_bytes", "max_image_size_bytes",
        "bytes_per_inode", "mount_timeout_seconds", "maintenance",
    ))

    def __init__(self, version=0, image_root="/tmp",
                 image_size_bytes=DEFAULT_IMAGE_SIZE_BYTES,
                 max_image_size_bytes=DEFAULT_MAX_IMAGE_SIZE_BYTES,
                 bytes_per_inode=16384, mount_timeout_seconds=30, maintenance=None):
        from .workspace_policy import MaintenancePolicy
        self.maintenance = None if maintenance is None else MaintenancePolicy(maintenance).to_dict()
        self.version = self._integer("version", version, 0, MAX_INTEGER)
        self.image_size_bytes = self._integer("image_size_bytes", image_size_bytes, MIN_IMAGE_SIZE_BYTES, MAX_INTEGER)
        self.max_image_size_bytes = self._integer("max_image_size_bytes", max_image_size_bytes, MIN_IMAGE_SIZE_BYTES, MAX_INTEGER)
        if self.image_size_bytes > self.max_image_size_bytes:
            raise ValueError("image_size_bytes exceeds max_image_size_bytes")
        self.bytes_per_inode = self._integer("bytes_per_inode", bytes_per_inode, 1024, 1024 ** 3)
        self.mount_timeout_seconds = self._integer("mount_timeout_seconds", mount_timeout_seconds, 1, 300)
        if not isinstance(image_root, str) or not image_root.startswith("/"):
            raise ValueError("image_root must be an absolute node-side path")
        if any(ord(char) < 32 or ord(char) == 127 for char in image_root) or ".." in image_root.split("/"):
            raise ValueError("image_root contains forbidden path components")
        self.image_root = posixpath.normpath(image_root)
        # POSIX permits implementation-defined // roots; the wire contract does not.
        self.image_root = "/" + self.image_root.lstrip("/")
        if self.image_root in ("/", "/dev", "/run") or self.image_root.split("/")[1] in ("proc", "sys"):
            raise ValueError("image_root must not be a filesystem root or virtual kernel tree")

    @staticmethod
    def _integer(name, value, minimum, maximum):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("{} must be an integer in [{}, {}]".format(name, minimum, maximum))
        return value

    def to_dict(self):
        return {name: getattr(self, name) for name in sorted(self.FIELDS) if name != "maintenance" or self.maintenance is not None}

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("workspace configuration must be an object")
        unknown = sorted(set(values) - cls.FIELDS)
        if unknown:
            raise ValueError("unknown workspace configuration fields: " + ", ".join(unknown))
        return cls(**values)

    def updated(self, changes):
        if not isinstance(changes, dict):
            raise ValueError("workspace configuration update must be an object")
        if "version" in changes:
            raise ValueError("workspace configuration version is controller-owned")
        if not changes:
            raise ValueError("workspace configuration update must not be empty")
        values = self.to_dict()
        values.update(changes)
        values["version"] = self.version + 1
        return self.from_dict(values)


class WorkspaceConfigStore:
    """Pure configuration state; controllers supply authorization and journaling."""

    def __init__(self):
        self._pools = {}

    @staticmethod
    def _identity(value, name):
        if not isinstance(value, str) or not value:
            raise ValueError("{} must be a nonempty string".format(name))

    @staticmethod
    def _effective(record, node_id):
        values = dict(record["defaults"])
        if node_id is not None:
            overrides = record["nodes"].get(node_id, {})
            policy = dict(values.get("maintenance") or {})
            policy.update(overrides.get("maintenance") or {})
            values.update(overrides)
            if policy:
                values["maintenance"] = policy
        values["version"] = record["version"]
        return WorkspaceConfig.from_dict(values)

    def get(self, pool, node_id=None):
        self._identity(pool, "pool")
        if node_id is not None:
            self._identity(node_id, "node_id")
        record = self._pools.get(pool, {"version": 0, "defaults": {}, "nodes": {}})
        return self._effective(record, node_id)

    @staticmethod
    def _validate_overrides(changes):
        if not isinstance(changes, dict):
            raise ValueError("workspace configuration overrides must be an object")
        unknown = sorted(set(changes) - (WorkspaceConfig.FIELDS - {"version"}))
        if unknown:
            raise ValueError("invalid workspace override fields: " + ", ".join(unknown))

    def _validate_record(self, record):
        if not isinstance(record, dict) or set(record) != {"version", "defaults", "nodes"}:
            raise ValueError("invalid workspace configuration record")
        self._validate_overrides(record["defaults"])
        if not isinstance(record["nodes"], dict):
            raise ValueError("workspace node overrides must be an object")
        self._effective(record, None)
        for node_id, overrides in record["nodes"].items():
            self._identity(node_id, "node_id")
            self._validate_overrides(overrides)
            self._effective(record, node_id)

    def update(self, pool, changes, node_id=None):
        self.get(pool, node_id)
        self._validate_overrides(changes)
        if not changes:
            raise ValueError("workspace configuration update must not be empty")
        record = copy.deepcopy(self._pools.get(pool, {"version": 0, "defaults": {}, "nodes": {}}))
        target = record["defaults"] if node_id is None else record["nodes"].setdefault(node_id, {})
        if "maintenance" in changes:
            from .workspace_policy import MaintenancePolicy
            if not isinstance(changes["maintenance"], dict):
                raise ValueError("maintenance policy must be an object")
            MaintenancePolicy(changes["maintenance"])
            changes = dict(changes, maintenance=dict(target.get("maintenance") or {}, **changes["maintenance"]))
        target.update(changes)
        record["version"] += 1
        self._validate_record(record)
        self._pools[pool] = record
        return copy.deepcopy(record)

    def restore(self, pool, record):
        self._identity(pool, "pool")
        self._validate_record(record)
        self._pools[pool] = copy.deepcopy(record)

    def record(self, pool):
        self._identity(pool, "pool")
        return copy.deepcopy(self._pools.get(pool, {"version": 0, "defaults": {}, "nodes": {}}))
