# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One durable client envelope: secret and retry parameters cannot diverge."""

import json
import math
import os
import stat
import uuid
import fcntl
from contextlib import contextmanager

from .identity import _name, new_secret, verifier


@contextmanager
def _private_parent(path):
    if ".." in os.fspath(path).split(os.sep):
        raise ValueError("credential path must not contain parent traversal")
    path = os.path.abspath(os.fspath(path))
    parent, name = os.path.split(path)
    if not name:
        raise ValueError("credential filename required")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        # Walk by descriptor so symlinked ancestors cannot redirect a write.
        for component in parent.split("/"):
            if component:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("credential parent must be an owner-only directory")
        yield fd, name
    finally:
        os.close(fd)


def prepare_registration(path, controller_url, username=None, name="default", expires_at=None, target_user_id=None):
    """Do not call the network until this returns successfully.

    The envelope is immutable; even after a failed/ambiguous network request,
    recovery reuses it. Registration success is queried at the controller, not
    inferred from local file existence or a client-written 'complete' flag.
    """
    _name(name)
    if username is not None:
        _name(username)
    if not isinstance(controller_url, str) or not 1 <= len(controller_url.encode("utf-8")) <= 8192:
        raise ValueError("invalid controller URL")
    if expires_at is not None:
        if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)) or not math.isfinite(expires_at):
            raise ValueError("invalid expiry")
        expires_at = float(expires_at)
    parameters = {"username": username, "name": name, "expires_at": expires_at}
    if target_user_id is not None:
        if not isinstance(target_user_id, str) or str(uuid.UUID(target_user_id)) != target_user_id:
            raise ValueError("invalid target identity")
        parameters["user_id"] = target_user_id
    return prepare_operation(path, controller_url, parameters)


def prepare_operation(path, controller_url, parameters):
    if not isinstance(controller_url, str) or not 1 <= len(controller_url.encode("utf-8")) <= 8192:
        raise ValueError("invalid controller URL")
    secret = new_secret()
    body = {"version": 1, "secret": secret, "controller_url": controller_url,
            "registration": dict(parameters, operation_id=str(uuid.uuid4()), verifier=verifier(secret))}
    encoded = (json.dumps(body, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > 16384:
        raise ValueError("credential envelope too large")
    with _private_parent(path) as (directory, filename):
        fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory)
    return body


def read_registration(path):
    with _private_parent(path) as (directory, filename):
        fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o077 or info.st_nlink != 1 or info.st_size > 16384):
                raise PermissionError("credential must be an owner-only, unlinked regular file")
            try:
                body = json.loads(stream.read(16385))
                if (type(body["version"]) is not int or body["version"] != 1 or verifier(body["secret"]) != body["registration"]["verifier"]
                        or not isinstance(body["controller_url"], str)):
                    raise ValueError()
                # A prior prepare may have failed while syncing. A successful
                # recovery read must establish durability before any retry.
                os.fsync(stream.fileno())
                os.fsync(directory)
                return body
            except (ValueError, KeyError, TypeError, PermissionError):
                raise ValueError("invalid credential envelope") from None


def lock_registration(path):
    with _private_parent(path) as (directory, filename):
        fd = os.open(filename, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    stream = os.fdopen(fd, "rb+")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
            raise PermissionError("invalid enrollment credential file")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another agent already uses this enrollment") from None
        return stream
    except BaseException:
        stream.close()
        raise
