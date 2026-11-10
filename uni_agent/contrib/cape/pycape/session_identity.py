# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logical identity and accepted-submission-only client session selection."""

import fcntl
import hashlib
import json
import os
import stat
import uuid
import unicodedata

from .credential_file import _private_parent


def validate_session_id(value):
    if (not isinstance(value, str) or not value or len(value.encode("utf-8")) > 256
            or any(c.isspace() or unicodedata.category(c) == "Cc" for c in value)):
        raise ValueError("session_id must be 1-256 UTF-8 bytes without whitespace/control characters")
    return value


class SessionSelection:
    """Serialize local selection through acceptance, without storing credentials."""

    def __init__(self, identity, explicit=None, directory=None, project=None):
        self.scheduler = identity["scheduler_id"]
        self.user = identity["user_id"]
        if not isinstance(self.scheduler, str) or not self.scheduler or not isinstance(self.user, str) or not self.user:
            raise ValueError("authenticated controller and user identity required")
        self.project = os.path.realpath(project or os.getcwd())
        self.explicit = validate_session_id(explicit) if explicit is not None else None
        root = directory or os.environ.get("CAPE_SESSION_STATE_DIR") or os.path.join(
            os.environ.get("XDG_STATE_HOME", os.path.join(os.path.realpath(os.path.expanduser("~")), ".local/state")), "cape", "sessions")
        self.root = os.path.abspath(root)
        key = hashlib.sha256("\0".join((self.scheduler, self.user, self.project)).encode("utf-8")).hexdigest()
        self.filename = key + ".json"
        self.lock = None
        self.parent = None

    def __enter__(self):
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        self.parent = _private_parent(os.path.join(self.root, self.filename))
        self.directory, _ = self.parent.__enter__()
        try:
            self.lock = os.open(self.filename + ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                                0o600, dir_fd=self.directory)
            self._check(self.lock)
            fcntl.flock(self.lock, fcntl.LOCK_EX)
            previous = None
            try:
                fd = os.open(self.filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.directory)
            except FileNotFoundError:
                pass
            else:
                with os.fdopen(fd, "rb") as stream:
                    self._check(stream.fileno())
                    body = json.load(stream)
                if (body.get("version") != 1 or body.get("scheduler_id") != self.scheduler
                        or body.get("user_id") != self.user or body.get("project") != self.project):
                    raise ValueError("session cache context mismatch")
                previous = validate_session_id(body["session_id"])
            self.session_id = self.explicit or previous or "session-" + uuid.uuid4().hex
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    @staticmethod
    def _check(fd):
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077
                or info.st_nlink != 1 or info.st_size > 16384):
            raise PermissionError("session cache must be an owner-only regular file")

    def accepted(self, response):
        if response.get("session_id") != self.session_id or not response.get("request_id"):
            raise ValueError("submission response does not confirm selected logical session")
        temporary = self.filename + "." + uuid.uuid4().hex + ".tmp"
        body = dict(version=1, scheduler_id=self.scheduler, user_id=self.user,
                    project=self.project, session_id=self.session_id)
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.directory)
            with os.fdopen(fd, "w") as stream:
                json.dump(body, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.filename, src_dir_fd=self.directory, dst_dir_fd=self.directory)
            os.fsync(self.directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=self.directory)
            except FileNotFoundError:
                pass

    def __exit__(self, *error):
        if self.lock is not None:
            os.close(self.lock)
            self.lock = None
        if self.parent is not None:
            self.parent.__exit__(*error)
            self.parent = None
