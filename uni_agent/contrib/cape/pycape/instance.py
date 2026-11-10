# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict instance lifecycle primitives, not a legacy auth-file migration."""

import fcntl
import os
import stat

from .identity import IdentityStore, _name, verifier, _admin, _audit
from .journal import SQLiteJournal


class Instance:
    def __init__(self, directory, store, lock):
        self.directory = directory
        self.store = store
        self.lock = lock
        self.owner_pid = os.getpid()
        self.database = os.path.join(directory, "state.sqlite3")

    @staticmethod
    def initialize(directory, admin_name, admin_secret):
        _name(admin_name)
        verifier(admin_secret)
        directory = os.path.abspath(os.fspath(directory))
        # mkdir is exclusive: init never repairs, truncates or adopts an old
        # directory. An interrupted init is retained for explicit inspection.
        os.mkdir(directory, 0o700)
        lock_path = os.path.join(directory, "controller.lock")
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        database = os.path.join(directory, "state.sqlite3")
        store = IdentityStore.create(database, admin_name, admin_secret)
        try:
            scheduler_id = store.snapshot()["scheduler_id"]
        finally:
            store.close()
        journal = SQLiteJournal(database)
        journal.close()
        for path in (directory, os.path.dirname(directory)):
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        return scheduler_id

    @classmethod
    def open(cls, directory, admin_secret):
        directory = os.path.abspath(os.fspath(directory))
        info = os.lstat(directory)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("instance must be an owner-only directory")
        fd = os.open(os.path.join(directory, "controller.lock"), os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
        lock = os.fdopen(fd, "rb+")
        store = None
        try:
            info = os.fstat(lock.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
                raise PermissionError("invalid instance lock")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            store = IdentityStore(os.path.join(directory, "state.sqlite3"))
            store.require_admin(admin_secret)
            if not store.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'").fetchone():
                raise ValueError("incomplete instance: event journal is missing")
            with store._transaction() as state:
                actor = _admin(state, admin_secret, store.clock())
                _audit(state, actor["user_id"], "controller.start", state["scheduler_id"], store.clock())
            return cls(directory, store, lock)
        except BaseException:
            if store is not None:
                store.close()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
            raise

    def close(self):
        try:
            self.store.close()
        finally:
            if not self.lock.closed and os.getpid() == self.owner_pid:
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()
