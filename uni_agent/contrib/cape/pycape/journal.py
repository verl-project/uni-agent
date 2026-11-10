# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import queue
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path


class EventRecord(object):
    def __init__(self, timestamp, event_type, entity_id, state=None, reason=None, details=None):
        self.timestamp = timestamp
        self.event_type = event_type
        self.entity_id = entity_id
        self.state = state
        self.reason = reason
        self.details = dict(details or {})


class MemoryJournal(object):
    def __init__(self):
        self.records = []

    def append(self, event):
        self.records.append(event)

    def flush(self):
        pass

    @contextmanager
    def batch(self):
        yield

    def list_events(self):
        return list(self.records)

    def count(self, event_type=None):
        if event_type is None:
            return len(self.records)
        return sum(1 for record in self.records if record.event_type == event_type)


class SQLiteJournal(object):
    """Ordered, bounded SQLite journal with one connection-owning writer thread.

    ``request_submitted`` is committed alone and acknowledged before ``append``
    returns. Other events are queued in FIFO order and committed in batches. A
    clean ``flush``/``close`` drains them; an abrupt process loss can lose at
    most the bounded queue plus the writer's current batch, but never a request
    id whose submit call returned successfully.
    """

    def __init__(self, path, batch_size=64, queue_capacity=256, batch_interval=0.005):
        self.path = Path(path)
        self.batch_size = max(1, int(batch_size))
        self.queue_capacity = max(1, int(queue_capacity))
        self.batch_interval = max(0.0, float(batch_interval))
        self._queue = queue.Queue(maxsize=self.queue_capacity)
        self._local = threading.local()
        self._state_lock = threading.Lock()
        self._error = None
        self._closed = False
        self._commit_count = 0
        ready = _Reply()
        self._thread = threading.Thread(
            target=self._writer_main,
            args=(ready,),
            name="cape-journal-writer",
            daemon=True,
        )
        self._thread.start()
        ready.result()

    @property
    def pending(self):
        return list(getattr(self._local, "pending", ()))

    @property
    def batch_depth(self):
        return int(getattr(self._local, "depth", 0))

    @property
    def commit_count(self):
        with self._state_lock:
            return self._commit_count

    def append(self, event):
        self._check_open()
        if event.event_type == "request_submitted":
            self._enqueue_local_pending()
            reply = _Reply()
            self._put(("durable", [event], reply))
            reply.result()
            return
        if self.batch_depth:
            self._local.pending.append(event)
            if len(self._local.pending) >= self.batch_size:
                pending = self._local.pending
                self._local.pending = []
                self._put(("write", pending, None))
            return
        self._put(("write", [event], None))

    @contextmanager
    def batch(self):
        self._check_open()
        if self.batch_depth == 0:
            self._local.pending = []
        self._local.depth = self.batch_depth + 1
        try:
            yield
        finally:
            self._local.depth = self.batch_depth - 1
            if self.batch_depth == 0:
                self._enqueue_local_pending()

    def flush(self):
        self._check_open()
        self._enqueue_local_pending()
        reply = _Reply()
        self._put(("flush", None, reply))
        reply.result()

    def list_events(self):
        self._check_open()
        self._enqueue_local_pending()
        reply = _Reply()
        self._put(("list", None, reply))
        return reply.result()

    def count(self, event_type=None):
        self._check_open()
        self._enqueue_local_pending()
        reply = _Reply()
        self._put(("count", event_type, reply))
        return int(reply.result())

    def close(self):
        with self._state_lock:
            if self._closed:
                return
        self._enqueue_local_pending()
        reply = _Reply()
        self._put(("close", None, reply))
        reply.result()
        self._thread.join()
        with self._state_lock:
            self._closed = True
        self._raise_writer_error()

    def _enqueue_local_pending(self):
        pending = getattr(self._local, "pending", None)
        if pending:
            self._local.pending = []
            self._put(("write", pending, None))

    def _put(self, command):
        self._raise_writer_error()
        self._queue.put(command)
        self._raise_writer_error()

    def _check_open(self):
        with self._state_lock:
            closed = self._closed
        if closed:
            raise RuntimeError("journal is closed")
        self._raise_writer_error()

    def _raise_writer_error(self):
        with self._state_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("journal writer failed: {}".format(error))

    def _record_error(self, error):
        with self._state_lock:
            if self._error is None:
                self._error = error

    def _record_commit(self):
        with self._state_lock:
            self._commit_count += 1

    def _writer_main(self, ready):
        connection = None
        carry = None
        current_reply = None
        try:
            if self.path.parent:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(self.path))
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "timestamp REAL NOT NULL, "
                "event_type TEXT NOT NULL, "
                "entity_id TEXT NOT NULL, "
                "state TEXT, "
                "reason TEXT, "
                "details TEXT NOT NULL)"
            )
            connection.commit()
            ready.set_result(None)
            while True:
                command = carry if carry is not None else self._queue.get()
                carry = None
                kind, payload, reply = command
                current_reply = reply
                if kind == "write":
                    events = list(payload)
                    while len(events) < self.batch_size:
                        try:
                            next_command = self._queue.get(timeout=self.batch_interval)
                        except queue.Empty:
                            break
                        if next_command[0] != "write":
                            carry = next_command
                            break
                        events.extend(next_command[1])
                    self._insert(connection, events)
                    current_reply = None
                    continue
                if kind == "durable":
                    self._insert(connection, payload)
                    reply.set_result(None)
                    current_reply = None
                    continue
                if kind == "flush":
                    reply.set_result(None)
                    current_reply = None
                    continue
                if kind == "list":
                    rows = connection.execute(
                        "SELECT timestamp, event_type, entity_id, state, reason, details "
                        "FROM events ORDER BY id"
                    ).fetchall()
                    reply.set_result([
                        EventRecord(row[0], row[1], row[2], row[3], row[4], json.loads(row[5]))
                        for row in rows
                    ])
                    current_reply = None
                    continue
                if kind == "count":
                    if payload is None:
                        row = connection.execute("SELECT COUNT(*) FROM events").fetchone()
                    else:
                        row = connection.execute(
                            "SELECT COUNT(*) FROM events WHERE event_type = ?", (payload,)
                        ).fetchone()
                    reply.set_result(int(row[0]))
                    current_reply = None
                    continue
                if kind == "close":
                    reply.set_result(None)
                    current_reply = None
                    break
                raise RuntimeError("unknown journal command: {}".format(kind))
        except BaseException as exc:
            self._record_error(exc)
            ready.set_exception(exc)
            if current_reply is not None:
                current_reply.set_exception(exc)
            if carry is not None and carry[2] is not None:
                carry[2].set_exception(exc)
            while True:
                try:
                    command = self._queue.get_nowait()
                except queue.Empty:
                    break
                if command[2] is not None:
                    command[2].set_exception(exc)
        finally:
            if connection is not None:
                connection.close()

    def _insert(self, connection, events):
        if not events:
            return
        rows = [
            (
                event.timestamp,
                event.event_type,
                event.entity_id,
                event.state,
                event.reason,
                json.dumps(event.details, sort_keys=True),
            )
            for event in events
        ]
        connection.executemany(
            "INSERT INTO events(timestamp, event_type, entity_id, state, reason, details) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
        self._record_commit()


class _Reply(object):
    def __init__(self):
        self._ready = threading.Event()
        self._value = None
        self._error = None

    def set_result(self, value):
        if not self._ready.is_set():
            self._value = value
            self._ready.set()

    def set_exception(self, error):
        if not self._ready.is_set():
            self._error = error
            self._ready.set()

    def result(self):
        self._ready.wait()
        if self._error is not None:
            raise self._error
        return self._value
