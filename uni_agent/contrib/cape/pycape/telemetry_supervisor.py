# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Linux process ownership for an instance-local Prometheus, shared across init/start."""

import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import time
import uuid
from urllib.request import ProxyHandler, build_opener


def process_record(pid):
    try:
        root = Path("/proc") / str(pid)
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z" or root.stat().st_uid != os.getuid():
            return None
        return {"pid": pid, "start_ticks": fields[19],
                "argv": (root / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")}
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


class PrometheusSupervisor:
    def __init__(self, state_dir, scheduler_id, binary="prometheus", hostname=None):
        self.root = Path(state_dir).resolve(strict=True).parent / "prometheus"
        self.scheduler_id = scheduler_id
        self.binary = binary
        self.hostname = hostname or socket.getfqdn()
        if not self.hostname or any(c in self.hostname for c in "/?#@\r\n"):
            raise ValueError("invalid metrics hostname")
        self.child = None
        self._record = None
        self.query_url = None
        self.receiver_url = None

    def _prepare(self):
        try:
            self.root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("Prometheus directory must be private and owned by current user")
        lock = os.open(self.root / "owner.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(lock)
            raise
        return lock

    def _write(self, name, value):
        temporary = self.root / (".publish-" + uuid.uuid4().hex)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.root / name)
        finally:
            temporary.unlink(missing_ok=True)

    def _owned(self, record):
        if not record:
            return False
        expected = ["--config.file=" + str(self.root / "prometheus.yml"),
                    "--storage.tsdb.path=" + str(self.root / "data")]
        return all(arg in record["argv"] for arg in expected)

    def _discover(self):
        records = [process_record(int(path.name)) for path in Path("/proc").iterdir() if path.name.isdigit()]
        owned = [record for record in records if self._owned(record)]
        if len(owned) > 1:
            raise RuntimeError("multiple Prometheus processes claim this instance")
        return owned[0] if owned else None

    def _signal(self, record, sig):
        # pidfd pins process identity between validation and signal delivery.
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.syscall(434, record["pid"], 0)  # pidfd_open on Linux x86_64/aarch64
        if fd < 0:
            if ctypes.get_errno() == 3:  # ESRCH: already exited.
                return
            if ctypes.get_errno() == 38:  # Older kernels lack pidfds.
                current = process_record(record["pid"])
                if current and current["start_ticks"] == record["start_ticks"] and self._owned(current):
                    try:
                        os.kill(record["pid"], sig)
                    except ProcessLookupError:
                        pass
                return
            raise OSError(ctypes.get_errno(), "cannot pin Prometheus process")
        try:
            current = process_record(record["pid"])
            if current is None or current["start_ticks"] != record["start_ticks"] or not self._owned(current):
                return
            if libc.syscall(424, fd, sig, ctypes.c_void_p(), 0) != 0:  # pidfd_send_signal
                if ctypes.get_errno() == 3:
                    return
                raise OSError(ctypes.get_errno(), "cannot signal owned Prometheus")
        finally:
            os.close(fd)

    def _urls(self, record):
        address = next((arg.split("=", 1)[1] for arg in record["argv"] if arg.startswith("--web.listen-address=")), None)
        if address is None or not address.startswith("0.0.0.0:"):
            raise ValueError("invalid owned Prometheus listener")
        port = int(address.rsplit(":", 1)[1])
        self.query_url = "http://127.0.0.1:" + str(port)
        host = "[" + self.hostname + "]" if ":" in self.hostname else self.hostname
        self.receiver_url = "http://{}:{}/api/v1/write".format(host, port)

    def start(self):
        lock = self._prepare()
        try:
            identity = self.root / "instance.json"
            if identity.exists():
                if json.loads(identity.read_text())["scheduler_id"] != self.scheduler_id:
                    raise ValueError("Prometheus directory belongs to another scheduler")
            else:
                self._write("instance.json", json.dumps({"scheduler_id": self.scheduler_id}))
            record = self._discover()
            if record is None:
                self._write("prometheus.yml", "global:\n  scrape_interval: 5s\nscrape_configs: []\n")
                with socket.socket() as reservation:
                    reservation.bind(("0.0.0.0", 0))
                    port = reservation.getsockname()[1]
                self.child = subprocess.Popen([self.binary, "--config.file=" + str(self.root / "prometheus.yml"),
                    "--storage.tsdb.path=" + str(self.root / "data"), "--web.listen-address=0.0.0.0:" + str(port),
                    "--web.enable-remote-write-receiver", "--storage.tsdb.retention.time=7d", "--storage.tsdb.retention.size=1GB"],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True, env={k: v for k, v in os.environ.items() if not k.startswith("CAPE_")})
                record = process_record(self.child.pid)
                if not self._owned(record):
                    raise RuntimeError("Prometheus process did not start")
            self._record = record
            self._urls(record)
            opener = build_opener(ProxyHandler({}))
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                current = process_record(record["pid"])
                if current is None or current["start_ticks"] != record["start_ticks"]:
                    raise RuntimeError("Prometheus exited before ready")
                try:
                    with opener.open(self.query_url + "/-/ready", timeout=0.3) as response:
                        if response.status == 200:
                            self._write("process.json", json.dumps(dict(record, query_url=self.query_url, receiver_url=self.receiver_url)))
                            return {"state": "ready", "query_url": self.query_url, "receiver_url": self.receiver_url}
                except OSError:
                    pass
                time.sleep(0.05)
            raise RuntimeError("Prometheus readiness timeout")
        except BaseException:
            if self.child is not None and self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait(timeout=3)
            raise
        finally:
            os.close(lock)

    def stop(self):
        lock = self._prepare()
        try:
            identity = self.root / "instance.json"
            if not identity.exists() or json.loads(identity.read_text())["scheduler_id"] != self.scheduler_id:
                raise ValueError("Prometheus ownership mismatch")
            record = self._discover()
            if record is None:
                return
            self._signal(record, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                current = process_record(record["pid"])
                if current is None or current["start_ticks"] != record["start_ticks"]:
                    break
                time.sleep(0.05)
            else:
                self._signal(record, signal.SIGKILL)
            if self.child is not None:
                self.child.wait(timeout=3)
        finally:
            os.close(lock)
