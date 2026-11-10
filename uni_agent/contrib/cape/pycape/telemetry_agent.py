# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional nonblocking bridge to the shared native metrics service."""
import json
import os
import subprocess
import threading


class MetricsLink:
    def __init__(self, controller_url, node_id, epoch, pool, gpu_uuids, token):
        self._lock = threading.Lock()
        self._status = {"state": "error", "reason": "metrics_unavailable"}
        self._process = None
        self._reader = None
        try:
            binary = os.environ.get("CAPE_METRIC_HELPER_BIN", "rscape")
            env = {key: value for key, value in os.environ.items()
                   if key not in ("CAPE_TOKEN", "CAPE_NODE_TOKEN", "CAPE_ADMIN_TOKEN", "CAPE_JOIN_TOKEN")}
            env["TOKIO_WORKER_THREADS"] = "2"
            self._process = subprocess.Popen([binary, "__metric-link"], stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                             text=True, env=env)
            boot = {"controller_url": controller_url, "node_id": node_id, "agent_epoch": epoch,
                    "pool_id": pool, "gpu_uuids": list(gpu_uuids), "token": token, "parent_pid": os.getpid()}
            self._bootstrap = json.dumps(boot) + "\n"
            self._reader = threading.Thread(target=self._read, name="cape-metrics-link", daemon=True)
            self._reader.start()
        except OSError:
            self.close()

    def _read(self):
        try:
            self._process.stdin.write(self._bootstrap)
            self._process.stdin.close()
            self._bootstrap = None
            for line in self._process.stdout:
                if len(line) > 16384:
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        continue
                    value = {key: value[key] for key in ("service_id", "url", "incarnation", "applied_version", "state", "reason", "collection_state") if key in value}
                    with self._lock:
                        self._status = value
                except ValueError:
                    continue
        except OSError:
            pass
        finally:
            self._bootstrap = None
            with self._lock:
                self._status = dict(self._status, state="error", reason="metrics_link_stopped")

    def status(self):
        with self._lock:
            return dict(self._status)

    def close(self):
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            if self._reader is not None:
                self._reader.join(timeout=1)
            if process.stdout is not None:
                process.stdout.close()
