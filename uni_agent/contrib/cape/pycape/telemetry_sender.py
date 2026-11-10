# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded batch adapter to Telegraf's remote-write serializer, never a WAL."""

import json
import os
import subprocess
import tempfile
from urllib.parse import urlsplit

from .telemetry_contract import finite


def receiver_url(value):
    if not isinstance(value, str) or len(value) > 4096 or any(ord(c) < 33 for c in value):
        raise ValueError("invalid telemetry receiver URL")
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or parsed.path != "/api/v1/write"):
        raise ValueError("invalid telemetry receiver URL")
    parsed.port  # Reject malformed or out-of-range ports.
    return value


def batch_configuration(source_path, receiver, timeout_seconds):
    receiver_url(receiver)
    finite(timeout_seconds, 0.1, 30)
    return '''[agent]
  omit_hostname = true
  metric_batch_size = 10000
  metric_buffer_limit = 10000
  skip_processors_after_aggregators = true
[[inputs.file]]
  files = [%s]
  data_format = "prometheus"
[[outputs.http]]
  url = %s
  timeout = "%ss"
  data_format = "prometheusremotewrite"
  [outputs.http.headers]
    Content-Type = "application/x-protobuf"
    Content-Encoding = "snappy"
    X-Prometheus-Remote-Write-Version = "0.1.0"
''' % (json.dumps(os.fspath(source_path)), json.dumps(receiver), timeout_seconds)


class BatchSender:
    """Call only from a telemetry worker, not the heartbeat/scheduler thread.

    One process and private temporary snapshot per batch. Failed batches have no
    surviving queue and are never retried by a later invocation. Scheduling an
    overlapping call is the service's responsibility; the service must serialize it.
    """

    def __init__(self, receiver, binary="telegraf", timeout_seconds=2, runtime_dir=None):
        self.receiver = receiver_url(receiver)
        self.binary = os.fspath(binary)
        self.timeout = finite(timeout_seconds, 0.1, 30)
        self.runtime_dir = runtime_dir

    def send(self, exposition):
        if not isinstance(exposition, str) or len(exposition.encode("utf-8")) > 4 * 1024 * 1024:
            raise ValueError("telemetry batch exceeds limit")
        if len([line for line in exposition.splitlines() if line and not line.startswith("#")]) > 10000:
            raise ValueError("telemetry series limit exceeded")
        if not exposition.strip():
            return {"outcome": "empty"}
        try:
            with tempfile.TemporaryDirectory(prefix="cape-metric-batch-", dir=self.runtime_dir) as root:
                source = os.path.join(root, "snapshot.prom")
                with open(source, "x", encoding="utf-8") as stream:
                    stream.write(exposition)
                config = batch_configuration(source, self.receiver, self.timeout)
                # No shell, no inherited CAPE credentials, no disk-buffer configuration.
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith(("CAPE_", "INFLUX_", "TELEGRAF_"))}
                result = subprocess.run([self.binary, "--once", "--config", "/dev/stdin"],
                                        input=config, text=True, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, env=env, timeout=self.timeout + 1)
                return {"outcome": "sent" if result.returncode == 0 else "dropped",
                        "reason": None if result.returncode == 0 else "sender_failed"}
        except subprocess.TimeoutExpired:
            return {"outcome": "dropped", "reason": "sender_timeout"}
        except OSError:
            return {"outcome": "dropped", "reason": "sender_unavailable"}
