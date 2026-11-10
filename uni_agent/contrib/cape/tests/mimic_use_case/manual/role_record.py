# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-secret local enrollment record, tied to the launcher's process record."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def process_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("write", "read"))
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--controller-url", required=True)
    parser.add_argument("--node-id")
    parser.add_argument("--pool")
    parser.add_argument("--allocation-id")
    args = parser.parse_args()
    path = args.pid_file.with_suffix(".json")
    if args.operation == "write":
        pid = os.getppid()
        if not all((args.node_id, args.pool, args.allocation_id)):
            parser.error("write requires node, pool and allocation identifiers")
        record = dict(version=1, pid=pid, sid=os.getsid(pid), start=process_start(pid),
                      controller_url=args.controller_url.rstrip("/"), node_id=args.node_id,
                      pool=args.pool, allocation_id=args.allocation_id)
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(record, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    else:
        record = json.loads(path.read_text())
        pid, sid, _label = args.pid_file.read_text().split()
        if (record["version"] != 1 or record["pid"] != int(pid)
                or record["sid"] != int(sid)
                or record["controller_url"] != args.controller_url.rstrip("/")):
            raise ValueError("role record does not match this process/controller")
        start = process_start(int(pid))
        if start is not None and start != record["start"]:
            raise ValueError("role process PID has been reused")
        for field in ("pool", "node_id"):
            value = record[field]
            if not isinstance(value, str) or not value or any(c.isspace() for c in value):
                raise ValueError("invalid role identifier")
            print(value)


if __name__ == "__main__":
    main()
