#!/usr/bin/env python3
"""Read-only Linux host sampler for an explicitly owned sandbox experiment.

The dynamic --pid-file is an atomically replaced JSON object, for example:
{"run_id": "trial-1", "processes": [{"pid": 123, "starttime_ticks": 456,
 "label": "provider", "include_descendants": true}]}

starttime_ticks is field 22 in /proc/PID/stat. Bare PIDs are rejected. A root's
discovered descendants remain tracked after reparenting until that root is
removed from the registry. Optional discovery requires BOTH an exact executable
allowlist and an absolute, dedicated --service-root found in argv or cwd.
Command lines are inspected only for discovery and are never written to output.

Node counters include unrelated workloads. Process counters cover only selected
identities and are cumulative since process start; sum their per-identity deltas,
not differences of sums across changing process inventories. Disk devices are
never summed: partitions are excluded, but stacked and physical devices may
still describe the same I/O. Missing/denied metrics are null, never zero.
An identity independently confirmed absent is recorded in exited_processes,
not assigned synthetic memory/I/O counters. Confirmation times bound the
sequential sampling uncertainty; they do not backdate the process's exit.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import posixpath
import signal
import stat as stat_mode
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

MEMORY_FIELDS = (
    "MemTotal",
    "MemFree",
    "MemAvailable",
    "Buffers",
    "Cached",
    "SwapCached",
    "AnonPages",
    "Shmem",
    "Slab",
    "SReclaimable",
    "SUnreclaim",
    "SwapTotal",
    "SwapFree",
    "Dirty",
    "Writeback",
)
SMAPS_FIELDS = {
    "rss_bytes": "Rss",
    "pss_bytes": "Pss",
    "anonymous_bytes": "Anonymous",
    "pss_anon_bytes": "Pss_Anon",
    "pss_file_bytes": "Pss_File",
    "pss_shmem_bytes": "Pss_Shmem",
    "swap_bytes": "Swap",
}
IO_FIELDS = ("rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes", "cancelled_write_bytes")
DISK_FIELDS = (
    "reads_completed",
    "reads_merged",
    "sectors_read",
    "read_ms",
    "writes_completed",
    "writes_merged",
    "sectors_written",
    "write_ms",
    "io_in_progress",
    "io_ms",
    "weighted_io_ms",
    "discards_completed",
    "discards_merged",
    "sectors_discarded",
    "discard_ms",
    "flushes_completed",
    "flush_ms",
)


def read_text_details(path: Path) -> tuple[str | None, str | None, int | None]:
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None, None
    except PermissionError as error:
        return None, "permission_denied", error.errno
    except FileNotFoundError as error:
        return None, "not_found", error.errno
    except OSError as error:
        return None, "read_error", error.errno


def read_text(path: Path) -> tuple[str | None, str | None]:
    value, error, _ = read_text_details(path)
    return value, error


def parse_kv(text: str) -> dict[str, int]:
    result = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        parts = value.split()
        try:
            result[key] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
        except (ValueError, IndexError):
            continue
    return result


def parse_proc_stat(text: str) -> dict:
    # comm can contain spaces and parentheses; the final ')' ends field 2.
    left, right = text.find("("), text.rfind(")")
    if left < 1 or right < left:
        raise ValueError("invalid proc stat")
    fields = text[right + 1 :].split()
    return {
        "pid": int(text[:left].strip()),
        "state": fields[0],
        "ppid": int(fields[1]),
        "user_ticks": int(fields[11]),
        "system_ticks": int(fields[12]),
        "num_threads": int(fields[17]),
        "starttime_ticks": int(fields[19]),
        "virtual_bytes": int(fields[20]),
        "rss_pages": int(fields[21]),
    }


def read_proc_stat_details(proc: Path, pid: int) -> tuple[dict | None, str | None, int | None]:
    text, error, error_number = read_text_details(proc / str(pid) / "stat")
    if text is None:
        return None, error, error_number
    try:
        value = parse_proc_stat(text)
        return value, None, None
    except (ValueError, IndexError):
        return None, "malformed", None


def read_proc_stat(proc: Path, pid: int) -> dict | None:
    value, _, _ = read_proc_stat_details(proc, pid)
    return value


def directory_identity(path: Path) -> dict:
    try:
        info = path.stat()
        return {
            "device": info.st_dev,
            "inode": info.st_ino,
            "error": None if stat_mode.S_ISDIR(info.st_mode) else "not_directory",
            "errno": None,
        }
    except OSError as error:
        category = (
            "not_found"
            if error.errno == errno.ENOENT
            else ("permission_denied" if error.errno in (errno.EACCES, errno.EPERM) else "read_error")
        )
        return {"device": None, "inode": None, "error": category, "errno": error.errno}


def directory_presence(path: Path) -> dict:
    try:
        path.lstat()
        return {"present": True, "error": None, "errno": None}
    except OSError as error:
        category = (
            "not_found"
            if error.errno == errno.ENOENT
            else ("permission_denied" if error.errno in (errno.EACCES, errno.EPERM) else "read_error")
        )
        return {"present": False if error.errno == errno.ENOENT else None, "error": category, "errno": error.errno}


def confirm_process_absence(
    proc: Path,
    identity: tuple[int, int],
    last_seen: dict | None,
    proc_identity: dict,
    trigger_error: str | None,
    trigger_errno: int | None,
) -> dict:
    """Require a previously observed identity and two independent ENOENT rounds."""
    pid, started = identity
    evidence = {
        "confirmed": False,
        "confirmation_started_monotonic_seconds": time.monotonic(),
        "last_observed_identity": {
            key: last_seen.get(key) for key in ("pid", "starttime_ticks", "observed_monotonic_seconds")
        }
        if last_seen
        else None,
        "trigger_error": trigger_error,
        "trigger_errno": trigger_errno,
        "proc_root_identity": proc_identity,
        "checks": [],
    }
    if (
        not last_seen
        or last_seen.get("pid") != pid
        or last_seen.get("starttime_ticks") != started
        or proc_identity.get("error") is not None
        or trigger_error != "not_found"
        or trigger_errno != errno.ENOENT
    ):
        return {**evidence, "reason": "missing_identity_or_no_definite_enoent"}
    for _ in range(2):
        root_before = directory_identity(proc)
        value, error, error_number = read_proc_stat_details(proc, pid)
        presence = directory_presence(proc / str(pid))
        root_after = directory_identity(proc)
        check = {
            "monotonic_seconds": time.monotonic(),
            "stat_error": error,
            "stat_errno": error_number,
            "observed_identity": {"pid": value["pid"], "starttime_ticks": value["starttime_ticks"]} if value else None,
            "pid_directory": presence,
            "proc_root_before": root_before,
            "proc_root_after": root_after,
        }
        evidence["checks"].append(check)
        if root_before != proc_identity or root_after != proc_identity:
            return {**evidence, "reason": "proc_root_changed_or_unavailable"}
        if value is not None:
            return {**evidence, "reason": "identity_reappeared_or_reused"}
        if (
            error != "not_found"
            or error_number != errno.ENOENT
            or presence["present"] is not False
            or presence["errno"] != errno.ENOENT
        ):
            return {**evidence, "reason": "absence_not_confirmed"}
    return {
        **evidence,
        "confirmed": True,
        "reason": None,
        "rule": "observed_identity_two_stat_and_directory_enoent_rounds_same_proc_root",
        "confirmed_monotonic_seconds": time.monotonic(),
    }


def scan_processes(proc: Path) -> dict[int, dict]:
    result = {}
    for entry in proc.iterdir():
        if entry.name.isdecimal():
            value = read_proc_stat(proc, int(entry.name))
            if value is not None:
                value["observed_monotonic_seconds"] = time.monotonic()
                result[value["pid"]] = value
    return result


def validate_registry(value: object) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("processes"), list):
        raise ValueError("registry requires a processes list")
    if value.get("run_id") is not None and not isinstance(value["run_id"], str):
        raise ValueError("run_id must be a string")
    result = {"run_id": value.get("run_id"), "processes": []}
    seen = set()
    for entry in value["processes"]:
        if not isinstance(entry, dict):
            raise ValueError("process entry must be an object")
        pid, started = entry.get("pid"), entry.get("starttime_ticks")
        if type(pid) is not int or pid <= 0 or type(started) is not int or started < 0:
            raise ValueError("positive pid and nonnegative starttime_ticks are required")
        if pid in seen:
            raise ValueError("duplicate pid")
        seen.add(pid)
        label = entry.get("label", "experiment")
        descendants = entry.get("include_descendants", True)
        if not isinstance(label, str) or not label or len(label) > 80 or type(descendants) is not bool:
            raise ValueError("invalid process label or descendant setting")
        result["processes"].append(
            {"pid": pid, "starttime_ticks": started, "label": label, "include_descendants": descendants}
        )
    return result


def path_is_within(candidate: str, root: str) -> bool:
    if not candidate.startswith("/"):
        return False
    # Avoid lexical '..' escaping the experiment root; no shell expansion.
    normalized = posixpath.normpath(candidate)
    return normalized == root or normalized.startswith(root + "/")


def discover_processes(proc: Path, inventory: dict[int, dict], root: str, names: set[str]) -> list[dict]:
    selected = []
    for pid, stat in inventory.items():
        try:
            executable = os.readlink(proc / str(pid) / "exe")
            if PurePosixPath(executable.removesuffix(" (deleted)")).name not in names:
                continue
            cwd = os.readlink(proc / str(pid) / "cwd")
            matched = path_is_within(cwd, root)
            if not matched:
                cmdline, _ = read_text(proc / str(pid) / "cmdline")
                matched = any(path_is_within(arg.split("=", 1)[-1], root) for arg in (cmdline or "").split("\0"))
            if matched:
                selected.append(
                    {
                        "pid": pid,
                        "starttime_ticks": stat["starttime_ticks"],
                        "label": "service",
                        "include_descendants": True,
                    }
                )
        except OSError:
            continue
    return selected


class Selector:
    def __init__(self):
        self.owned: dict[tuple[int, int], tuple[int, int]] = {}

    def select(self, inventory: dict[int, dict], roots: list[dict]) -> tuple[dict, list]:
        root_map = {(x["pid"], x["starttime_ticks"]): x for x in roots}
        current = {(pid, st["starttime_ticks"]): st for pid, st in inventory.items()}
        self.owned = {
            identity: root
            for identity, root in self.owned.items()
            if identity in current and root in root_map and root_map[root]["include_descendants"]
        }
        missing = []
        for root, entry in root_map.items():
            if root in current:
                self.owned[root] = root
            else:
                missing.append(
                    {
                        "pid": root[0],
                        "starttime_ticks": root[1],
                        "reason": "identity_mismatch" if root[0] in inventory else "not_visible",
                    }
                )
        changed = True
        while changed:
            changed = False
            owners_by_pid = {identity[0]: root for identity, root in self.owned.items()}
            for identity, stat in current.items():
                if identity in self.owned:
                    continue
                root = owners_by_pid.get(stat["ppid"])
                parent = inventory.get(stat["ppid"])
                if (
                    root is not None
                    and root_map[root]["include_descendants"]
                    and parent
                    and stat["starttime_ticks"] >= parent["starttime_ticks"]
                ):
                    self.owned[identity] = root
                    changed = True
        return (
            {
                identity: {"label": root_map[root]["label"], "root_pid": root[0], "root_starttime_ticks": root[1]}
                for identity, root in self.owned.items()
            },
            missing,
        )


def sample_process(
    proc: Path,
    identity: tuple[int, int],
    ownership: dict,
    clock_ticks: int,
    page_bytes: int,
    *,
    inventory_stat: dict | None = None,
    proc_identity: dict | None = None,
) -> dict:
    pid, started = identity
    proc_identity = proc_identity if proc_identity is not None else directory_identity(proc)
    stat, stat_error, stat_errno = read_proc_stat_details(proc, pid)
    result = {"pid": pid, "starttime_ticks": started, **ownership, "status": "ok", "errors": {}}
    if stat is None or stat["pid"] != pid or stat["starttime_ticks"] != started:
        evidence = confirm_process_absence(proc, identity, inventory_stat, proc_identity, stat_error, stat_errno)
        return {
            **result,
            "status": "confirmed_exited" if evidence["confirmed"] else "identity_unavailable",
            "errors": {"stat": stat_error or "identity_mismatch"},
            "read_errno": {"stat": stat_errno},
            "observed_identity": {"pid": stat["pid"], "starttime_ticks": stat["starttime_ticks"]} if stat else None,
            "exit_evidence": evidence,
        }
    stat["observed_monotonic_seconds"] = time.monotonic()
    smaps, smaps_error, smaps_errno = read_text_details(proc / str(pid) / "smaps_rollup")
    values = parse_kv(smaps or "")
    result["memory"] = {field: values.get(source) for field, source in SMAPS_FIELDS.items()}
    result["memory"]["stat_rss_bytes"] = stat["rss_pages"] * page_bytes
    result["memory"]["virtual_bytes"] = stat["virtual_bytes"]
    result["errors"]["smaps_rollup"] = smaps_error
    result["read_errno"] = {"smaps_rollup": smaps_errno}
    result["memory_observation"] = "smaps_rollup" if smaps_error is None else "unavailable"
    result["address_space_released"] = False
    result["memory_missing_fields"] = [field for field, source in SMAPS_FIELDS.items() if source not in values]
    maps = None
    # A task may outlive its user address space while exiting. A failed smaps
    # read alone proves nothing; independently check empty maps and the final
    # stat for this exact identity. Permission failures remain unavailable.
    if smaps_error is not None and smaps_error != "permission_denied":
        maps, maps_error, maps_errno = read_text_details(proc / str(pid) / "maps")
        result["errors"]["maps"] = maps_error
        result["read_errno"]["maps"] = maps_errno
        result["release_evidence"] = {
            "maps_read_succeeded": maps_error is None,
            "maps_characters": len(maps) if maps is not None else None,
            "initial_stat": {key: stat[key] for key in ("starttime_ticks", "virtual_bytes", "rss_pages", "state")},
        }
    io, io_error = read_text(proc / str(pid) / "io")
    io_values = parse_kv(io or "")
    result["io"] = {key: io_values.get(key) for key in IO_FIELDS}
    result["errors"]["io"] = io_error
    result["cpu"] = {
        "user_ticks": stat["user_ticks"],
        "system_ticks": stat["system_ticks"],
        "seconds": (stat["user_ticks"] + stat["system_ticks"]) / clock_ticks,
    }
    result["state"], result["num_threads"] = stat["state"], stat["num_threads"]
    # Reject a sample spanning exit/PID reuse, rather than joining two processes.
    end_stat, end_error, end_errno = read_proc_stat_details(proc, pid)
    if end_stat is None or end_stat["pid"] != pid or end_stat["starttime_ticks"] != started:
        evidence = confirm_process_absence(proc, identity, stat, proc_identity, end_error, end_errno)
        return {
            "pid": pid,
            "starttime_ticks": started,
            **ownership,
            "status": "confirmed_exited" if evidence["confirmed"] else "identity_changed_during_sample",
            "errors": {**result["errors"], "final_stat": end_error or "identity_mismatch"},
            "read_errno": {**result["read_errno"], "final_stat": end_errno},
            "address_space_released": False,
            "exit_evidence": evidence,
            "release_evidence": {**result.get("release_evidence", {}), "final_identity_matched": False},
        }
    if "release_evidence" in result:
        result["release_evidence"].update(
            {
                "final_identity_matched": True,
                "confirmation_stat": {
                    key: end_stat[key] for key in ("starttime_ticks", "virtual_bytes", "rss_pages", "state")
                },
            }
        )
        if (
            result["release_evidence"]["maps_read_succeeded"]
            and maps == ""
            and end_stat["virtual_bytes"] == 0
            and end_stat["rss_pages"] == 0
        ):
            result["memory"].update(dict.fromkeys(SMAPS_FIELDS, 0))
            result["memory"].update({"stat_rss_bytes": 0, "virtual_bytes": 0})
            result["memory_observation"] = "address_space_released"
            result["address_space_released"] = True
            result["smaps_missing_fields"] = result["memory_missing_fields"]
            result["memory_missing_fields"] = []
            result["release_evidence"]["rule"] = "same_identity_empty_maps_zero_vsize_zero_rss"
            # This concerns user mappings only, not retained cgroup/kernel/cache memory.
    return result


def sample_node(proc: Path, sys_root: Path, devices: set[str]) -> dict:
    memory_text, memory_error = read_text(proc / "meminfo")
    memory = parse_kv(memory_text or "")
    cpu_text, cpu_error = read_text(proc / "stat")
    cpu = None
    for line in (cpu_text or "").splitlines():
        if line.startswith("cpu "):
            try:
                cpu = dict(
                    zip(
                        ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal", "guest", "guest_nice"),
                        map(int, line.split()[1:]),
                        strict=False,
                    )
                )
            except ValueError:
                cpu_error = "malformed"
            break
    disk_text, disk_error = read_text(proc / "diskstats")
    disks = [] if disk_text is not None else None
    for line in (disk_text or "").splitlines():
        parts = line.split()
        if len(parts) < 14:
            continue
        major, minor, name = parts[:3]
        if devices and name not in devices:
            continue
        device_path = sys_root / "class" / "block" / name
        is_partition = (device_path / "partition").exists() if device_path.exists() else None
        if is_partition:
            continue
        try:
            counters = list(map(int, parts[3:]))
            slaves_path = device_path / "slaves"
            slaves = sorted(x.name for x in slaves_path.iterdir()) if slaves_path.exists() else None
            disks.append(
                {
                    "device": name,
                    "major_minor": f"{major}:{minor}",
                    "is_partition": is_partition,
                    "slaves": slaves,
                    "sector_bytes": 512,
                    "counters": dict(zip(DISK_FIELDS, counters, strict=False)),
                    "raw_counters": counters,
                }
            )
        except (ValueError, OSError):
            disk_error = "partial_read"
    pressure = {}
    for kind in ("memory", "cpu", "io"):
        value, error = read_text(proc / "pressure" / kind)
        pressure[kind] = {"value": value.strip() if value is not None else None, "error": error}
    return {
        "scope": "node_wide_including_other_workloads",
        "memory_bytes": {key: memory.get(key) for key in MEMORY_FIELDS},
        "cpu_ticks": cpu,
        "disk_devices": disks,
        "pressure": pressure,
        "disk_scope": "non_partition_devices_not_additive_across_storage_layers",
        "errors": {"meminfo": memory_error, "cpu": cpu_error, "diskstats": disk_error},
    }


def aggregate_memory(processes: list[dict], coverage_known: bool) -> dict:
    groups = {}
    for item in processes:
        groups.setdefault(item["label"], []).append(item)
    result = {}
    for label, group in groups.items():
        metrics = {}
        for field in SMAPS_FIELDS:
            known = [x.get("memory", {}).get(field) for x in group]
            metrics[field] = sum(known) if coverage_known and all(x is not None for x in known) else None
        result[label] = {
            "process_count": len(group),
            "memory": metrics,
            "rss_is_process_sum_not_unique_memory": True,
            "complete": coverage_known and all(v is not None for v in metrics.values()),
        }
    return result


class Sampler:
    def __init__(
        self,
        pid_file: Path | None,
        *,
        proc: Path = Path("/proc"),
        sys_root: Path = Path("/sys"),
        service_root: str | None = None,
        names: set[str] | None = None,
        devices: set[str] | None = None,
        clock_ticks: int | None = None,
        page_bytes: int | None = None,
    ):
        self.pid_file, self.proc, self.sys_root = pid_file, proc, sys_root
        self.service_root, self.names, self.devices = service_root, names or set(), devices or set()
        self.clock_ticks = clock_ticks if clock_ticks is not None else os.sysconf("SC_CLK_TCK")
        self.page_bytes = page_bytes if page_bytes is not None else os.sysconf("SC_PAGE_SIZE")
        self.selector = Selector()
        self.registry = {"run_id": None, "processes": []}
        self.previous_sample = None

    def sample(self) -> dict:
        started = time.monotonic()
        timestamp = datetime.now(timezone.utc).isoformat()
        registry_error = None
        if self.pid_file:
            text, registry_error = read_text(self.pid_file)
            if text is not None:
                try:
                    self.registry = validate_registry(json.loads(text))
                except (ValueError, TypeError):
                    registry_error = "invalid_registry"
        proc_identity = directory_identity(self.proc)
        inventory = scan_processes(self.proc)
        roots = list(self.registry["processes"])
        if self.service_root:
            registered = {x["pid"] for x in roots}
            roots += [
                x
                for x in discover_processes(self.proc, inventory, self.service_root, self.names)
                if x["pid"] not in registered
            ]
        selected, missing = self.selector.select(inventory, roots)
        for item in missing:
            if item["reason"] == "not_visible":
                item["reason"] = "stat_unreadable" if (self.proc / str(item["pid"])).exists() else "not_present"
        coverage_known = registry_error is None and not any(
            x["reason"] in {"stat_unreadable", "identity_mismatch"} for x in missing
        )
        observations = [
            sample_process(
                self.proc,
                identity,
                owner,
                self.clock_ticks,
                self.page_bytes,
                inventory_stat=inventory.get(identity[0]),
                proc_identity=proc_identity,
            )
            for identity, owner in sorted(selected.items())
        ]
        exited = [item for item in observations if item["status"] == "confirmed_exited"]
        processes = [item for item in observations if item["status"] != "confirmed_exited"]
        coverage_known = coverage_known and all(item["status"] == "ok" for item in processes)
        result = {
            "schema_version": 1,
            "utc": timestamp,
            "monotonic_seconds": started,
            "run_id": self.registry["run_id"],
            "sampler_pid": os.getpid(),
            "clock_ticks_per_second": self.clock_ticks,
            "page_bytes": self.page_bytes,
            "registry_error": registry_error,
            "registry_stale": registry_error is not None,
            "selection_coverage_known": coverage_known,
            "registered_roots": len(self.registry["processes"]),
            "missing_roots": missing,
            "processes": processes,
            "exited_processes": exited,
            "timestamp_semantics": "sample_start; exit absence is observed only at each confirmation time",
            "exit_confirmation_delay_seconds": max(
                (item["exit_evidence"]["confirmed_monotonic_seconds"] - started for item in exited), default=None
            ),
            "memory_by_label": aggregate_memory(processes, coverage_known),
            "node": sample_node(self.proc, self.sys_root, self.devices),
            "actual_interval_seconds": (started - self.previous_sample if self.previous_sample is not None else None),
        }
        self.previous_sample = started
        result["sample_duration_seconds"] = time.monotonic() - started
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--duration", type=float, help="Seconds; omitted means until SIGINT/SIGTERM")
    parser.add_argument("--service-root", help="Dedicated absolute Linux experiment directory")
    parser.add_argument("--allow-name", action="append", default=[], help="Exact executable basename for discovery")
    parser.add_argument("--device", action="append", default=[], help="Exact whole-device name; repeatable")
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        parser.error("Linux /proc and /sys are required")
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be positive and finite")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be positive and finite")
    if not args.pid_file and not args.service_root:
        parser.error("--pid-file or --service-root is required")
    if args.service_root:
        if not os.path.isabs(args.service_root):
            parser.error("--service-root must be absolute")
        args.service_root = os.path.realpath(args.service_root)
        if len(PurePosixPath(args.service_root).parts) < 3 or not args.allow_name:
            parser.error("--service-root needs a dedicated directory and --allow-name")
    sampler = Sampler(
        args.pid_file, service_root=args.service_root, names=set(args.allow_name), devices=set(args.device)
    )
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    deadline = started + args.duration if args.duration is not None else float("inf")
    # Exclusive creation protects completed runs from accidental overwrite.
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", buffering=1) as stream:
        next_sample = started
        while not stopping and time.monotonic() < deadline:
            row = sampler.sample()
            row["requested_interval_seconds"] = args.interval
            row["overrun"] = row["sample_duration_seconds"] > args.interval
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            next_sample += args.interval
            now = time.monotonic()
            if next_sample < now:
                next_sample = now
            time.sleep(max(0, min(next_sample, deadline) - now))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
