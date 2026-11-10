#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze a CAPE mimic evidence bundle.

The script is intentionally stdlib-only so it can run on the same cluster
storage path as the shell harness. It consumes the files produced by start.sh
and writes:

  - summary.json
  - summary.md
  - isolation_checks.json

Missing optional evidence is reported as "UNKNOWN"; required correctness
evidence fails closed.
"""
import argparse
import csv
import glob
import json
import math
import os
import sqlite3
import sys
from collections import defaultdict


TERMINAL_STATES = set(["COMPLETED", "FAILED", "CANCELLED", "EXPIRED", "LOST", "INFEASIBLE"])


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def as_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def percentile(values, pct):
    values = sorted([v for v in values if v is not None])
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * (pct / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def read_kv(path):
    data = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def user_from_output_path(bundle_dir, path):
    rel = os.path.relpath(path, bundle_dir)
    parts = rel.split(os.sep)
    if len(parts) >= 4 and parts[0] == "users" and parts[2] == "output":
        return parts[1]
    if len(parts) >= 3 and parts[0] == "outputs":
        return parts[1]
    return os.path.basename(os.path.dirname(path))


def scan_outputs(bundle_dir, submits):
    expected = defaultdict(set)
    for row in submits:
        uid = row.get("uid") or row.get("user") or row.get("owner")
        label = (row.get("request") or "").upper()
        if uid and label in ("A", "B"):
            expected[uid].add("gemm_silu_result.txt" if label == "A" else "cutedsl_add_result.txt")

    outputs = []
    for root, _dirs, files in os.walk(bundle_dir):
        for name in files:
            if name in ("gemm_silu_result.txt", "cutedsl_add_result.txt"):
                path = os.path.join(root, name)
                kv = read_kv(path)
                uid = user_from_output_path(bundle_dir, path)
                run_tag = kv.get("run_tag", "")
                owner_tag_ok = (not uid) or run_tag.startswith(uid + "@") or run_tag.startswith(uid + "-")
                status_ok = kv.get("status") == "PASS"
                checksum_ok = True
                if name == "gemm_silu_result.txt":
                    checksum = as_float(kv.get("checksum"))
                    checksum_ok = checksum is not None and math.isfinite(checksum)
                outputs.append({
                    "path": os.path.relpath(path, bundle_dir),
                    "uid": uid,
                    "file": name,
                    "status": kv.get("status"),
                    "run_tag": run_tag,
                    "owner_tag_ok": owner_tag_ok,
                    "checksum_ok": checksum_ok,
                    "cuda_visible_devices": kv.get("cuda_visible_devices", ""),
                    "pass": bool(status_ok and owner_tag_ok and checksum_ok),
                })

    seen = defaultdict(set)
    for item in outputs:
        seen[item["uid"]].add(item["file"])
    missing = []
    for uid, files in expected.items():
        for name in sorted(files - seen.get(uid, set())):
            missing.append({"uid": uid, "file": name})

    return {
        "outputs": outputs,
        "missing_outputs": missing,
        "workspace_tag_pass": all(item["owner_tag_ok"] for item in outputs) and not missing and bool(outputs),
        "correctness_pass": all(item["pass"] for item in outputs) and not missing and bool(outputs),
    }


def analyze_submits(rows, submit_threshold):
    latencies = [as_float(row.get("latency") or row.get("submit_latency")) for row in rows]
    latencies = [v for v in latencies if v is not None]
    starts = [as_float(row.get("t_start")) for row in rows]
    returns = [as_float(row.get("t_return")) for row in rows]
    starts = [v for v in starts if v is not None]
    returns = [v for v in returns if v is not None]
    p50 = percentile(latencies, 50)
    p99 = percentile(latencies, 99)
    return {
        "request_count": len(rows),
        "submit_latency_p50": p50,
        "submit_latency_p99": p99,
        "submit_latency_max": max(latencies) if latencies else None,
        "first_submit": min(starts) if starts else None,
        "last_submit_return": max(returns) if returns else None,
        "latency_threshold_s": submit_threshold,
        "latency_pass": p99 is not None and p99 < submit_threshold,
    }


def analyze_utilization(rows, grace_seconds, busy_threshold):
    samples = []
    for row in rows:
        t = as_float(row.get("t"))
        total = as_int(row.get("gpus_total"), 0)
        leased = as_int(row.get("gpus_leased"), 0)
        idle = as_int(row.get("idle"), max(0, total - leased))
        queued = as_int(row.get("queued"), 0)
        running = as_int(row.get("running"), 0)
        if t is None:
            continue
        samples.append({"t": t, "total": total, "leased": leased, "idle": idle, "queued": queued, "running": running})

    violations = []
    start = None
    last = None
    for sample in samples:
        bad = sample["idle"] > 0 and sample["queued"] > 0
        if bad and start is None:
            start = sample["t"]
        if bad:
            last = sample["t"]
        if not bad and start is not None:
            duration = max(0.0, (last or start) - start)
            if duration > grace_seconds:
                violations.append({"start": start, "end": last, "duration": duration})
            start = None
            last = None
    if start is not None:
        duration = max(0.0, (last or start) - start)
        if duration > grace_seconds:
            violations.append({"start": start, "end": last, "duration": duration})

    busy_values = []
    startup_excluded = 0
    tail_excluded = 0
    steady_open = False
    has_reached_steady_state = False
    capacity = None
    for sample in samples:
        total = sample["total"]
        if total <= 0:
            continue
        if total != capacity:
            capacity = total
            steady_open = False

        demand = sample["queued"] + sample["running"]
        if demand < total:
            if has_reached_steady_state:
                tail_excluded += 1
            else:
                startup_excluded += 1
            steady_open = False
            continue

        if not steady_open:
            if sample["leased"] < total:
                startup_excluded += 1
                continue
            steady_open = True
            has_reached_steady_state = True

        busy_values.append(float(sample["leased"]) / float(total))

    busy_fraction = sum(busy_values) / len(busy_values) if busy_values else None
    return {
        "samples": len(samples),
        "steady_state_samples": len(busy_values),
        "startup_excluded_samples": startup_excluded,
        "tail_excluded_samples": tail_excluded,
        "invariant_violations": len(violations),
        "violation_windows": violations,
        "busy_fraction": busy_fraction,
        "busy_threshold": busy_threshold,
        "work_conserving_pass": bool(samples) and len(violations) == 0,
        "busy_pass": busy_fraction is not None and busy_fraction >= busy_threshold,
    }


def read_journal(path):
    if not os.path.exists(path):
        return []
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute("SELECT timestamp, event_type, entity_id, state, reason, details FROM events ORDER BY id").fetchall()
    finally:
        conn.close()
    events = []
    for timestamp, event_type, entity_id, state, reason, details in rows:
        try:
            parsed = json.loads(details)
        except Exception:
            parsed = {}
        events.append({
            "timestamp": timestamp,
            "event_type": event_type,
            "entity_id": entity_id,
            "state": state,
            "reason": reason,
            "details": parsed,
        })
    return events


def analyze_journal(events):
    active_by_gpu = {}
    lease_violations = []
    session_gpu = {}
    request_gpu = {}
    request_session = {}
    terminal_session_states = set(["STOPPED", "FAILED", "LOST", "PURGED"])
    gpu_free_session_states = set(["CACHED_IDLE", "WARM_IDLE", "IDLE", "SUSPENDED", "PURGING"])
    terminal_request_states = set(["COMPLETED", "FAILED", "CANCELLED", "EXPIRED", "LOST", "INFEASIBLE"])

    for event in events:
        details = event.get("details", {})
        state = event.get("state")
        if event.get("event_type") == "session_state":
            sid = details.get("execution_session_id") or event.get("entity_id")
            gpus = tuple(details.get("gpu_uuids", ()))
            if gpus:
                session_gpu[sid] = gpus
            if state in terminal_session_states or state in gpu_free_session_states:
                for gpu, owner in list(active_by_gpu.items()):
                    if owner == sid:
                        del active_by_gpu[gpu]
                continue
            if state in ("CREATING", "STARTING", "RUNNING", "RUNNING_COMMAND", "RECONCILING"):
                for gpu in gpus:
                    owner = active_by_gpu.get(gpu)
                    if owner is not None and owner != sid:
                        lease_violations.append({
                            "timestamp": event.get("timestamp"),
                            "gpu_uuid": gpu,
                            "existing_session": owner,
                            "new_session": sid,
                        })
                    active_by_gpu[gpu] = sid
        elif event.get("event_type") in ("request_state", "request_queued", "request_submitted"):
            rid = details.get("request_id") or event.get("entity_id")
            if details.get("execution_session_id"):
                request_session[rid] = details.get("execution_session_id")
            if details.get("gpu_uuids"):
                request_gpu[rid] = tuple(details.get("gpu_uuids", ()))
            if state in terminal_request_states:
                pass
    return {
        "available": bool(events),
        "event_count": len(events),
        "lease_uniqueness_violations": lease_violations,
        "lease_uniqueness_pass": bool(events) and not lease_violations,
        "request_gpu_uuids": {k: list(v) for k, v in request_gpu.items()},
        "request_sessions": request_session,
    }


def analyze_status(rows):
    terminal = {}
    latest = {}
    for row in rows:
        jobid = row.get("jobid") or row.get("request_id")
        if not jobid:
            continue
        latest[jobid] = row
        if row.get("state") in TERMINAL_STATES:
            terminal[jobid] = row
    times = [as_float(row.get("t")) for row in terminal.values()]
    return {
        "available": bool(rows),
        "latest_count": len(latest),
        "terminal_count": len(terminal),
        "last_terminal": max([t for t in times if t is not None]) if times else None,
        "all_terminal": bool(rows) and len(latest) == len(terminal),
    }


def analyze_runtime_evidence(bundle_dir, rows, expected_adapter=None, expected_policy=None):
    latest = {}
    for row in rows:
        jobid = row.get("jobid") or row.get("request_id")
        if jobid:
            latest[jobid] = row

    checks = []
    for jobid in sorted(latest):
        row = latest[jobid]
        adapter = row.get("runtime_adapter") or ""
        policy = row.get("isolation_policy") or ""
        node_id = row.get("node_id") or ""
        command_id = row.get("command_id") or ""
        candidates = glob.glob(os.path.join(
            bundle_dir,
            "runtime-command-logs",
            "%s-*-%s.log" % (jobid, command_id),
        ))
        if node_id and command_id:
            candidates.append(os.path.join(
                bundle_dir, "pool", node_id, "agent-logs", command_id + ".log"
            ))
            candidates.append(os.path.join(
                bundle_dir, "pool", "agent-logs", node_id, command_id + ".log"
            ))
        log_path = next((path for path in candidates if os.path.isfile(path)), None)
        log_text = ""
        if log_path:
            with open(log_path, errors="replace") as fh:
                log_text = fh.read()
        adapter_ok = bool(adapter) and (
            expected_adapter is None or adapter == expected_adapter
        )
        policy_ok = bool(policy) and (
            expected_policy is None or policy == expected_policy
        )
        header_ok = bool(log_text) and all([
            "runtime-adapter: %s\n" % adapter in log_text,
            "isolation-policy: %s\n" % policy in log_text,
            "\nexec: " in log_text,
            "\nargv-json: " in log_text,
            "\nexit-code: 0\n" in log_text,
        ])
        checks.append({
            "jobid": jobid,
            "runtime_adapter": adapter,
            "isolation_policy": policy,
            "node_id": node_id,
            "command_id": command_id,
            "log_path": os.path.relpath(log_path, bundle_dir) if log_path else None,
            "adapter_pass": adapter_ok,
            "policy_pass": policy_ok,
            "exact_command_log_pass": header_ok,
            "pass": bool(adapter_ok and policy_ok and command_id and header_ok),
        })
    passed = bool(checks) and all(item["pass"] for item in checks)
    return {
        "enabled": bool(expected_adapter or expected_policy),
        "expected_runtime_adapter": expected_adapter,
        "expected_isolation_policy": expected_policy,
        "checks": checks,
        "pass": passed,
        "status": "PASS" if passed else ("FAIL" if checks else "UNKNOWN"),
    }


def read_auth_probes(bundle_dir):
    path = os.path.join(bundle_dir, "auth_probes.json")
    if not os.path.exists(path):
        return {"status": "UNKNOWN", "pass": False, "reason": "auth_probes.json is missing"}
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception as exc:
        return {"status": "FAIL", "pass": False, "reason": "failed to parse auth_probes.json: %s" % exc}
    cross = data.get("cross_user_status", {})
    node_submit = data.get("node_agent_submit", {})
    passed = bool(data.get("pass")) and bool(cross.get("denied")) and bool(node_submit.get("denied"))
    data["pass"] = passed
    data["status"] = "PASS" if passed else data.get("status", "FAIL")
    if data["status"] == "PASS" and not passed:
        data["status"] = "FAIL"
    return data


def compare_visibility(outputs, journal):
    request_gpus = journal.get("request_gpu_uuids", {})
    expected_values = set()
    for gpus in request_gpus.values():
        for gpu in gpus:
            expected_values.add(gpu)
    checks = []
    for item in outputs:
        visible = item.get("cuda_visible_devices") or ""
        if not visible:
            checks.append({"path": item["path"], "status": "UNKNOWN", "visible": visible, "reason": "missing CUDA_VISIBLE_DEVICES"})
        elif expected_values and visible not in expected_values:
            checks.append({"path": item["path"], "status": "FAIL", "visible": visible, "expected_any": sorted(expected_values)})
        else:
            checks.append({"path": item["path"], "status": "PASS" if expected_values else "UNKNOWN", "visible": visible})
    statuses = set(item["status"] for item in checks)
    return {
        "checks": checks,
        "pass": "FAIL" not in statuses and "UNKNOWN" not in statuses and bool(checks),
        "status": "FAIL" if "FAIL" in statuses else ("PASS" if checks and "UNKNOWN" not in statuses else "UNKNOWN"),
    }


def build_summary(bundle_dir, args):
    submits = read_csv(os.path.join(bundle_dir, "submits.csv"))
    utilization = read_csv(os.path.join(bundle_dir, "utilization_series.csv"))
    status_rows = read_csv(os.path.join(bundle_dir, "status_transitions.csv"))
    journal_path = os.path.join(bundle_dir, "journal.sqlite")
    journal_events = read_journal(journal_path)

    submit_summary = analyze_submits(submits, args.submit_threshold)
    util_summary = analyze_utilization(utilization, args.grace_seconds, args.busy_threshold)
    status_summary = analyze_status(status_rows)
    runtime_summary = analyze_runtime_evidence(
        bundle_dir,
        status_rows,
        expected_adapter=args.expected_runtime_adapter,
        expected_policy=args.expected_isolation_policy,
    )
    output_summary = scan_outputs(bundle_dir, submits)
    journal_summary = analyze_journal(journal_events)
    visibility = compare_visibility(output_summary["outputs"], journal_summary)
    auth_summary = read_auth_probes(bundle_dir)

    first_submit = submit_summary.get("first_submit")
    last_terminal = status_summary.get("last_terminal")
    makespan = (last_terminal - first_submit) if first_submit is not None and last_terminal is not None else None
    throughput = (float(submit_summary["request_count"]) / makespan) if makespan and makespan > 0 else None
    async_decoupled = False
    if makespan and submit_summary.get("submit_latency_max") is not None:
        async_decoupled = submit_summary["submit_latency_max"] < max(args.submit_threshold, makespan * 0.25)

    gates = {
        "precondition_correctness": output_summary["correctness_pass"],
        "r1_utilization": util_summary["work_conserving_pass"] and util_summary["busy_pass"],
        "r2_isolation": output_summary["workspace_tag_pass"] and journal_summary["lease_uniqueness_pass"] and visibility["pass"] and auth_summary["pass"],
        "r3_async": submit_summary["latency_pass"] and async_decoupled,
    }
    if runtime_summary["enabled"]:
        gates["runtime_evidence"] = runtime_summary["pass"]
    overall = all(gates.values())

    isolation = {
        "workspace": {
            "pass": output_summary["workspace_tag_pass"],
            "outputs": output_summary["outputs"],
            "missing_outputs": output_summary["missing_outputs"],
        },
        "lease_ownership": {
            "available": journal_summary["available"],
            "pass": journal_summary["lease_uniqueness_pass"],
            "violations": journal_summary["lease_uniqueness_violations"],
        },
        "visibility": visibility,
        "auth": auth_summary,
    }

    summary = {
        "bundle_dir": os.path.abspath(bundle_dir),
        "overall_pass": overall,
        "gates": gates,
        "submits": submit_summary,
        "utilization": util_summary,
        "status": status_summary,
        "runtime_evidence": runtime_summary,
        "correctness": {
            "pass": output_summary["correctness_pass"],
            "outputs_checked": len(output_summary["outputs"]),
            "missing_outputs": output_summary["missing_outputs"],
        },
        "isolation": isolation,
        "journal": {
            "path": journal_path if os.path.exists(journal_path) else None,
            "event_count": journal_summary["event_count"],
        },
        "makespan_s": makespan,
        "throughput_req_per_s": throughput,
        "async_decoupled": async_decoupled,
    }
    return summary, isolation


def write_json(path, data):
    with open(path, "w") as fh:
        json.dump(data, fh, sort_keys=True, indent=2)
        fh.write("\n")


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "%.6g" % value
    return str(value)


def write_markdown(path, summary):
    lines = []
    lines.append("# CAPE Mimic Summary")
    lines.append("")
    lines.append("- overall: `%s`" % ("PASS" if summary["overall_pass"] else "FAIL"))
    for key in sorted(summary["gates"]):
        lines.append("- %s: `%s`" % (key, "PASS" if summary["gates"][key] else "FAIL"))
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("- requests: %s" % summary["submits"]["request_count"])
    lines.append("- submit latency p50/p99/max: %s / %s / %s s" % (
        fmt(summary["submits"]["submit_latency_p50"]),
        fmt(summary["submits"]["submit_latency_p99"]),
        fmt(summary["submits"]["submit_latency_max"]),
    ))
    lines.append("- makespan: %s s" % fmt(summary["makespan_s"]))
    lines.append("- throughput: %s req/s" % fmt(summary["throughput_req_per_s"]))
    lines.append("- utilization samples: %s" % summary["utilization"]["samples"])
    lines.append("- steady-state utilization samples: %s" % summary["utilization"]["steady_state_samples"])
    lines.append("- excluded startup/tail samples: %s / %s" % (
        summary["utilization"]["startup_excluded_samples"],
        summary["utilization"]["tail_excluded_samples"],
    ))
    lines.append("- invariant violations: %s" % summary["utilization"]["invariant_violations"])
    lines.append("- steady-state busy fraction: %s" % fmt(summary["utilization"]["busy_fraction"]))
    lines.append("- outputs checked: %s" % summary["correctness"]["outputs_checked"])
    lines.append("- exact runtime logs checked: %s" % len(summary["runtime_evidence"]["checks"]))
    lines.append("- journal events: %s" % summary["journal"]["event_count"])
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    if summary["isolation"]["auth"].get("status") == "UNKNOWN":
        lines.append("- Auth cross-user probes were not present in the bundle.")
    elif not summary["isolation"]["auth"].get("pass"):
        lines.append("- Auth cross-user or node-agent-submit denial probes failed.")
    if not summary["journal"]["path"]:
        lines.append("- `journal.sqlite` was not present, so lease uniqueness is not proven.")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Analyze a CAPE mimic evidence bundle")
    parser.add_argument("bundle_dir", help="mimic evidence directory")
    parser.add_argument("--grace-seconds", type=float, default=2.0)
    parser.add_argument("--busy-threshold", type=float, default=0.9)
    parser.add_argument("--submit-threshold", type=float, default=2.0)
    parser.add_argument("--expected-runtime-adapter")
    parser.add_argument("--expected-isolation-policy")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    bundle_dir = args.bundle_dir
    if not os.path.isdir(bundle_dir):
        sys.stderr.write("bundle directory does not exist: %s\n" % bundle_dir)
        return 2
    summary, isolation = build_summary(bundle_dir, args)
    write_json(os.path.join(bundle_dir, "summary.json"), summary)
    write_json(os.path.join(bundle_dir, "isolation_checks.json"), isolation)
    write_markdown(os.path.join(bundle_dir, "summary.md"), summary)
    sys.stdout.write("overall=%s summary=%s\n" % (
        "PASS" if summary["overall_pass"] else "FAIL",
        os.path.join(bundle_dir, "summary.json"),
    ))
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
