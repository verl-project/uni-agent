#!/usr/bin/env python3
"""Analyze raw single-node offload trials without inventing missing coverage.

Input: manifest.json, results.jsonl and TRIAL/{result,cleanup}.json,
TRIAL/{events,metrics}.jsonl produced by demo.py and host_sampler.py.
Output: summary.json and a Chinese report.md; source files are never changed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

GIB = 1024**3
MODES = ("resident", "offload")


def number(value):
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values, fraction):
    values = sorted(x for x in values if number(x))
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def statistics(values):
    clean = [x for x in values if number(x)]
    return {
        "n": len(clean),
        "p50": percentile(clean, 0.5),
        "p95": percentile(clean, 0.95),
        "max": max(clean) if clean else None,
    }


def curve_summary(points, start, end, max_gap=1.0):
    """Trapezoids with interpolated endpoints, no extrapolation or missing-value bridging."""
    result = {
        "window_seconds": end - start,
        "integral_byte_seconds": None,
        "integral_gib_seconds": None,
        "peak_bytes": None,
        "minimum_bytes": None,
        "status": "unverified",
        "reason": None,
    }
    points = sorted(points, key=lambda point: point[0])
    if not number(start) or not number(end) or end <= start or len(points) < 2:
        return {**result, "reason": "insufficient_samples_or_invalid_window"}
    if points[0][0] > start or points[-1][0] < end:
        return {**result, "reason": "window_not_covered", "observed_start": points[0][0], "observed_end": points[-1][0]}
    area, extrema = 0.0, []
    for (left, a), (right, b) in zip(points, points[1:], strict=False):
        if right <= start or left >= end:
            continue
        if right <= left:
            return {**result, "reason": "non_increasing_time"}
        if not number(a) or not number(b):
            return {**result, "reason": "missing_metric_inside_window"}
        if right - left > max_gap:
            return {**result, "reason": "sample_gap_exceeded", "gap_seconds": right - left}
        lo, hi = max(start, left), min(end, right)
        a_lo = a + (b - a) * (lo - left) / (right - left)
        b_hi = a + (b - a) * (hi - left) / (right - left)
        area += (a_lo + b_hi) * (hi - lo) / 2
        extrema.extend((a_lo, b_hi))
    if not extrema:
        return {**result, "reason": "no_covered_segments"}
    return {
        **result,
        "integral_byte_seconds": area,
        "integral_gib_seconds": area / GIB,
        "peak_bytes": max(extrema),
        "minimum_bytes": min(extrema),
        "status": "observed",
        "reason": None,
    }


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError):
        return {}, "unreadable_or_invalid_json"


def read_jsonl(path):
    rows, errors = [], []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], ["file_unavailable"]
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row must be an object")
            rows.append(value)
        except ValueError:
            errors.append(f"invalid_line_{index + 1}")
    return rows, errors


def event_named(events, name):
    return next((x for x in events if x.get("event") == name), None)


def process_pss(row, labels=None):
    if row.get("registry_error") or row.get("selection_coverage_known") is not True:
        return None
    if not isinstance(row.get("processes"), list):
        return None
    total = 0
    for process in row["processes"]:
        if labels is not None and process.get("label") not in labels:
            continue
        value = process.get("memory", {}).get("pss_bytes")
        if process.get("status") != "ok" or not number(value):
            return None
        total += value
    return total


def align_metrics(result, events, raw):
    origin = result.get("measurement_started_monotonic_seconds")
    event = event_named(events, "measurement_started") or event_named(events, "trial_started")
    if not number(origin) and event:
        origin = event.get("monotonic_seconds")
    calibration = result.get("clock_calibration") or {}
    offset = calibration.get("client_minus_node_seconds")
    uncertainty = calibration.get("uncertainty_seconds")
    calibrated = number(origin) and number(offset) and number(uncertainty) and uncertainty >= 0
    aligned = []
    for row in raw:
        if calibrated and number(row.get("monotonic_seconds")):
            elapsed = row["monotonic_seconds"] + offset - origin
        elif event and number(event.get("wall_time_ns")) and isinstance(row.get("utc"), str):
            try:
                elapsed = datetime.fromisoformat(row["utc"].replace("Z", "+00:00")).timestamp()
                elapsed -= event["wall_time_ns"] / 1e9
            except ValueError:
                continue
        else:
            continue
        aligned.append({"time": elapsed, "row": row})
    return sorted(aligned, key=lambda x: x["time"]), {
        "status": "calibrated" if calibrated else "unverified_wall_clock_fallback",
        "uncertainty_seconds": uncertainty if calibrated else None,
        "origin_client_monotonic_seconds": origin,
    }


def counter_deltas(samples, start, end):
    """Per-identity observed increments; never invent initial/final short-lived I/O."""
    identities, disks = {}, {}
    chosen = [sample for sample in samples if start <= sample["time"] <= end]
    for sample in chosen:
        for process in sample["row"].get("processes", []):
            if process.get("status") != "ok":
                continue
            key = (process.get("pid"), process.get("starttime_ticks"))
            entry = identities.setdefault(
                key, {"pid": key[0], "starttime_ticks": key[1], "label": process.get("label"), "points": []}
            )
            entry["points"].append((sample["time"], process.get("io", {})))
        for device in sample["row"].get("node", {}).get("disk_devices") or []:
            key = (device.get("device"), device.get("major_minor"))
            entry = disks.setdefault(
                key, {"device": key[0], "major_minor": key[1], "slaves": device.get("slaves"), "points": []}
            )
            counters = device.get("counters", {})
            sector_bytes = device.get("sector_bytes", 512)
            read, write = counters.get("sectors_read"), counters.get("sectors_written")
            entry["points"].append(
                (
                    sample["time"],
                    {
                        "read_bytes": read * sector_bytes if number(read) else None,
                        "write_bytes": write * sector_bytes if number(write) else None,
                    },
                )
            )
    process_rows, device_rows = [], []
    for collection, target in ((identities, process_rows), (disks, device_rows)):
        for entry in collection.values():
            points = entry.pop("points")
            entry["first_observed_seconds"], entry["last_observed_seconds"] = points[0][0], points[-1][0]
            entry["samples"] = len(points)
            for metric in ("read_bytes", "write_bytes"):
                values = [value.get(metric) for _, value in points]
                valid = len(values) >= 2 and all(number(x) for x in values)
                reset = valid and any(b < a for a, b in zip(values, values[1:], strict=False))
                entry[f"observed_{metric}"] = values[-1] - values[0] if valid and not reset else None
                entry[f"first_counter_{metric}"] = values[0] if number(values[0]) else None
                entry[f"{metric}_status"] = (
                    "counter_reset" if reset else "observed_lower_bound" if valid else "insufficient_data"
                )
            entry["initial_and_terminal_unobserved_io_not_included"] = True
            target.append(entry)
    totals = {}
    for metric in ("read_bytes", "write_bytes"):
        values = [row[f"observed_{metric}"] for row in process_rows]
        measured = [x for x in values if number(x)]
        totals[f"observed_{metric}_lower_bound"] = sum(measured) if measured else None
        totals[f"{metric}_incomplete_identities"] = sum(x is None for x in values)
    return {
        "process_identities": process_rows,
        "process_totals": totals,
        "disk_devices": device_rows,
        "scope": "sampled_interval_only; process_initial_and_terminal_gaps; node_devices_not_additive",
    }


def measurement_summary(trial, horizon, max_gap):
    samples = trial["aligned"]
    aligned_info = trial["clock"]
    uncertainty = aligned_info.get("uncertainty_seconds")
    timing_valid = aligned_info["status"] == "calibrated" and bool(samples)
    if timing_valid:
        timing_valid = samples[0]["time"] <= -uncertainty and samples[-1]["time"] >= horizon + uncertainty
    common = {
        "start_seconds": 0.0,
        "end_seconds": horizon,
        "clock": aligned_info,
        "time_coverage_with_uncertainty": timing_valid,
        "metric_file_errors": trial["metric_errors"],
        "event_file_errors": trial["event_errors"],
    }
    curves = {}
    for name, getter in {
        "assigned_process_pss": process_pss,
        "sandbox_process_pss": lambda row: process_pss(row, {"sandbox"}),
        "node_anon": lambda row: row.get("node", {}).get("memory_bytes", {}).get("AnonPages"),
        "node_cache": lambda row: row.get("node", {}).get("memory_bytes", {}).get("Cached"),
        "node_available": lambda row: row.get("node", {}).get("memory_bytes", {}).get("MemAvailable"),
    }.items():
        curves[name] = curve_summary([(x["time"], getter(x["row"])) for x in samples], 0, horizon, max_gap)
    if not any(
        process.get("label") == "sandbox" for sample in samples for process in sample["row"].get("processes", [])
    ):
        curves["sandbox_process_pss"] = {
            "status": "unverified",
            "reason": "sandbox_never_observed",
            "integral_gib_seconds": None,
            "integral_byte_seconds": None,
            "peak_bytes": None,
            "minimum_bytes": None,
        }
    verified = timing_valid and not trial["metric_errors"] and not trial["event_errors"]
    return {
        **common,
        "curves": curves,
        "comparable": verified
        and all(curves[name]["status"] == "observed" for name in ("assigned_process_pss", "sandbox_process_pss")),
        "node_observations_complete": verified
        and all(curves[name]["status"] == "observed" for name in ("node_anon", "node_cache", "node_available")),
        "io": counter_deltas(samples, 0, horizon),
    }


def load_trials(root):
    manifest, manifest_error = read_json(root / "manifest.json")
    rows, input_errors = read_jsonl(root / "results.jsonl")
    if manifest_error:
        input_errors.append("manifest_" + manifest_error)
    by_id = {}
    for row in rows:
        trial_id = row.get("trial_id")
        if (
            isinstance(trial_id, str)
            and Path(trial_id).name == trial_id
            and trial_id not in {".", ".."}
            and "/" not in trial_id
            and "\\" not in trial_id
        ):
            if trial_id in by_id:
                input_errors.append("duplicate_result_" + trial_id)
            by_id[trial_id] = row
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / "events.jsonl").is_file():
            continue
        if directory.name in by_id:
            continue
        result, error = read_json(directory / "result.json")
        events, _ = read_jsonl(directory / "events.jsonl")
        beginning = event_named(events, "trial_started")
        if error and beginning:
            result = {
                "trial_id": directory.name,
                "mode": beginning.get("mode"),
                "success": False,
                "incomplete": True,
                "cleanup": {"confirmed": False},
            }
        if result:
            by_id[directory.name] = result
    trials = []
    tail = manifest.get("parameters", {}).get("io_tail_seconds", 2.0)
    tail = tail if number(tail) and tail >= 0 else 2.0
    for trial_id, result in sorted(by_id.items()):
        events, event_errors = read_jsonl(root / trial_id / "events.jsonl")
        metrics, metric_errors = read_jsonl(root / trial_id / "metrics.jsonl")
        aligned, clock = align_metrics(result, events, metrics)
        cleanup = event_named(events, "cleanup_confirmed")
        origin = clock.get("origin_client_monotonic_seconds")
        required = (
            cleanup["monotonic_seconds"] - origin + tail
            if cleanup and number(cleanup.get("monotonic_seconds")) and number(origin)
            else None
        )
        finished = result.get("measurement_finished_monotonic_seconds")
        diagnostic_horizon = (
            finished - origin if number(finished) and number(origin) else aligned[-1]["time"] if aligned else None
        )
        pair = result.get("pair_index")
        if type(pair) is not int:
            match = re.match(r"pair-(\d+)-", trial_id)
            pair = int(match.group(1)) if match else None
        trials.append(
            {
                "result": result,
                "pair_index": pair,
                "events": events,
                "event_errors": event_errors,
                "metric_errors": metric_errors,
                "aligned": aligned,
                "clock": clock,
                "required_horizon_seconds": required,
                "diagnostic_horizon_seconds": diagnostic_horizon,
            }
        )
    return manifest, trials, input_errors


def plan_completeness(manifest, trials):
    parameters = manifest.get("parameters", {})
    count, mode = parameters.get("pairs"), parameters.get("mode")
    if type(count) is not int or count <= 0 or mode not in (*MODES, "both"):
        return {"status": "unverified", "reason": "plan_missing_or_invalid", "planned_tasks": None}
    modes = MODES if mode == "both" else (mode,)
    expected = {(index, selected_mode) for index in range(count) for selected_mode in modes}
    observed = {}
    for trial in trials:
        selected_mode = trial["result"].get("mode")
        key = (trial["pair_index"], selected_mode if isinstance(selected_mode, str) else None)
        observed[key] = observed.get(key, 0) + 1
    missing = sorted(expected - observed.keys())
    invalid = [key for key, occurrences in observed.items() if key not in expected or occurrences != 1]
    incomplete = [
        trial["result"]["trial_id"]
        for trial in trials
        if trial["result"].get("incomplete") is True or not number(trial["result"].get("task_seconds"))
    ]
    status = "unverified" if invalid else "unfinished" if missing or incomplete else "pass"
    return {
        "status": status,
        "planned_tasks": len(expected),
        "reason": "unexpected_or_duplicate_trial"
        if invalid
        else "planned_trials_missing"
        if missing
        else "started_trials_incomplete"
        if incomplete
        else None,
        "incomplete_trial_ids": incomplete,
        "missing_trials": [{"pair_index": index, "mode": selected_mode} for index, selected_mode in missing],
        "unexpected_or_duplicate_trials": [
            {"pair_index": index, "mode": selected_mode} for index, selected_mode in invalid
        ],
    }


def analyze_directory(
    root, *, memory_saving_min=0.2, task_p95_increase_max=0.05, first_tool_extra_max=1.0, max_gap=1.0
):
    manifest, trials, input_errors = load_trials(root)
    modes = {}
    for mode in MODES:
        group = [x["result"] for x in trials if x["result"].get("mode") == mode]
        success = [x for x in group if x.get("success") is True]
        modes[mode] = {
            "started_tasks": len(group),
            "successful_tasks": len(success),
            "failed_or_incomplete_tasks": len(group) - len(success),
            "success_rate": len(success) / len(group) if group else None,
            "cleanup_confirmed_tasks": sum(x.get("cleanup", {}).get("confirmed") is True for x in group),
            "cleanup_rate": (
                sum(x.get("cleanup", {}).get("confirmed") is True for x in group) / len(group) if group else None
            ),
            "task_seconds_all_terminal_observations": statistics([x.get("task_seconds") for x in group]),
            "task_seconds_successful": statistics([x.get("task_seconds") for x in success]),
            "first_tool_seconds_task_p95": statistics(
                [percentile([c.get("first_tool_seconds") for c in x.get("cycles", [])], 0.95) for x in success]
            ),
            "restore_oracle_seconds_task_p95": statistics(
                [percentile([c.get("restore_oracle_seconds") for c in x.get("cycles", [])], 0.95) for x in success]
            ),
        }
    pair_indices = sorted({x["pair_index"] for x in trials if x["pair_index"] is not None})
    pairs, measurements = [], {}
    task_pairs, tool_extras, savings, resident_integrals = [], [], [], []
    for pair_index in pair_indices:
        group = [x for x in trials if x["pair_index"] == pair_index]
        groups = {mode: [x for x in group if x["result"].get("mode") == mode] for mode in MODES}
        pair = {"pair_index": pair_index, "status": "unverified", "reason": None}
        if any(len(groups[mode]) != 1 for mode in MODES):
            pairs.append({**pair, "reason": "missing_or_duplicate_mode"})
            continue
        resident, offload = groups["resident"][0], groups["offload"][0]
        pair["trial_ids"] = {mode: groups[mode][0]["result"]["trial_id"] for mode in MODES}
        needed = [x["required_horizon_seconds"] for x in (resident, offload)]
        horizon = max(needed) if all(number(x) and x > 0 for x in needed) else None
        pair["common_window_seconds"] = horizon
        if horizon is not None:
            for trial in (resident, offload):
                measurements[trial["result"]["trial_id"]] = measurement_summary(trial, horizon, max_gap)
        else:
            pair["reason"] = "cleanup_or_measurement_origin_missing"
        correct = all(
            x["result"].get("success") is True and x["result"].get("cleanup", {}).get("confirmed") is True
            for x in (resident, offload)
        )
        pair["correct_and_clean"] = correct
        if correct:
            a, b = resident["result"], offload["result"]
            if number(a.get("task_seconds")) and number(b.get("task_seconds")) and a["task_seconds"] > 0:
                pair["task_latency_change"] = b["task_seconds"] / a["task_seconds"] - 1
                task_pairs.append((a["task_seconds"], b["task_seconds"]))
            a_cycles = {x.get("cycle"): x for x in a.get("cycles", [])}
            b_cycles = {x.get("cycle"): x for x in b.get("cycles", [])}
            cycle_extras = [
                b_cycles[key]["first_tool_seconds"] - a_cycles[key]["first_tool_seconds"]
                for key in a_cycles.keys() & b_cycles.keys()
                if number(a_cycles[key].get("first_tool_seconds")) and number(b_cycles[key].get("first_tool_seconds"))
            ]
            extra = percentile(cycle_extras, 0.95)
            pair["first_tool_extra_seconds_task_p95"] = extra
            pair["matched_cycles"] = len(cycle_extras)
            if extra is not None and len(cycle_extras) == len(a_cycles) == len(b_cycles):
                tool_extras.append(extra)
            if horizon is not None:
                a_metric, b_metric = (measurements[x["trial_id"]] for x in (a, b))
                if a_metric["comparable"] and b_metric["comparable"]:
                    a_pss = a_metric["curves"]["assigned_process_pss"]["integral_gib_seconds"]
                    b_pss = b_metric["curves"]["assigned_process_pss"]["integral_gib_seconds"]
                    pair["assigned_process_pss_saved_gib_seconds"] = a_pss - b_pss
                    pair["assigned_process_pss_saving_fraction"] = 1 - b_pss / a_pss if a_pss > 0 else None
                    if a_pss > 0:
                        savings.append(a_pss - b_pss)
                        resident_integrals.append(a_pss)
                        pair["status"], pair["reason"] = "observed", None
                else:
                    pair["reason"] = "common_window_or_metric_coverage_insufficient"
        else:
            pair["reason"] = "task_or_cleanup_failed"
        pairs.append(pair)
    # Preserve diagnostics for failed/unpaired trials, without promoting their shorter window to a fair A/B result.
    for trial in trials:
        trial_id = trial["result"]["trial_id"]
        horizon = trial["required_horizon_seconds"] or trial["diagnostic_horizon_seconds"]
        if trial_id not in measurements and number(horizon) and horizon > 0:
            measurement = measurement_summary(trial, horizon, max_gap)
            measurement["unpaired_diagnostic_only"] = True
            measurements[trial_id] = measurement
    relative_saving = sum(savings) / sum(resident_integrals) if resident_integrals else None
    a_p95 = percentile([a for a, _ in task_pairs], 0.95)
    b_p95 = percentile([b for _, b in task_pairs], 0.95)
    latency_change = b_p95 / a_p95 - 1 if number(a_p95) and a_p95 > 0 else None
    tool_extra = percentile(tool_extras, 0.95)
    all_correct = bool(trials) and all(
        x["result"].get("success") is True and x["result"].get("cleanup", {}).get("confirmed") is True for x in trials
    )
    plan = plan_completeness(manifest, trials)
    all_paired = bool(pairs) and len(trials) == 2 * len(pairs) and not input_errors and plan["status"] == "pass"

    def gate(value, threshold, direction, complete):
        if value is None or not complete:
            return "unverified"
        return "pass" if (value >= threshold if direction == "min" else value <= threshold) else "fail"

    gates = {
        "planned_run_complete": plan["status"],
        "task_correctness_and_cleanup": "unverified"
        if input_errors and all_correct
        else "pass"
        if all_correct
        else "fail",
        "assigned_process_pss_saving": gate(
            relative_saving, memory_saving_min, "min", all_paired and len(savings) == len(pairs)
        ),
        "task_p95_increase": gate(
            latency_change, task_p95_increase_max, "max", all_paired and len(task_pairs) == len(pairs)
        ),
        "first_tool_extra_task_p95": gate(
            tool_extra, first_tool_extra_max, "max", all_paired and len(tool_extras) == len(pairs)
        ),
        "physical_node_saving_attribution": "unverified",
    }
    planned = plan["planned_tasks"]
    if plan["status"] != "pass":
        acceptance = "unfinished" if plan["status"] == "unfinished" else "unverified"
    else:
        acceptance = (
            "fail"
            if "fail" in gates.values()
            else ("pass" if all(value == "pass" for value in gates.values()) else "unverified")
        )
    cost_records = []
    for trial in trials:
        value = (
            measurements.get(trial["result"]["trial_id"], {})
            .get("curves", {})
            .get("assigned_process_pss", {})
            .get("integral_gib_seconds")
        )
        if number(value):
            cost_records.append({"value": value, "success": trial["result"].get("success") is True})
    failed_costs = [x["value"] for x in cost_records if not x["success"]]
    any_failed = any(x["result"].get("success") is not True for x in trials)
    return {
        "schema_version": 1,
        "agentenv_sha": manifest.get("agentenv_sha"),
        "required_agentenv_sha": manifest.get("required_agentenv_sha", manifest.get("agentenv_sha")),
        "workload": manifest.get("workload", "unknown"),
        "parameters": manifest.get("parameters", {}),
        "source_sha256": manifest.get("source_sha256", {}),
        "started_tasks": len(trials),
        "planned_tasks": planned,
        "unknown_mode_tasks": sum(x["result"].get("mode") not in MODES for x in trials),
        "unstarted_tasks": max(0, planned - len(trials)) if planned is not None else None,
        "input_errors": input_errors,
        "by_mode": modes,
        "pairs": pairs,
        "planned_run": plan,
        "acceptance_status": acceptance,
        "observed_pss_costs": {
            "measured_tasks": len(cost_records),
            "unmeasured_tasks": len(trials) - len(cost_records),
            "all_started_measured_gib_seconds": sum(x["value"] for x in cost_records) if cost_records else None,
            "failed_tasks_measured_gib_seconds": sum(failed_costs) if failed_costs else None if any_failed else 0,
            "scope": "observed_task_windows_including_failures; not_a_fair_pairwise_saving_estimate",
        },
        "trial_measurements": measurements,
        "trial_outcomes": [
            {
                "trial_id": x["result"]["trial_id"],
                "mode": x["result"].get("mode"),
                "success": x["result"].get("success") is True,
                "cleanup_confirmed": x["result"].get("cleanup", {}).get("confirmed") is True,
                "incomplete": x["result"].get("incomplete", False),
                "error_type": x["result"].get("error_type"),
                "event_file_errors": x["event_errors"],
            }
            for x in trials
        ],
        "paired_summary": {
            "independent_pairs": len(pairs),
            "memory_comparable_pairs": len(savings),
            "assigned_process_pss_saving_fraction": relative_saving,
            "task_p95_relative_increase": latency_change,
            "first_tool_extra_seconds_task_p95": tool_extra,
        },
        "thresholds": {
            "memory_saving_min": memory_saving_min,
            "task_p95_increase_max": task_p95_increase_max,
            "first_tool_extra_max_seconds": first_tool_extra_max,
            "max_sample_gap_seconds": max_gap,
        },
        "gates": gates,
        "inference_scope": "assigned_process_PSS_only; whole_node_counters_not_causal_attribution",
        "sandbox_process_pss_scope": "observed_lower_bound_of_explicit_sandbox_label_windows; "
        "pre_registration_VM_is_in_provider_descendants_and_assigned_total; "
        "not_complete_VM_integral_or_billing; acceptance_uses_assigned_total",
        "physical_node_benefit": "unverified",
        "statistical_unit": "independent_task_pair; cycles_are_correlated_within_task",
        "percentile_method": "linear_interpolation_at_(n-1)*q; exploratory_small_sample",
    }


def show(value, digits=3):
    return f"{value:.{digits}f}" if number(value) else "不可验证"


def report_markdown(summary):
    lines = [
        "# AgentEnv 常驻与卸载恢复实验分析",
        "",
        f"目标固定源码：`{summary.get('required_agentenv_sha') or '未记录'}`。",
        f"运行版本记录：`{summary.get('agentenv_sha') or '需结合部署证据核对，目标版本不代表实际版本'}`。",
        "负载为外部模拟推理等待与 HTTP 状态探针文件/计数操作；不代表真实模型推理或完整 agent 工作负载性能。",
        f"已开始 {summary['started_tasks']} 个任务；计划 {summary['planned_tasks']} 个。失败和中断保留在分母中。",
        "",
        "## 任务结果与延迟",
        "",
        "分位数以独立任务为统计单位。首工具指标先取各任务内周期的 p95，再对任务取分位数；周期不当作独立样本。",
        "清理率依据 result.cleanup.confirmed；若其只确认 API/PID 消失，仍不代表持久化产物和设备已完成审计。",
        "延迟表仅列成功任务，同时明确所有失败数量；小样本分位数仅供探索，不能证明生产成功率或尾延迟 SLA。",
        "",
        "| 模式 | 成功/已开始 | 清理确认/已开始 | 任务 p50 秒 | 任务 p95 秒 | 首工具任务 p95 秒 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, data in summary["by_mode"].items():
        lines.append(
            f"| {mode} | {data['successful_tasks']}/{data['started_tasks']} | "
            f"{data['cleanup_confirmed_tasks']}/{data['started_tasks']} | "
            f"{show(data['task_seconds_successful']['p50'])} | "
            f"{show(data['task_seconds_successful']['p95'])} | "
            f"{show(data['first_tool_seconds_task_p95']['p95'])} |"
        )
    lines += [
        "",
        "## 配对内存窗口",
        "",
        "每对从各自 measurement_started 对齐，使用较慢任务完成清理后的 I/O 收尾作为共同结束点。",
        "只在两侧完整覆盖该窗口、时钟校准和指标齐全时计算收益；不把缺测补零，不外推较短一侧。",
        "积分使用梯形法及边界线性插值；缺测或超出最大采样间隔使该曲线不可验证。",
        "",
        "| 配对 | 共同窗口秒 | 归属进程 PSS 节省 GiB·s | 比例 | 状态/原因 |",
        "|---|---:|---:|---:|---|",
    ]
    for pair in summary["pairs"]:
        saving = pair.get("assigned_process_pss_saving_fraction")
        lines.append(
            f"| {pair['pair_index']} | {show(pair.get('common_window_seconds'))} | "
            f"{show(pair.get('assigned_process_pss_saved_gib_seconds'))} | "
            f"{show(saving * 100 if number(saving) else None)}% | "
            f"{pair['status']}: {pair.get('reason') or '完整'} |"
        )
    lines += [
        "",
        "## 节点与进程边界",
        "",
        "归属进程 PSS 包含已登记的服务、存储服务及其已观察后代；主验收使用该合计，不代表节点物理内存节省。",
        "sandbox_process_pss 仅是明确 sandbox 标签窗口的观测下界。显式登记前的 VM 已由 provider 后代跟踪",
        "计入归属进程合计，但不在 sandbox 标签内；该辅助曲线不是完整 VM 内存积分或计费量。",
        "node 指 AgentEnv 采样所在执行节点。若服务运行在 remote145 的嵌套节点内，这些是内层 Linux 的口径；",
        "outer145 上承载该节点的进程、宿主页缓存和内存回收没有由这些数据证明。外层数据缺失时物理收益仍为未验证。",
        "全节点 AnonPages、Cached、MemAvailable 含背景负载；共享缓存不能分摊为某一沙箱收益。",
        "暂停写快照期间可能同时出现匿名内存下降和缓存增长；必须共同呈现，不能只选匿名内存下降作为物理收益。",
        "",
        "| 任务 | 归属 PSS GiB·s | sandbox 标签 PSS 下界 GiB·s | 节点 anon GiB·s | 节点 cache GiB·s | "
        "节点 available GiB·s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for trial_id, data in summary["trial_measurements"].items():
        curves = data["curves"]
        values = [
            show(curves[key]["integral_gib_seconds"])
            for key in ("assigned_process_pss", "sandbox_process_pss", "node_anon", "node_cache", "node_available")
        ]
        lines.append(f"| {trial_id} | " + " | ".join(values) + " |")
    for pair in summary["pairs"]:
        ids = pair.get("trial_ids", {})
        resident = summary["trial_measurements"].get(ids.get("resident"), {}).get("curves", {})
        offload = summary["trial_measurements"].get(ids.get("offload"), {}).get("curves", {})
        values = [
            data.get(key, {}).get("integral_gib_seconds")
            for data in (resident, offload)
            for key in ("node_anon", "node_cache")
        ]
        if all(number(value) for value in values):
            anon_change, cache_change = values[2] - values[0], values[3] - values[1]
            if anon_change < 0 < cache_change:
                lines += [
                    "",
                    f"配对 {pair['pair_index']} 的节点匿名内存积分减少 {show(-anon_change)} GiB·s，"
                    f"缓存积分同时增加 {show(cache_change)} GiB·s。",
                    "这是两类变化并存的观测，尚不能证明它们一一转移，也不能据此宣称节点物理收益。",
                ]
    measured_cost = show(summary["observed_pss_costs"]["all_started_measured_gib_seconds"])
    lines += [
        "",
        "各曲线峰值、最小值、时钟误差和缺测原因在 summary.json 中保留。",
        "",
        f"已测任务 PSS 成本合计 {measured_cost} GiB·s；",
        f"其中失败任务 {show(summary['observed_pss_costs']['failed_tasks_measured_gib_seconds'])} GiB·s。",
        f"另有 {summary['observed_pss_costs']['unmeasured_tasks']} 个任务缺少可积分观测；这些成本不补零。",
        "此合计保留失败消耗，但各诊断窗口可能不同，不能作为公平的模式间节省率。",
        "",
        "## 磁盘 I/O",
        "",
        "进程 I/O 按 (pid,starttime_ticks) 各自差分后相加；新进程首样本前和退出前未采到的 I/O 不冒充精确值。",
        "结果为观察下界，同时保留每个身份首个累计计数及缺测。设备数据为整个节点，只按单个设备列出；",
        "逻辑设备和底层物理设备可能重复描述同一次 I/O，不能把各设备相加。边界采样之外的字节不外推。",
        "",
        "| 任务 | 设备 | 观察读字节 | 观察写字节 | 状态 |",
        "|---|---|---:|---:|---|",
    ]
    for trial_id, data in summary["trial_measurements"].items():
        for device in data["io"]["disk_devices"]:
            lines.append(
                f"| {trial_id} | {device['device']} ({device['major_minor']}) | "
                f"{show(device['observed_read_bytes'], 0)} | {show(device['observed_write_bytes'], 0)} | "
                f"{device['read_bytes_status']}/{device['write_bytes_status']} |"
            )
    lines += [
        "",
        "## 建议门槛",
        "",
        "门槛由分析参数预先指定；通过进程 PSS 门槛也不能宣称节点物理收益或生产可用。",
        "",
        "| 项目 | 结论 |",
        "|---|---|",
    ]
    labels = {
        "planned_run_complete": "计划任务完整执行",
        "task_correctness_and_cleanup": "已开始任务正确性与清理",
        "assigned_process_pss_saving": "归属进程 PSS 节省",
        "task_p95_increase": "任务 p95 增幅",
        "first_tool_extra_task_p95": "首工具额外 p95",
        "physical_node_saving_attribution": "节点物理节省归因",
    }
    translations = {"pass": "通过", "fail": "未通过", "unverified": "不可验证", "unfinished": "未完成"}
    for key, value in summary["gates"].items():
        lines.append(f"| {labels[key]} | {translations[value]} |")
    lines += [
        "",
        f"整体验收状态：{translations[summary['acceptance_status']]}。",
        "计划未完成时保留已完成配对的诊断数字，未开始任务不加入失败分母，也不能据此宣称整轮验收通过。",
        f"门槛参数：`{json.dumps(summary['thresholds'], ensure_ascii=False)}`。",
        "原始结果在 results.jsonl、各任务 events.jsonl/metrics.jsonl；summary.json 保留完整失败分母与分析口径。",
    ]
    if summary["input_errors"]:
        lines += ["", "输入存在缺失或格式问题：" + "、".join(summary["input_errors"]) + "。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--memory-saving-min", type=float, default=0.2)
    parser.add_argument("--task-p95-increase-max", type=float, default=0.05)
    parser.add_argument("--first-tool-extra-max", type=float, default=1.0)
    parser.add_argument("--max-gap", type=float, default=1.0)
    args = parser.parse_args(argv)
    for value in (args.memory_saving_min, args.task_p95_increase_max, args.first_tool_extra_max, args.max_gap):
        if not math.isfinite(value) or value < 0:
            parser.error("thresholds must be nonnegative finite values")
    if args.max_gap == 0 or args.memory_saving_min > 1:
        parser.error("max-gap must be positive and memory-saving-min must be at most one")
    summary = analyze_directory(
        args.directory,
        memory_saving_min=args.memory_saving_min,
        task_p95_increase_max=args.task_p95_increase_max,
        first_tool_extra_max=args.first_tool_extra_max,
        max_gap=args.max_gap,
    )
    for name, text in (
        ("summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
        ("report.md", report_markdown(summary)),
    ):
        temporary = args.directory / (name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(args.directory / name)
    print(json.dumps({"started_tasks": summary["started_tasks"], "gates": summary["gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
