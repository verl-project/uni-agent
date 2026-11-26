#!/usr/bin/env python3
"""Plot an existing offload result directory without changing its analysis.

Requires matplotlib and numpy. For a local environment contaminated by user-site
packages, invoke Python with -s. Example:
python -s plot_results.py OUTPUT_DIR --output comparison.png
The output extension may be .png or .svg; all observations come from saved data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Patch

MODES = ("resident", "offload")
COLORS = {"resident": "#245B87", "offload": "#23856D"}
BUDGET_COLOR = "#A65E2E"
MIB = 1024**2
GIB = 1024**3
RESOURCE_COLORS = {"resident": "#2864D9", "offload": "#E46B16"}
RESOURCE_NAMES = {"resident": "常驻", "offload": "卸载恢复"}


def finite(value):
    return (
        isinstance(value, int | float | np.integer | np.floating)
        and not isinstance(value, bool | np.bool_)
        and math.isfinite(value)
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as error:
                message = f"Invalid JSON in {path.name}, line {index + 1}; no rows were silently dropped"
                raise ValueError(message) from error
    return rows


def pss_bytes(row):
    if row.get("registry_error") or row.get("selection_coverage_known") is not True:
        return float("nan")
    if not isinstance(row.get("processes"), list):
        return float("nan")
    total = 0
    for process in row["processes"]:
        value = process.get("memory", {}).get("pss_bytes")
        if process.get("status") != "ok" or not finite(value):
            return float("nan")
        total += value
    return total


def raw_curve(directory, max_gap):
    result = read_json(directory / "result.json")
    clock = result.get("clock_calibration") or {}
    offset = clock.get("client_minus_node_seconds")
    origin = result.get("measurement_started_monotonic_seconds")
    if not finite(offset) or not finite(origin):
        raise ValueError(f"{directory.name}: a calibrated monotonic time origin is required")
    times, values = [], []
    previous = None
    for row in read_jsonl(directory / "metrics.jsonl"):
        node_time = row.get("monotonic_seconds")
        if not finite(node_time):
            raise ValueError(f"{directory.name}: a raw sample has no valid timestamp")
        elapsed = node_time + offset - origin
        if previous is not None:
            if elapsed <= previous:
                raise ValueError(f"{directory.name}: raw sample timestamps are not increasing")
            if elapsed - previous > max_gap:
                times.append((previous + elapsed) / 2)
                values.append(float("nan"))
        times.append(elapsed)
        values.append(pss_bytes(row) / MIB)
        previous = elapsed
    return np.asarray(times), np.asarray(values)


def percentile(values, fraction):
    values = sorted(value for value in values if finite(value))
    if not values:
        return None
    location = (len(values) - 1) * fraction
    left, right = math.floor(location), math.ceil(location)
    return values[left] + (values[right] - values[left]) * (location - left)


def clip_observations(times, values, horizon):
    """Retain raw points; interpolate only the two integration boundaries."""

    def boundary(moment):
        right = int(np.searchsorted(times, moment))
        if right < len(times) and times[right] == moment:
            return values[right]
        if right == 0 or right == len(times):
            return float("nan")
        left = right - 1
        if not finite(values[left]) or not finite(values[right]):
            return float("nan")
        fraction = (moment - times[left]) / (times[right] - times[left])
        return values[left] + fraction * (values[right] - values[left])

    chosen = (times > 0) & (times < horizon)
    return (
        np.concatenate(([0.0], times[chosen], [horizon])),
        np.concatenate(([boundary(0.0)], values[chosen], [boundary(horizon)])),
    )


def cumulative_pss(times, values):
    """Absolute GiB-seconds become unknown after any missing interval."""
    output = np.full(len(times), np.nan)
    if not len(times) or not finite(values[0]):
        return output
    output[0] = 0.0
    for index in range(1, len(times)):
        if not all(finite(value) for value in (output[index - 1], values[index - 1], values[index])):
            break
        output[index] = (
            output[index - 1] + (values[index - 1] + values[index]) / 2 * (times[index] - times[index - 1]) / GIB
        )
    return output


def lifecycle_transitions(result, events):
    """Known business-VM phases, with transition periods deliberately unknown."""
    origin = result["measurement_started_monotonic_seconds"]
    transitions = []
    for event in events:
        moment = event.get("monotonic_seconds")
        if not finite(moment):
            continue
        name = event.get("event")
        elapsed = moment - origin
        if name == "created" and finite(result.get("create_seconds")):
            transitions.append((elapsed - result["create_seconds"], None, None))
        elif name == "node_observation" and event.get("phase") == "before_create":
            nodes = event.get("nodes")
            if isinstance(nodes, list) and len(nodes) == 1 and nodes[0].get("sandboxCount") == 0:
                transitions.append((elapsed, 0, None))
        elif name == "firecracker_identity":
            processes = event.get("processes", [])
            if len(processes) == 1:
                process = processes[0]
                identity = (process.get("pid"), process.get("starttime_ticks"))
                transitions.append((elapsed, 1, identity))
        elif name in {"pause_requested", "resume_requested", "delete_requested"}:
            transitions.append((elapsed, None, None))
        elif name in {"resources_released", "cleanup_confirmed"}:
            transitions.append((elapsed, 0, None))
    return sorted(transitions, key=lambda item: item[0])


def verified_vm_count(row, elapsed, transitions):
    if row.get("registry_error") or row.get("selection_coverage_known") is not True:
        return float("nan")
    if not isinstance(row.get("processes"), list):
        return float("nan")
    phase, identity = None, None
    for moment, observed_phase, observed_identity in transitions:
        if moment > elapsed:
            break
        phase, identity = observed_phase, observed_identity
    if phase == 0:
        return 0.0
    if phase != 1:
        return float("nan")
    matching = [
        process for process in row["processes"] if (process.get("pid"), process.get("starttime_ticks")) == identity
    ]
    if len(matching) != 1 or matching[0].get("status") != "ok":
        return float("nan")
    if matching[0].get("state") == "Z" or matching[0].get("address_space_released"):
        return float("nan")
    return 1.0


def resource_observations(directory, max_gap):
    result = read_json(directory / "result.json")
    clock = result.get("clock_calibration") or {}
    offset, uncertainty = clock.get("client_minus_node_seconds"), clock.get("uncertainty_seconds")
    origin = result.get("measurement_started_monotonic_seconds")
    if not all(finite(value) for value in (offset, uncertainty, origin)) or uncertainty < 0:
        raise ValueError(f"{directory.name}: calibrated monotonic time and uncertainty are required")
    transitions = lifecycle_transitions(result, read_jsonl(directory / "events.jsonl"))
    series = {key: [] for key in ("pss", "anon", "cache", "vm")}
    times, durations = [], []
    previous = None
    for row in read_jsonl(directory / "metrics.jsonl"):
        node_time = row.get("monotonic_seconds")
        if not finite(node_time):
            raise ValueError(f"{directory.name}: raw sample timestamp is invalid")
        elapsed = node_time + offset - origin
        if previous is not None:
            if elapsed <= previous:
                raise ValueError(f"{directory.name}: raw sample timestamps are not increasing")
            if elapsed - previous > max_gap:
                times.append((elapsed + previous) / 2)
                for values in series.values():
                    values.append(float("nan"))
        memory = row.get("node", {}).get("memory_bytes", {})
        times.append(elapsed)
        series["pss"].append(pss_bytes(row))
        series["anon"].append(memory.get("AnonPages") if finite(memory.get("AnonPages")) else float("nan"))
        series["cache"].append(memory.get("Cached") if finite(memory.get("Cached")) else float("nan"))
        series["vm"].append(verified_vm_count(row, elapsed, transitions))
        if finite(row.get("sample_duration_seconds")):
            durations.append(row["sample_duration_seconds"])
        previous = elapsed
    return {
        "times": np.asarray(times),
        "series": {key: np.asarray(value) for key, value in series.items()},
        "clock_uncertainty": uncertainty,
        "max_sample_duration": max(durations, default=0),
        "result": result,
    }


def resource_font():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei", "WenQuanYi Zen Hei"):
        if candidate in installed:
            return candidate
    raise ValueError("Resource cards require a CJK font, for example Microsoft YaHei or Noto Sans CJK SC")


def resume_lead_annotation(parameters):
    lead = parameters.get("resume_lead_seconds", 0)
    if not finite(lead) or lead <= 0:
        return None
    return f"提前 {lead:g} 秒预恢复；已知合成期限，底层恢复未加速"


def resource_card(figure, rectangle, title, subtitle):
    left, bottom, width, height = rectangle
    figure.add_artist(
        FancyBboxPatch(
            (left, bottom),
            width,
            height,
            boxstyle="round,pad=0.003,rounding_size=0.01",
            transform=figure.transFigure,
            linewidth=0.8,
            edgecolor="#DCE3ED",
            facecolor="white",
            zorder=-10,
        )
    )
    figure.text(
        left + 0.013, bottom + height - 0.028, title, fontsize=13.5, color="#172033", va="top", fontweight="medium"
    )
    figure.text(left + 0.013, bottom + height - 0.061, subtitle, fontsize=8.7, color="#627086", va="top")
    axis = figure.add_axes((left + 0.052, bottom + 0.052, width - 0.071, height - 0.137))
    axis.set_facecolor("white")
    axis.grid(color="#E4E9F0", linewidth=0.65)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#D8DFE9")
    axis.tick_params(colors="#627086", labelsize=8.5, length=0, pad=6)
    axis.set_xlabel("距测量起点的时间（秒）", fontsize=9, color="#536177", labelpad=8)
    axis.yaxis.label.set_color("#536177")
    return axis


def plot_resource_cards(root, summary, manifest, pair):
    horizon = pair.get("common_window_seconds")
    if not finite(horizon) or horizon <= 0:
        raise ValueError("Resource cards require an explicitly recorded positive common window")
    max_gap = summary.get("thresholds", {}).get("max_sample_gap_seconds", 1.0)
    observations = {mode: resource_observations(root / pair["trial_ids"][mode], max_gap) for mode in MODES}
    plt.rcParams.update(
        {
            "font.family": resource_font(),
            "font.size": 9,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "axes.labelcolor": "#536177",
            "text.color": "#172033",
        }
    )
    figure = plt.figure(figsize=(16, 11.2), facecolor="#F1F4F9")
    parameters = manifest.get("parameters", {})
    pairs = summary.get("paired_summary", {}).get("independent_pairs", len(summary.get("pairs", [])))
    figure.text(
        0.035,
        0.972,
        f"AgentEnv 资源时间对照 · {parameters.get('memory_mib', '?')} MiB 内存状态",
        va="top",
        fontsize=23,
        fontweight="medium",
    )
    figure.text(
        0.035,
        0.926,
        f"外部合成等待 {parameters.get('wait_seconds', '?')} 秒 × {parameters.get('cycles', '?')} 周期  |  "
        f"全组 {pairs} 对，展示配对 {pair['pair_index']}  |  共同窗口 {horizon:.2f} 秒",
        fontsize=10.6,
        color="#536177",
    )
    lead_note = resume_lead_annotation(parameters)
    if lead_note:
        figure.text(0.035, 0.902, lead_note, fontsize=10, color="#8A4E22")
    legend_height = 0.878 if lead_note else 0.893
    figure.text(0.035, legend_height, "━━ 常驻 resident", fontsize=11, color=RESOURCE_COLORS["resident"])
    figure.text(0.515, legend_height, "━━ 卸载恢复 offload", fontsize=11, color=RESOURCE_COLORS["offload"])
    axes = [
        resource_card(figure, (0.035, 0.505, 0.447, 0.357), "归属进程 PSS", "服务、存储与已观察后代，包含预热池成本"),
        resource_card(
            figure, (0.515, 0.505, 0.447, 0.357), "累计 PSS 占用", "按原始点梯形积分；缺测后的完整累计值保持未知"
        ),
        resource_card(
            figure,
            (0.035, 0.105, 0.447, 0.357),
            "业务 VM 驻留 · 已验证阶段",
            "1 = 已归属的活动实例，0 = 已确认卸载；转换阶段留白",
        ),
        resource_card(
            figure, (0.515, 0.105, 0.447, 0.357), "节点匿名页与缓存变化", "各自相对测量起点；实线为匿名页，虚线为缓存"
        ),
    ]
    endpoints, clock_bounds, sample_bounds = {}, [], []
    complete_window = pair.get("status") == "observed"
    for mode in MODES:
        observed = observations[mode]
        color = RESOURCE_COLORS[mode]
        times, raw = observed["times"], observed["series"]
        complete_window = (
            complete_window
            and bool(len(times))
            and (times[0] <= -observed["clock_uncertainty"] and times[-1] >= horizon + observed["clock_uncertainty"])
        )
        x, y = clip_observations(times, raw["pss"], horizon)
        axes[0].plot(x, y / MIB, color=color, linewidth=1.55, label=RESOURCE_NAMES[mode])
        cumulative = cumulative_pss(x, y)
        endpoints[mode] = cumulative[-1]
        suffix = f"{cumulative[-1]:.3f} GiB·s" if finite(cumulative[-1]) else "累计未知"
        axes[1].plot(x, cumulative, color=color, linewidth=1.7, label=f"{RESOURCE_NAMES[mode]} {suffix}")
        inside = (times >= 0) & (times <= horizon)
        axes[2].plot(
            times[inside],
            raw["vm"][inside],
            color=color,
            linewidth=1.6,
            drawstyle="steps-post",
            label=RESOURCE_NAMES[mode],
        )
        for metric, style, label in (("anon", "-", "匿名页"), ("cache", "--", "缓存")):
            node_x, node_y = clip_observations(times, raw[metric], horizon)
            delta = (node_y - node_y[0]) / MIB if finite(node_y[0]) else np.full(len(node_y), np.nan)
            axes[3].plot(
                node_x, delta, color=color, linestyle=style, linewidth=1.35, label=f"{RESOURCE_NAMES[mode]}·{label}"
            )
        clock_bounds.append(observed["clock_uncertainty"])
        sample_bounds.append(observed["max_sample_duration"])
    for axis in axes:
        axis.set_xlim(0, horizon)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("MiB", fontsize=9)
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("GiB·s", fontsize=9)
    axes[2].set_ylim(-0.12, 1.18)
    axes[2].set_yticks([0, 1])
    axes[2].set_ylabel("业务 VM 数", fontsize=9)
    axes[3].set_ylabel("变化量（MiB）", fontsize=9)
    axes[3].axhline(0, color="#AAB5C4", linewidth=0.75, zorder=0)
    axes[1].legend(loc="upper left", fontsize=8, frameon=False)
    axes[3].legend(loc="best", fontsize=7.8, ncol=2, frameon=True, facecolor="white", framealpha=0.94)
    if complete_window and all(finite(endpoints[mode]) for mode in MODES) and endpoints["resident"] > 0:
        saving = 1 - endpoints["offload"] / endpoints["resident"]
        axes[1].text(
            0.98,
            0.04,
            f"本配对累计下降 {saving:.2%}",
            transform=axes[1].transAxes,
            ha="right",
            fontsize=10,
            color="#34445C",
        )
    else:
        axes[1].text(
            0.98,
            0.04,
            "共同窗未完整验证：不报告完整节省率",
            transform=axes[1].transAxes,
            ha="right",
            fontsize=8.5,
            color="#A05C21",
        )
    figure.text(
        0.035,
        0.065,
        "原始采样点，无平滑；缺样与超长采样间隔保留断线。VM 图仅显示已验证业务阶段，预热池仍计入前两图。",
        fontsize=9,
        color="#627086",
    )
    figure.text(
        0.035,
        0.042,
        f"口径：嵌套节点的归属进程 PSS / 全节点匿名页与缓存，非外层物理 RAM 净节省。"
        f"时钟误差界 ≤ {max(clock_bounds):.3f}s；单次采样耗时 ≤ {max(sample_bounds):.3f}s。",
        fontsize=8.6,
        color="#627086",
    )
    figure.text(
        0.035,
        0.020,
        f"来源：{root.name}  |  低/高熵：{parameters.get('entropy', '?')}  |  "
        f"dirty-ratio：{parameters.get('dirty_ratio', '?')}  |  同部署池与缓存设置，缓存不是免费资源",
        fontsize=8.3,
        color="#627086",
    )
    return figure


def style_axes(axis, title, ylabel):
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#DDE3E8", linewidth=0.65)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=9)


def show(value, digits=3):
    return f"{value:.{digits}f}" if finite(value) else "unverified"


def plot_pss(axis, root, summary, pair):
    style_axes(axis, f"A  Assigned-process PSS | pair {pair['pair_index']}", "PSS (MiB)")
    max_gap = summary.get("thresholds", {}).get("max_sample_gap_seconds", 1.0)
    for mode in MODES:
        trial_id = pair["trial_ids"][mode]
        times, values = raw_curve(root / trial_id, max_gap)
        measurement = summary.get("trial_measurements", {}).get(trial_id, {})
        integral = measurement.get("curves", {}).get("assigned_process_pss", {}).get("integral_gib_seconds")
        axis.plot(times, values, color=COLORS[mode], linewidth=1.65, label=f"{mode}: {show(integral)} GiB s")
    horizon = pair.get("common_window_seconds")
    if finite(horizon):
        axis.axvline(horizon, color="#687480", linestyle="--", linewidth=1)
        axis.text(
            horizon - 0.8,
            0.98,
            f"common window\nends {horizon:.2f}s",
            ha="right",
            va="top",
            transform=axis.get_xaxis_transform(),
            fontsize=8,
            color="#4C5965",
        )
    axis.set_xlabel("Seconds from measurement start (calibrated)")
    axis.set_ylim(bottom=0)
    axis.legend(loc="upper left", fontsize=8, frameon=True, facecolor="white", framealpha=0.95)
    saving = pair.get("assigned_process_pss_saving_fraction")
    axis.text(
        0.02,
        0.04,
        f"Common-window PSS saving: {show(saving * 100 if finite(saving) else None)}%\n"
        "Includes provider, storage and observed descendants; gaps stay gaps.",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 3},
    )


def plot_task_latency(axis, summary, results):
    style_axes(axis, "B  Task latency | successful tasks", "Seconds")
    width = 0.31
    all_values = []
    labels = []
    for index, mode in enumerate(MODES):
        data = summary["by_mode"][mode]
        stats = data["task_seconds_successful"]
        labels.append(f"{mode}\n{data['successful_tasks']}/{data['started_tasks']} succeeded")
        for shift, quantile, hatch in ((-width / 2, "p50", None), (width / 2, "p95", "///")):
            value = stats.get(quantile)
            if finite(value):
                axis.bar(
                    index + shift,
                    value,
                    width,
                    color=COLORS[mode],
                    alpha=0.75 if hatch else 1,
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=0.7,
                )
                axis.text(index + shift, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
                all_values.append(value)
        values = [row.get("task_seconds") for row in results if row.get("mode") == mode and row.get("success") is True]
        values = [value for value in values if finite(value)]
        if values:
            axis.scatter(np.full(len(values), index), values, s=18, color="#23313E", zorder=4)
    axis.set_xticks(range(2), labels)
    axis.set_ylim(0, max(all_values, default=1) * 1.3)
    axis.legend(
        handles=[
            Patch(facecolor="#7A8792", label="p50"),
            Patch(facecolor="#7A8792", hatch="///", edgecolor="white", label="p95"),
        ],
        loc="upper left",
        fontsize=8,
        ncol=2,
    )
    change = summary["paired_summary"].get("task_p95_relative_increase")
    axis.text(
        0.98,
        0.98,
        f"Paired-cohort p95 change: {show(change * 100 if finite(change) else None)}%",
        ha="right",
        va="top",
        transform=axis.transAxes,
        fontsize=8,
    )


def plot_first_tool(axis, summary, results):
    style_axes(axis, "C  First tool after wait | per-task p95", "Seconds, including restore + full-RAM oracle")
    values_by_mode = {}
    for mode in MODES:
        values_by_mode[mode] = [
            percentile([cycle.get("first_tool_seconds") for cycle in row.get("cycles", [])], 0.95)
            for row in results
            if row.get("mode") == mode and row.get("success") is True
        ]
        values_by_mode[mode] = [value for value in values_by_mode[mode] if finite(value)]
    heights = []
    for index, mode in enumerate(MODES):
        value = summary["by_mode"][mode]["first_tool_seconds_task_p95"].get("p95")
        if finite(value):
            axis.bar(index, value, width=0.5, color=COLORS[mode])
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
            heights.append(value)
        observed = values_by_mode[mode]
        if observed:
            jitter = np.linspace(-0.08, 0.08, len(observed)) if len(observed) > 1 else np.array([0.0])
            axis.scatter(index + jitter, observed, color="#1E2D39", edgecolor="white", linewidth=0.4, s=24, zorder=4)
    baseline = summary["by_mode"]["resident"]["first_tool_seconds_task_p95"].get("p95")
    budget = summary.get("thresholds", {}).get("first_tool_extra_max_seconds", 1.0)
    if finite(baseline) and finite(budget):
        reference = baseline + budget
        axis.axhline(reference, color=BUDGET_COLOR, linestyle="--", linewidth=1.15)
        axis.text(
            0.02,
            reference,
            f" resident p95 + {budget:.2f}s reference",
            color=BUDGET_COLOR,
            fontsize=8,
            va="bottom",
            transform=axis.get_yaxis_transform(),
        )
        heights.append(reference)
    extra = summary["paired_summary"].get("first_tool_extra_seconds_task_p95")
    axis.text(
        0.02,
        0.98,
        f"Paired extra p95: {show(extra)}s (budget {show(budget)}s)\n"
        "Line is a visual reference; the gate uses paired cycle deltas.\n"
        "Dots: one p95 per task; bars: p95 across independent tasks.",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
    )
    axis.set_xticks(range(2), [f"{mode}\nN={len(values_by_mode[mode])} task(s)" for mode in MODES])
    axis.set_ylim(0, max(heights, default=1) * 1.47)


def plot_disk(axis, summary, device):
    style_axes(axis, f"D  Root-device I/O | {device} only", "Observed delta per task (MB, decimal)")
    values = {mode: {"read": [], "write": []} for mode in MODES}
    accepted_pairs = 0
    for pair in summary["pairs"]:
        ids = pair.get("trial_ids", {})
        selected = {}
        for mode in MODES:
            rows = summary.get("trial_measurements", {}).get(ids.get(mode), {}).get("io", {}).get("disk_devices", [])
            matches = [row for row in rows if row.get("device") == device]
            if len(matches) == 1 and all(
                finite(matches[0].get(f"observed_{kind}_bytes")) for kind in ("read", "write")
            ):
                selected[mode] = matches[0]
        if len(selected) != 2:
            continue
        accepted_pairs += 1
        for mode in MODES:
            for kind in ("read", "write"):
                values[mode][kind].append(selected[mode][f"observed_{kind}_bytes"] / 1e6)
    width, heights = 0.32, []
    for mode_index, mode in enumerate(MODES):
        for kind_index, kind in enumerate(("read", "write")):
            observations = values[mode][kind]
            position = kind_index + (mode_index - 0.5) * width
            if observations:
                average = sum(observations) / len(observations)
                axis.bar(position, average, width, color=COLORS[mode], label=mode if kind_index == 0 else None)
                axis.text(position, average, f"{average:.2f}", ha="center", va="bottom", fontsize=9)
                jitter = np.linspace(-0.055, 0.055, len(observations)) if len(observations) > 1 else np.array([0.0])
                axis.scatter(position + jitter, observations, s=18, color="#1E2D39", zorder=4)
                heights += observations
    axis.set_xticks(range(2), ["Read", "Write"])
    axis.set_ylim(0, max(heights, default=1) * 1.35 or 1)
    if accepted_pairs:
        axis.legend(loc="upper left", fontsize=8, ncol=2)
    else:
        axis.text(0.5, 0.5, "No paired observations for this device", ha="center", transform=axis.transAxes)
    axis.text(
        0.98,
        0.98,
        f"Mean per task; dots = task observations\n"
        f"{accepted_pairs}/{len(summary['pairs'])} pairs observed; sampled boundary tails excluded\n"
        "Whole nested node; no sum over stacked devices.",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pair", type=int, help="Pair index for the complete raw PSS traces; defaults to the first pair"
    )
    parser.add_argument("--device", default="vda", help="One root device, never a sum over multiple storage layers")
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument(
        "--view",
        choices=("summary", "resources"),
        default="summary",
        help="summary: latency/I/O overview; resources: four raw resource-time cards",
    )
    args = parser.parse_args(argv)
    if args.output.suffix.lower() not in {".png", ".svg"}:
        parser.error("--output must end in .png or .svg")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    summary = read_json(args.directory / "summary.json")
    manifest = read_json(args.directory / "manifest.json")
    results = read_jsonl(args.directory / "results.jsonl")
    candidates = [
        pair
        for pair in summary["pairs"]
        if all(mode in pair.get("trial_ids", {}) for mode in MODES)
        and (args.pair is None or pair.get("pair_index") == args.pair)
    ]
    if not candidates:
        parser.error("no pair with both trial IDs is available for raw-curve plotting")
    pair = candidates[0]
    if args.view == "resources":
        figure = plot_resource_cards(args.directory, summary, manifest, pair)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=args.dpi, facecolor=figure.get_facecolor())
        plt.close(figure)
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "view": args.view,
                    "pair_index": pair["pair_index"],
                    "raw_source": str(args.directory),
                }
            )
        )
        return 0
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
            "axes.labelcolor": "#33414D",
            "text.color": "#24333F",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 10.2))
    figure.subplots_adjust(left=0.075, right=0.965, top=0.86, bottom=0.14, hspace=0.42, wspace=0.25)
    figure.suptitle(
        "AgentEnv sandbox: resident vs offload", x=0.075, y=0.972, ha="left", fontsize=21, fontweight="bold"
    )
    count = summary["paired_summary"]["independent_pairs"]
    success = sum(summary["by_mode"][mode]["successful_tasks"] for mode in MODES)
    parameters = manifest.get("parameters", {})
    figure.text(
        0.075,
        0.927,
        f"N={count} independent pair(s), {summary['started_tasks']} started tasks, {success} successful | "
        f"{parameters.get('cycles', '?')} cycles/task | "
        f"{parameters.get('wait_seconds', '?')}s synthetic wait | "
        f"{parameters.get('memory_mib', '?')} MiB probe",
        fontsize=10.5,
    )
    lead_note = resume_lead_annotation(parameters)
    if lead_note:
        figure.text(
            0.075,
            0.898,
            lead_note + "；重复周期不视为独立任务。",
            fontsize=10,
            color="#8A4E22",
            fontfamily=resource_font(),
        )
    else:
        figure.text(
            0.075,
            0.898,
            "Exploratory sample: repeated cycles are not independent tasks; plotted gates are unchanged.",
            fontsize=9,
            color="#596671",
        )
    plot_pss(axes[0, 0], args.directory, summary, pair)
    plot_task_latency(axes[0, 1], summary, results)
    plot_first_tool(axes[1, 0], summary, results)
    plot_disk(axes[1, 1], summary, args.device)
    figure.text(
        0.075,
        0.075,
        "Scope: synthetic wait; full-RAM oracle; assigned-process PSS; nested Linux node. "
        "Physical host saving is unverified.",
        fontsize=9,
        fontweight="bold",
    )
    figure.text(
        0.075,
        0.049,
        "Memory includes provider/storage processes and observed descendants, including warm runtimes. "
        "Node page cache is a separate cost; PSS is not total host RAM.",
        fontsize=8.5,
    )
    figure.text(
        0.075,
        0.024,
        f"Source: {args.directory.name} | acceptance: {summary.get('acceptance_status', 'unverified')}",
        fontsize=8,
        color="#596671",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, facecolor="white")
    plt.close(figure)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "pss_pair_index": pair["pair_index"],
                "independent_pairs": count,
                "device": args.device,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
