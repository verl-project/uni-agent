# ruff: noqa: E501
"""Analyze reproduce/logs/*.log to study Modal sandbox retries and concurrency.

Each run writes one ``<run_id>.log`` (see ``reproduce/run.py`` ->
``add_file_handler``). This script parses every log, identifies the two kinds of
retry that happen against Modal, and correlates failures / slowness with how
many runs were active at the same time.

Two retry levels are distinguished:

  * deploy-level   - ModalDeployment.start() retry loop. Markers:
                     "Failed to create modal sandbox: ..." (one per failed
                     attempt) and "Retrying modal deployment startup in Ns...".
  * request-level  - swerex RemoteRuntime retrying a single HTTP request.
                     Marker: "Client error making request <id>: Cannot connect
                     to host ...". The same <id> repeated == retries of one
                     request; distinct <id>s == distinct stuck requests.

Usage:
    python -m reproduce.analyze_logs                  # text report
    python -m reproduce.analyze_logs --logs-dir DIR   # custom log dir
    python -m reproduce.analyze_logs --json           # machine-readable
    python -m reproduce.analyze_logs --list-failures   # print failing run ids
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*(?P<comp>\w+)\s*\|\s*(?P<level>\w+)\s*\|\s*(?P<msg>.*)$"
)
SANDBOX_CREATED_RE = re.compile(r"Sandbox \((?P<sid>sb-\w+)\) created in (?P<secs>[\d.]+)s")
RUNTIME_STARTED_RE = re.compile(r"Runtime started in (?P<secs>[\d.]+)s")
RETRY_RE = re.compile(r"Retrying modal deployment startup in (?P<secs>\d+) seconds")
CONN_ERR_RE = re.compile(r"making request (?P<rid>[0-9a-f-]+): Cannot connect to host")
TIMEOUT_RE = re.compile(r"did not start within (?P<secs>[\d.]+)s")

TS_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass
class RunInfo:
    run_id: str
    start: datetime | None = None
    end: datetime | None = None
    # outcome: resolved | wa | eval_error | patch_no_eval | deploy_failed | incomplete
    outcome: str = "incomplete"
    # timings (from the successful attempt)
    sandbox_create_secs: float | None = None
    runtime_start_secs: float | None = None
    # deploy-level retries
    sandbox_attempts: int = 0
    deploy_failures: int = 0
    deploy_retries: int = 0
    # request-level retries
    conn_error_lines: int = 0
    stuck_requests: set[str] = field(default_factory=set)
    had_timeout: bool = False
    # concurrency (filled in later)
    concurrency_at_start: int = 0

    @property
    def duration(self) -> float | None:
        if self.start and self.end:
            return (self.end - self.start).total_seconds()
        return None

    @property
    def resolved(self) -> bool:
        return self.outcome == "resolved"

    @property
    def failed(self) -> bool:
        return self.outcome in ("eval_error", "patch_no_eval", "deploy_failed", "incomplete")


def parse_log(path: Path) -> RunInfo:
    info = RunInfo(run_id=path.stem)
    saw_patch = saw_eval = False
    for raw in path.read_text(errors="replace").splitlines():
        m = LINE_RE.match(raw)
        if not m:
            continue
        ts = datetime.strptime(m["ts"], TS_FMT)
        msg = m["msg"]
        if info.start is None:
            info.start = ts
        info.end = ts

        if msg.startswith("Starting modal sandbox with image"):
            info.sandbox_attempts += 1
        elif (cm := SANDBOX_CREATED_RE.search(msg)):
            info.sandbox_create_secs = float(cm["secs"])
        elif (rm := RUNTIME_STARTED_RE.search(msg)):
            info.runtime_start_secs = float(rm["secs"])
        elif msg.startswith("Failed to create modal sandbox"):
            info.deploy_failures += 1
            if TIMEOUT_RE.search(msg):
                info.had_timeout = True
        elif RETRY_RE.search(msg):
            info.deploy_retries += 1
        elif (em := CONN_ERR_RE.search(msg)):
            info.conn_error_lines += 1
            info.stuck_requests.add(em["rid"])
        elif msg.startswith("Applied patch successfully"):
            saw_patch = True
        elif msg.startswith("Failed to evaluate"):
            info.outcome = "eval_error"
        elif msg.startswith("Eval report"):
            saw_eval = True
            info.outcome = "resolved" if "'resolved': True" in msg else "wa"

    # classify when no eval report was produced
    if not saw_eval and info.outcome not in ("eval_error",):
        if saw_patch:
            info.outcome = "patch_no_eval"  # patch applied but eval hung/timed out
        elif info.deploy_failures and info.runtime_start_secs is None:
            info.outcome = "deploy_failed"
        else:
            info.outcome = "incomplete"
    return info


def compute_concurrency(runs: list[RunInfo]) -> tuple[int, dict[str, int]]:
    """Return (peak concurrent runs, per-minute active count). Also fills
    each run's concurrency_at_start."""
    events: list[tuple[datetime, int]] = []
    for r in runs:
        if r.start and r.end:
            events.append((r.start, +1))
            events.append((r.end, -1))
    events.sort(key=lambda e: (e[0], -e[1]))
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)

    # concurrency_at_start: active runs at the instant each run begins
    starts_ends = [(r.start, r.end) for r in runs if r.start and r.end]
    for r in runs:
        if not r.start:
            continue
        r.concurrency_at_start = sum(1 for s, e in starts_ends if s <= r.start <= e)

    per_minute: dict[str, int] = {}
    if events:
        t0 = min(r.start for r in runs if r.start)
        t1 = max(r.end for r in runs if r.end)
        minute = t0.replace(second=0)
        while minute <= t1:
            # instantaneous concurrency sampled at the minute boundary, so these
            # bars stay consistent with peak_concurrent_runs (<= peak).
            active = sum(1 for s, e in starts_ends if s <= minute <= e)
            per_minute[minute.strftime("%H:%M")] = active
            minute += timedelta(minutes=1)
    return peak, per_minute


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round((p / 100) * (len(values) - 1)))))
    return values[k]


def _fmt_stats(name: str, values: list[float], unit: str = "s") -> str:
    if not values:
        return f"  {name:<22} (no data)"
    return (
        f"  {name:<22} n={len(values):<4} "
        f"mean={statistics.mean(values):6.1f}{unit}  "
        f"p50={_pct(values, 50):6.1f}{unit}  "
        f"p90={_pct(values, 90):6.1f}{unit}  "
        f"p99={_pct(values, 99):6.1f}{unit}  "
        f"max={max(values):6.1f}{unit}"
    )


def build_report(runs: list[RunInfo]) -> dict:
    peak, per_minute = compute_concurrency(runs)
    n = len(runs)
    outcomes: dict[str, int] = {}
    for r in runs:
        outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1

    runs_with_deploy_retry = [r for r in runs if r.deploy_failures > 0]
    runs_with_conn_err = [r for r in runs if r.conn_error_lines > 0]

    # concurrency-bucketed view: does pressure cause retries / slowness?
    buckets: dict[str, dict] = {}
    if runs:
        cmax = max((r.concurrency_at_start for r in runs), default=0)
        edges = [(1, 8), (9, 16), (17, 32), (33, 64), (65, cmax if cmax > 64 else 64)]
        for lo, hi in edges:
            sub = [r for r in runs if lo <= r.concurrency_at_start <= hi]
            if not sub:
                continue
            ct = [r.sandbox_create_secs for r in sub if r.sandbox_create_secs is not None]
            buckets[f"{lo}-{hi}"] = {
                "runs": len(sub),
                "deploy_retry_rate": round(sum(1 for r in sub if r.deploy_failures) / len(sub), 3),
                "conn_err_rate": round(sum(1 for r in sub if r.conn_error_lines) / len(sub), 3),
                "fail_rate": round(sum(1 for r in sub if r.failed) / len(sub), 3),
                "mean_sandbox_create_s": round(statistics.mean(ct), 1) if ct else None,
            }

    return {
        "total_runs": n,
        "outcomes": outcomes,
        "resolved_rate": round(outcomes.get("resolved", 0) / n, 4) if n else 0.0,
        "deploy_retry": {
            "runs_with_failed_attempt": len(runs_with_deploy_retry),
            "total_failed_attempts": sum(r.deploy_failures for r in runs),
            "total_retries": sum(r.deploy_retries for r in runs),
            "runs_hit_startup_timeout": sum(1 for r in runs if r.had_timeout),
        },
        "request_retry": {
            "runs_with_conn_errors": len(runs_with_conn_err),
            "total_conn_error_lines": sum(r.conn_error_lines for r in runs),
            "distinct_stuck_requests": sum(len(r.stuck_requests) for r in runs),
        },
        "concurrency": {
            "peak_concurrent_runs": peak,
            "per_minute_active": per_minute,
        },
        "concurrency_buckets": buckets,
        "_runs": runs,  # kept for text rendering / failure listing
    }


def print_report(rep: dict, list_failures: bool) -> None:
    runs: list[RunInfo] = rep["_runs"]
    n = rep["total_runs"]
    print("=" * 72)
    print(f" Modal concurrency analysis  ({n} runs)")
    print("=" * 72)

    print("\n[Outcomes]")
    labels = {
        "resolved": "resolved        ",
        "wa": "wrong-answer    ",
        "eval_error": "eval error      ",
        "patch_no_eval": "patch, no eval  ",
        "deploy_failed": "deploy failed   ",
        "incomplete": "incomplete      ",
    }
    for key, label in labels.items():
        c = rep["outcomes"].get(key, 0)
        if c:
            print(f"  {label} {c:>4}  ({c / n:.1%})")
    print(f"  resolved rate         {rep['resolved_rate']:.2%}")

    dr = rep["deploy_retry"]
    print("\n[Deploy-level retries]  (ModalDeployment.start loop)")
    print(f"  runs with >=1 failed attempt   {dr['runs_with_failed_attempt']:>4}  ({dr['runs_with_failed_attempt'] / n:.1%})")
    print(f"  total failed attempts          {dr['total_failed_attempts']:>4}")
    print(f"  total retries triggered        {dr['total_retries']:>4}")
    print(f"  runs hitting startup timeout   {dr['runs_hit_startup_timeout']:>4}")

    rr = rep["request_retry"]
    print("\n[Request-level retries]  (swerex RemoteRuntime -> sandbox host)")
    print(f"  runs with connection errors    {rr['runs_with_conn_errors']:>4}  ({rr['runs_with_conn_errors'] / n:.1%})")
    print(f"  distinct stuck requests        {rr['distinct_stuck_requests']:>4}")
    print(f"  total 'Cannot connect' lines   {rr['total_conn_error_lines']:>4}")

    print("\n[Timings]")
    print(_fmt_stats("sandbox create", [r.sandbox_create_secs for r in runs if r.sandbox_create_secs is not None]))
    print(_fmt_stats("runtime start", [r.runtime_start_secs for r in runs if r.runtime_start_secs is not None]))
    print(_fmt_stats("run wall time", [d for r in runs if (d := r.duration) is not None]))

    cc = rep["concurrency"]
    print("\n[Concurrency]")
    print(f"  peak concurrent runs           {cc['peak_concurrent_runs']:>4}")
    pm = cc["per_minute_active"]
    if pm:
        print("  active runs per minute:")
        for minute, active in pm.items():
            bar = "█" * active
            print(f"    {minute}  {active:>3} {bar}")

    if rep["concurrency_buckets"]:
        print("\n[Pressure vs retries]  (bucketed by #active runs when a run started)")
        print(f"  {'bucket':>8} {'runs':>5} {'deploy_retry':>13} {'conn_err':>9} {'fail':>6} {'mean_create':>12}")
        for b, d in rep["concurrency_buckets"].items():
            mc = f"{d['mean_sandbox_create_s']}s" if d["mean_sandbox_create_s"] is not None else "-"
            print(f"  {b:>8} {d['runs']:>5} {d['deploy_retry_rate']:>12.1%} {d['conn_err_rate']:>8.1%} {d['fail_rate']:>5.1%} {mc:>12}")

    if list_failures:
        print("\n[Failing runs]")
        for r in runs:
            if r.failed:
                print(f"  {r.run_id}  outcome={r.outcome:<14} attempts={r.sandbox_attempts} conn_errs={r.conn_error_lines} dur={r.duration}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Modal sandbox retries / concurrency from reproduce logs.")
    ap.add_argument("--logs-dir", default="reproduce/logs", help="directory of <run_id>.log files")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    ap.add_argument("--list-failures", action="store_true", help="list failing run ids")
    args = ap.parse_args()

    log_dir = Path(args.logs_dir)
    paths = sorted(log_dir.glob("*.log"))
    if not paths:
        raise SystemExit(f"No .log files found in {log_dir}")

    runs = [parse_log(p) for p in paths]
    rep = build_report(runs)

    if args.json:
        payload = {k: v for k, v in rep.items() if k != "_runs"}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_report(rep, list_failures=args.list_failures)


if __name__ == "__main__":
    main()
