"""Measure cancellation bursts without including observer ps calls in loop lag.

Run with PYTHONPATH pointing at the checkout under test. All workers are local,
finite-lived fixtures. Residual targets are recorded BEFORE emergency cleanup.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def worker(directory: Path, role: str, detached: bool) -> None:
    record = {"pid": os.getpid(), "pgid": os.getpgrp()}
    tmp = directory / f"{role}.tmp"
    tmp.write_text(json.dumps(record))
    tmp.replace(directory / f"{role}.json")
    if role == "parent":
        subprocess.Popen(
            [sys.executable, __file__, "--worker", str(directory), "child", str(int(detached))],
            start_new_session=detached,
        )
    time.sleep(60)


def live(pids: list[int]) -> list[int]:
    # One passive snapshot. Failed observations are errors, never proof of exit.
    result = subprocess.run(["ps", "-A", "-o", "pid=,stat="], capture_output=True, text=True, check=True, timeout=5)
    states = {int(row.split()[0]): row.split()[1] for row in result.stdout.splitlines()}
    return [pid for pid in pids if pid in states and not states[pid].startswith("Z")]


async def run_case(count: int, detached: bool, delay_ms: float, round_number: int) -> dict:
    from uni_agent.sandbox import local

    with tempfile.TemporaryDirectory(prefix="ua-cleanup-burst-") as temporary:
        root = Path(temporary)
        directories = [root / str(index) for index in range(count)]
        tasks = []
        pids = []
        rescue = []
        ps_times = []
        lag = []
        monitor = None
        original_run = subprocess.run
        sibling = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            start_new_session=True,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        async def heartbeat():
            previous = time.perf_counter()
            while True:
                await asyncio.sleep(0.005)
                now = time.perf_counter()
                lag.append(max(0, now - previous - 0.005))
                previous = now

        def timed_run(argv, *args, **kwargs):
            if argv == ["ps", "-A", "-o", "pid=,ppid=,pgid="]:
                start = time.perf_counter()
                try:
                    if delay_ms:
                        time.sleep(delay_ms / 1000)
                    return original_run(argv, *args, **kwargs)
                finally:
                    ps_times.append(time.perf_counter() - start)
            return original_run(argv, *args, **kwargs)

        try:
            for directory in directories:
                directory.mkdir()
                tasks.append(
                    asyncio.create_task(
                        local.LocalSandbox().exec(
                            [sys.executable, __file__, "--worker", str(directory), "parent", str(int(detached))],
                            timeout=45,
                        )
                    )
                )
            deadline = time.perf_counter() + 15
            while not all((d / "child.json").exists() for d in directories):
                if time.perf_counter() > deadline:
                    raise AssertionError("fixture startup deadline exceeded")
                await asyncio.sleep(0.01)
            pids = [json.loads(p.read_text())["pid"] for d in directories for p in d.glob("*.json")]
            assert len(pids) == 2 * count and len(live(pids)) == len(pids)
            assert live([sibling.pid]) == [sibling.pid]
            subprocess.run = timed_run
            monitor = asyncio.create_task(heartbeat())
            await asyncio.sleep(0.02)
            started = time.perf_counter()
            for task in tasks:
                task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.perf_counter() - started
            assert all(isinstance(result, asyncio.CancelledError) for result in results)
            await asyncio.sleep(0.02)
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            subprocess.run = original_run
            deadline = time.perf_counter() + 5
            while live(pids) and time.perf_counter() < deadline:
                await asyncio.sleep(0.02)
            remaining = live(pids)
            sibling_alive = bool(live([sibling.pid]))
            record = {
                "count": count,
                "detached": detached,
                "injected_ps_delay_ms": delay_ms,
                "round": round_number,
                "cancel_return_ms": elapsed * 1000,
                "max_loop_lag_ms": max(lag, default=0) * 1000,
                "ps_calls": len(ps_times),
                "total_ps_ms": sum(ps_times) * 1000,
                "live_before_rescue": remaining,
                "unrelated_sibling_alive": sibling_alive,
            }
        finally:
            subprocess.run = original_run
            if monitor is not None:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Restrict signals to exact test-owned PIDs whose records still exist.
            known = [json.loads(p.read_text())["pid"] for d in directories for p in d.glob("*.json")]
            for pid in live(known):
                try:
                    os.kill(pid, signal.SIGKILL)
                    rescue.append(pid)
                except ProcessLookupError:
                    pass
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
            if sibling.returncode is None:
                sibling.kill()
            await asyncio.wait_for(sibling.wait(), 5)
            await asyncio.sleep(0.05)
        record["rescue_signals"] = rescue
        return record


async def main(args) -> int:
    from uni_agent.sandbox import local

    output = {
        "platform": platform.platform(),
        "python": sys.version,
        "source": local.__file__,
        "cases": [],
        "note": "Local Python fixtures, not OpenCode throughput or complete containment. Delay is injected.",
    }
    for delay in args.ps_delay_ms:
        for group in args.groups:
            detached = group == "detached"
            for count in args.counts:
                for round_number in range(args.rounds):
                    record = await run_case(count, detached, delay, round_number)
                    output["cases"].append(record)
                    print(json.dumps(record), flush=True)
                    args.output.write_text(json.dumps(output, indent=2) + "\n")
    return int(
        any(c["live_before_rescue"] or c["rescue_signals"] or not c["unrelated_sibling_alive"] for c in output["cases"])
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), sys.argv[3], bool(int(sys.argv[4])))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--counts", nargs="+", type=int, default=[1, 8, 32])
        parser.add_argument("--rounds", type=int, default=3)
        parser.add_argument("--groups", nargs="+", choices=["same", "detached"], default=["same", "detached"])
        parser.add_argument("--ps-delay-ms", nargs="+", type=float, default=[0, 20])
        parser.add_argument("--output", type=Path, required=True)
        arguments = parser.parse_args()
        if min(arguments.counts) < 1 or arguments.rounds < 1 or min(arguments.ps_delay_ms) < 0:
            parser.error("counts/rounds must be positive and delays non-negative")
        raise SystemExit(asyncio.run(main(arguments)))
