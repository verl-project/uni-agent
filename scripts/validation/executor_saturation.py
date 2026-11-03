"""Bounded default-executor saturation probes with real detached process trees."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from scripts.validation import local_cleanup_burst as fixture
from uni_agent.sandbox import local


def rescue(pids):
    sent = []
    for pid in fixture.live(pids):
        try:
            os.kill(pid, signal.SIGKILL)
            sent.append(pid)
        except ProcessLookupError:
            pass
    return sent


async def case(root, workers, count, hold_ms, mode, sibling):
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ua-saturation")
    loop.set_default_executor(pool)
    release = threading.Event()
    entered = [threading.Event() for _ in range(workers)]
    spawn_gate, timeout_gate = asyncio.Event(), asyncio.Event()
    tasks, pids, spawned, sandboxes = [], [], [], []
    original_create = asyncio.create_subprocess_exec
    original_wait = asyncio.wait_for
    blocker_futures = []
    heartbeat = None
    timer = None
    record = {}

    def occupy(index):
        entered[index].set()
        if not release.wait(10):
            raise AssertionError("test blocker safety deadline expired")

    async def create(*args, **kwargs):
        proc = await original_create(*args, **kwargs)
        spawned.append(proc)
        if mode in {"spawn-cancel", "shutdown"}:
            await spawn_gate.wait()
        return proc

    async def wait_for(awaitable, timeout):
        if timeout == 0.1:
            await timeout_gate.wait()
        return await original_wait(awaitable, timeout)

    async def beats():
        while True:
            record["heartbeat_ticks"] += 1
            await asyncio.sleep(0.01)

    record["sibling_pid"] = sibling.pid
    try:
        blocker_futures = [loop.run_in_executor(None, occupy, i) for i in range(workers)]
        deadline = loop.time() + 3
        while not all(x.is_set() for x in entered):
            assert loop.time() < deadline
            await asyncio.sleep(0.01)
        with (
            patch.object(local.asyncio, "create_subprocess_exec", create),
            patch.object(local.asyncio, "wait_for", wait_for),
        ):
            for i in range(count):
                directory = root / str(i)
                directory.mkdir()
                sandbox = local.LocalSandbox()
                sandboxes.append(sandbox)
                tasks.append(
                    asyncio.create_task(
                        sandbox.exec(
                            [sys.executable, fixture.__file__, "--worker", str(directory), "parent", "1"],
                            timeout=0.1 if mode == "timeout" else 30,
                        )
                    )
                )
            deadline = loop.time() + 5
            while not all((root / str(i) / "child.json").exists() for i in range(count)):
                assert loop.time() < deadline
                await asyncio.sleep(0.01)
            pids = [json.loads(p.read_text())["pid"] for p in root.glob("*/*.json")]
            assert len(pids) == count * 2 and len(fixture.live(pids)) == len(pids)
            record.update(
                workers=workers,
                commands=count,
                hold_ms=hold_ms,
                mode=mode,
                target_pids=pids,
                heartbeat_ticks=0,
                deferred=mode in {"spawn-cancel", "shutdown"},
            )
            heartbeat = asyncio.create_task(beats())
            started = time.perf_counter()
            if mode == "timeout":
                timeout_gate.set()
            else:
                for task in tasks:
                    task.cancel()
            if mode in {"spawn-cancel", "shutdown"}:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                assert all(isinstance(r, asyncio.CancelledError) for r in results)
                spawn_gate.set()
            deadline = loop.time() + 3
            while len(local._cleanup_tasks) < count:
                assert loop.time() < deadline, "cleanup was not submitted while executor was occupied"
                await asyncio.sleep(0.01)
            assert all(not f.done() for f in local._cleanup_tasks)
            if mode == "repeat-cancel":
                for task in tasks:
                    task.cancel()
                await asyncio.sleep(0)
                for task in tasks:
                    task.cancel()
            await asyncio.sleep(hold_ms / 1000)
            record["queued_cleanup_count"] = len(local._cleanup_tasks)
            assert record["heartbeat_ticks"] >= 3
            assert all(not f.done() for f in local._cleanup_tasks)
            assert len(fixture.live(pids)) == len(pids), "targets must remain live until pool is released"
            if mode not in {"spawn-cancel", "shutdown"}:
                assert all(not t.done() for t in tasks), "caller abandoned pending cleanup"
            # Subprocess I/O and timers remain usable even though to_thread work cannot run.
            other = await original_create(sys.executable, "-c", "print('responsive')", stdout=subprocess.PIPE)
            out, _ = await original_wait(other.communicate(), 2)
            assert other.returncode == 0 and out.strip() == b"responsive"
            record["unrelated_loop_work_completed"] = True
            record["queued_ms"] = (time.perf_counter() - started) * 1000
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            if mode == "shutdown":
                # Release from another thread, not a timer on the loop being shut down.
                timer = threading.Timer(hold_ms / 1000, release.set)
                timer.start()
                record["returned_from_main_at"] = time.perf_counter()
                # Explicit owner teardown must drain transports before asyncio.run closes.
                await asyncio.gather(*(sandbox.stop() for sandbox in sandboxes))
                record["owner_stop_completed"] = True
                return record
            released_at = time.perf_counter()
            release.set()
            await asyncio.gather(*blocker_futures)
            results = await original_wait(asyncio.gather(*tasks, return_exceptions=True), 5)
            if mode == "timeout":
                assert all(getattr(r, "exit_code", None) == -1 for r in results)
            else:
                assert all(isinstance(r, asyncio.CancelledError) for r in results)
            while local._cleanup_tasks:
                await original_wait(asyncio.gather(*list(local._cleanup_tasks)), 5)
                await asyncio.sleep(0)
            await asyncio.gather(*(sandbox.stop() for sandbox in sandboxes))
            record["cleanup_after_release_ms"] = (time.perf_counter() - released_at) * 1000
            deadline = loop.time() + 3
            while fixture.live(pids) and loop.time() < deadline:
                await asyncio.sleep(0.02)
            record["live_before_rescue"] = fixture.live(pids)
            record["sibling_alive"] = bool(fixture.live([sibling.pid]))
            assert not record["live_before_rescue"] and record["sibling_alive"]
            return record
    finally:
        if mode != "shutdown" or timer is None:
            release.set()
            spawn_gate.set()
            timeout_gate.set()
            if heartbeat:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await original_wait(asyncio.gather(*tasks, return_exceptions=True), 5)
            if local._cleanup_tasks:
                await original_wait(asyncio.gather(*list(local._cleanup_tasks)), 5)
            # Cleanup only after the assertions above; record any emergency intervention.
            record["rescue_signals"] = rescue(pids)
            for proc in spawned:
                await original_wait(proc.wait(), 5)
            await asyncio.gather(*blocker_futures, return_exceptions=True)


def run_one(workers, count, hold_ms, mode):
    with tempfile.TemporaryDirectory(prefix="ua-executor-saturation-") as tmp:
        # The control process is owned outside asyncio and reaped before exit.
        sibling = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            record = asyncio.run(case(Path(tmp), workers, count, hold_ms, mode, sibling))
            if mode == "shutdown":
                record["shutdown_drain_ms"] = (time.perf_counter() - record.pop("returned_from_main_at")) * 1000
                record["live_before_rescue"] = fixture.live(record["target_pids"])
                record["sibling_alive"] = bool(fixture.live([sibling.pid]))
                record["rescue_signals"] = rescue(record["target_pids"])
                assert not record["live_before_rescue"] and record["sibling_alive"]
            record["pending_cleanup_futures_after_run"] = len(local._cleanup_tasks)
            assert not record["rescue_signals"]
            return record
        finally:
            if sibling.poll() is None:
                sibling.kill()
            sibling.wait(timeout=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", nargs=4)
    parser.add_argument("--require-clean-shutdown", action="store_true")
    args = parser.parse_args()
    if args.case:
        workers, count, hold_ms, mode = args.case
        row = run_one(int(workers), int(count), int(hold_ms), mode)
        gc.collect()
        print(json.dumps(row), flush=True)
        raise SystemExit(0)
    if args.output is None:
        parser.error("provide --output")
    records = []
    cases = [
        (w, n, ms, mode)
        for w in [1, 4]
        for n in [1, 8]
        for ms in [100, 500]
        for mode in ["cancel", "repeat-cancel", "timeout", "spawn-cancel"]
    ]
    cases += [(w, 8, ms, "shutdown") for w in [1, 4] for ms in [100, 500]]
    for parameters in cases:
        result = subprocess.run(
            [sys.executable, __file__, "--case", *map(str, parameters)], text=True, capture_output=True, timeout=30
        )
        assert result.returncode == 0, (parameters, result.stdout, result.stderr)
        row = json.loads(result.stdout)
        row["shutdown_diagnostics"] = result.stderr
        if row["mode"] != "shutdown" or args.require_clean_shutdown:
            assert not result.stderr, (parameters, result.stderr)
        row["observation"] = "transport_shutdown_warning" if result.stderr else "clean"
        records.append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
