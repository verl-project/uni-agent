"""Black-box process tests for the sandbox runtime-state oracle."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PROBE = Path(__file__).resolve().parents[1] / "scripts" / "agentenv_offload" / "guest_probe.py"


class RunningProbe:
    def __init__(self, root: Path, *, entropy: str = "low", dirty_ratio: str = "0") -> None:
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(PROBE),
                "--memory-mib",
                "2",
                "--port",
                "0",
                "--host",
                "127.0.0.1",
                "--workdir",
                str(root),
                "--seed",
                "19",
                "--entropy",
                entropy,
                "--dirty-ratio",
                dirty_ratio,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=lambda: lines.put(self.process.stdout.readline()), daemon=True).start()
        try:
            self.ready_line = lines.get(timeout=15)
            self.ready = json.loads(self.ready_line)
        except Exception:
            self.process.kill()
            self.process.communicate(timeout=5)
            raise
        self.url = f"http://127.0.0.1:{self.ready['port']}"

    def request(self, path: str, body: dict | None = None) -> tuple[int, dict]:
        request = Request(
            self.url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, json.load(response)
        except HTTPError as response:
            return response.code, json.load(response)

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.request("/shutdown", {})
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process.communicate(timeout=5)


class ProbeProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="offload-probe-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.probes: list[RunningProbe] = []

    def start(self, **kwargs: str) -> RunningProbe:
        probe = RunningProbe(self.root, **kwargs)
        self.probes.append(probe)
        self.addCleanup(probe.close)
        return probe

    def test_observation_is_read_only_and_nonce_is_not_written(self) -> None:
        probe = self.start()
        status, initial = probe.request("/state")
        self.assertEqual(status, 200)
        self.assertEqual(initial["memory_bytes"], 2 * 1024 * 1024)
        self.assertEqual(initial["pid"], probe.process.pid)
        self.assertEqual(len(initial["nonce"]), 64)
        expected_memory = bytes.fromhex(initial["nonce"]) + bytes([19]) * (initial["memory_bytes"] - 32)
        self.assertEqual(initial["memory_sha256"], hashlib.sha256(expected_memory).hexdigest())
        self.assertEqual(probe.request("/health")[0], 200)
        self.assertEqual(probe.request("/state")[1], initial)
        self.assertNotIn(initial["nonce"], probe.ready_line)
        for file in self.root.rglob("*"):
            if file.is_file():
                data = file.read_bytes()
                self.assertNotIn(initial["nonce"].encode(), data)
                self.assertNotIn(bytes.fromhex(initial["nonce"]), data)

    def test_tool_changes_file_and_fd_but_observation_does_not(self) -> None:
        probe = self.start(dirty_ratio="0.25", entropy="high")
        before = probe.request("/state")[1]
        status, result = probe.request("/tool", {"command_id": "step-1", "read_bytes": 23, "payload": "change"})
        self.assertEqual(status, 200)
        after = result["state"]
        self.assertEqual(after["nonce"], before["nonce"])
        self.assertEqual(after["pid"], before["pid"])
        self.assertEqual(after["process_started_ns"], before["process_started_ns"])
        self.assertEqual(after["counter"], 1)
        self.assertEqual(after["fd_offset"], before["fd_offset"] + 23)
        self.assertEqual(result["read_hex"], bytes(range(7, 30)).hex())
        self.assertEqual(after["fd_sha256"], before["fd_sha256"])
        self.assertNotEqual(after["journal_sha256"], before["journal_sha256"])
        self.assertNotEqual(after["memory_sha256"], before["memory_sha256"])
        self.assertGreater(result["dirty_bytes"], 0)
        self.assertEqual(probe.request("/state")[1], after)
        lines = (Path(after["owned_dir"]) / "tool-journal.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(line)["counter"] for line in lines], [1])

    def test_concurrent_duplicate_command_has_one_effect_and_conflicts_are_rejected(self) -> None:
        probe = self.start()
        command = {"command_id": "repeat", "read_bytes": 11, "payload": "once"}
        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(workers.map(lambda _: probe.request("/tool", command), range(4)))
        self.assertTrue(all(status == 200 for status, _ in results))
        self.assertEqual(sum(not result["duplicate"] for _, result in results), 1)
        state = probe.request("/state")[1]
        self.assertEqual(state["counter"], 1)
        self.assertEqual(state["fd_offset"], 18)
        self.assertEqual(probe.request("/tool", {**command, "payload": "different"})[0], 409)
        self.assertEqual(probe.request("/state")[1], state)
        self.assertEqual(len((Path(state["owned_dir"]) / "tool-journal.jsonl").read_text().splitlines()), 1)

    def test_restart_is_detectable_and_shutdown_affects_only_one_probe(self) -> None:
        first = self.start()
        second = self.start()
        first_state = first.request("/state")[1]
        second_state = second.request("/state")[1]
        self.assertNotEqual(first_state["nonce"], second_state["nonce"])
        self.assertNotEqual(first_state["memory_sha256"], second_state["memory_sha256"])
        self.assertNotEqual(first_state["owned_dir"], second_state["owned_dir"])
        self.assertEqual(first.request("/shutdown", {})[0], 200)
        first.process.wait(timeout=5)
        self.assertEqual(first.process.returncode, 0)
        self.assertEqual(second.request("/state")[1], second_state)
        self.assertTrue(Path(first_state["owned_dir"]).exists())

    def test_invalid_tool_has_no_effect(self) -> None:
        probe = self.start()
        before = probe.request("/state")[1]
        for command in ({}, {"command_id": "x", "read_bytes": -1}, {"command_id": "x", "read_bytes": True}):
            self.assertEqual(probe.request("/tool", command)[0], 400)
        self.assertEqual(probe.request("/state")[1], before)


if __name__ == "__main__":
    unittest.main()
