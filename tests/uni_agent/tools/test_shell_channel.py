import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from uni_agent.sandbox import ExecResult
from uni_agent.tools.shell import _APT_LOCK_WAIT, TmuxShell


class RecordingBackend:
    def __init__(self) -> None:
        self.exec_calls: list[list[str]] = []
        self.writes: list[tuple[str, bytes | str]] = []

    async def exec(self, argv: list[str], **kwargs: Any) -> ExecResult:
        self.exec_calls.append(list(argv))
        return ExecResult(exit_code=0, stdout="", stderr="")

    async def write_file(self, path: str, content: bytes | str) -> None:
        self.writes.append((path, content))


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_start_command_keeps_large_payload_out_of_tmux_argv():
    backend = RecordingBackend()
    channel = TmuxShell(backend, session_id="test-session")  # type: ignore[arg-type]
    command = "printf 'large payload\\n'\n" * 10_000

    command_id = await channel.start_command(command)

    command_path = "/tmp/uni-agent-shell/test-session/cmd_1.input"
    assert command_id == 1
    assert backend.writes == [(command_path, command)]
    assert len(backend.exec_calls) == 1

    send_keys_argv = backend.exec_calls[0]
    injected_line = send_keys_argv[-2]
    assert send_keys_argv[-1] == "Enter"
    assert f'eval "$(cat {command_path})"' in injected_line
    assert "large payload" not in injected_line
    assert len(injected_line) < 1_000


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_interrupt_returns_when_ctrl_c_finishes_command(monkeypatch: pytest.MonkeyPatch):
    backend = RecordingBackend()
    channel = TmuxShell(backend, session_id="test-session")  # type: ignore[arg-type]

    async def sleep(_seconds: float) -> None:
        return None

    async def poll(_command_id: int) -> int:
        return 130

    monkeypatch.setattr("uni_agent.tools.shell.asyncio.sleep", sleep)
    monkeypatch.setattr(channel, "poll", poll)

    code = await channel.interrupt(1)

    assert code == 130
    send_keys_calls = [call for call in backend.exec_calls if "send-keys" in call]
    assert [call[-1] for call in send_keys_calls] == ["C-c"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_interrupt_suspends_and_kills_job_when_ctrl_c_fails(monkeypatch: pytest.MonkeyPatch):
    backend = RecordingBackend()
    channel = TmuxShell(backend, session_id="test-session")  # type: ignore[arg-type]
    poll_results = iter([None, 124])

    async def sleep(_seconds: float) -> None:
        return None

    async def poll(_command_id: int) -> int | None:
        return next(poll_results)

    monkeypatch.setattr("uni_agent.tools.shell.asyncio.sleep", sleep)
    monkeypatch.setattr(channel, "poll", poll)

    code = await channel.interrupt(1)

    assert code == 124
    send_keys_calls = [call for call in backend.exec_calls if "send-keys" in call]
    assert send_keys_calls[0][-1] == "C-c"
    assert send_keys_calls[1][-1] == "C-z"
    assert send_keys_calls[2][-1] == "Enter"
    fallback_line = send_keys_calls[2][-2]
    assert "kill -KILL %+" in fallback_line


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_run_marks_timeout_even_when_interrupt_reports_an_exit_code(
    monkeypatch: pytest.MonkeyPatch,
):
    backend = RecordingBackend()
    channel = TmuxShell(backend, session_id="test-session")  # type: ignore[arg-type]

    async def start_command(_command: str) -> int:
        return 1

    async def poll(_command_id: int) -> None:
        return None

    async def interrupt(_command_id: int) -> int:
        return 130

    monkeypatch.setattr(channel, "start_command", start_command)
    monkeypatch.setattr(channel, "poll", poll)
    monkeypatch.setattr(channel, "interrupt", interrupt)

    result = await channel.run("sleep forever", timeout=0.0)

    assert result.timed_out is True
    assert result.exit_code == 130


def _real_installer_running() -> bool:
    """Whether an unrelated live apt/dpkg process is visible on the host."""
    if sys.platform != "linux":
        return False
    for comm_path in Path("/proc").glob("[0-9]*/comm"):
        try:
            name = comm_path.read_text().strip()
            if name not in {"apt", "apt-get", "dpkg", "unattended-upgr"}:
                continue
            state = (comm_path.parent / "stat").read_text().split()
            if len(state) >= 3 and state[2] != "Z":
                return True
        except OSError:
            continue
    return False


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.skipif(sys.platform != "linux", reason="requires /proc comm inspection")
@pytest.mark.parametrize("fake_name", ["apt-get", "unattended-upgrades"])
def test_apt_lock_wait_blocks_until_existing_installer_exits(fake_name):
    if _real_installer_running():
        pytest.skip("unrelated apt/dpkg installer active on host")
    # PR_SET_NAME (=15) writes comm directly and the kernel truncates it to
    # 15 characters (unattended-upgrades -> unattended-upgr). Setting the name
    # explicitly is more portable than relying on symlink/exec filename
    # semantics, which vary across kernels and container runtimes.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import ctypes, time; ctypes.CDLL(None).prctl(15, b'{fake_name}', 0, 0, 0); time.sleep(3)",
        ]
    )
    try:
        start = time.monotonic()
        result = subprocess.run(
            ["bash", "-c", _APT_LOCK_WAIT + "; echo waited"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        elapsed = time.monotonic() - start
    finally:
        proc.kill()
        proc.wait()
    assert result.returncode == 0
    assert result.stdout.strip() == "waited"
    # The wait blocked until the fake installer exited instead of racing its lock.
    assert elapsed >= 2.5


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.skipif(sys.platform != "linux", reason="requires /proc comm inspection")
def test_apt_lock_wait_returns_immediately_without_installers():
    if _real_installer_running():
        pytest.skip("unrelated apt/dpkg installer active on host")
    start = time.monotonic()
    result = subprocess.run(
        ["bash", "-c", _APT_LOCK_WAIT + "; echo waited"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    elapsed = time.monotonic() - start
    assert result.returncode == 0
    assert result.stdout.strip() == "waited"
    assert elapsed < 3
