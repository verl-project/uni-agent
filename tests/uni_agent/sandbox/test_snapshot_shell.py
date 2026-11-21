import asyncio
import time
from pathlib import Path

import pytest
import pytest_asyncio

from uni_agent.sandbox.base import ExecResult
from uni_agent.sandbox.snapshot_shell import ShellTimeoutError, SnapshotShell, SnapshotStateError
from uni_agent.tools.shell import SandboxShell

pytestmark = [pytest.mark.cpu, pytest.mark.level0, pytest.mark.asyncio]


class LocalBackend:
    def __init__(self):
        self.calls = 0
        self.alive = True
        # "timeout": the RPC is lost before dispatch; "detach": the runner keeps
        # going remotely while the caller sees a transport timeout; "lost": the
        # command completes but its reply is lost; any exception instance is raised as-is.
        self.fault = None
        self.detached = []

    async def _exec(self, argv, *, timeout=None):
        self.calls += 1
        fault, self.fault = self.fault, None
        if fault == "timeout":
            raise TimeoutError("transport deadline")
        if isinstance(fault, BaseException):
            raise fault
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if fault == "detach":
            self.detached.append(asyncio.create_task(proc.communicate()))
            raise TimeoutError("transport deadline")
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
        except BaseException:
            proc.kill()
            await proc.wait()
            raise
        if fault == "lost":
            raise TimeoutError("reply lost")
        return ExecResult(proc.returncode, stdout.decode(), stderr.decode())

    async def write_file(self, path, content):
        Path(path).write_text(content)

    async def is_alive(self):
        return self.alive

    def _is_timeout_error(self, exc):
        return isinstance(exc, TimeoutError)


@pytest_asyncio.fixture
async def opened(tmp_path):
    backend = LocalBackend()
    shell = SnapshotShell(backend, env={"INHERITED": "base"}, cwd=str(tmp_path))
    await shell.open()
    try:
        yield backend, shell
    finally:
        await asyncio.gather(*backend.detached, return_exceptions=True)
        await shell.close()


async def test_exit_unset_unexport_and_path(opened, tmp_path):
    backend, shell = opened
    assert backend.calls == 1
    assert (await shell.run("pwd")).stdout.strip() == str(tmp_path)
    res = await shell.run("export VALUE=new; unset INHERITED; export -n HOME; export PATH=/missing; exit 7")
    assert res.exit_code == 7
    res = await shell.run('printf "%s|%s|%s|%s" "$VALUE" "${INHERITED-missing}" "${HOME-missing}" "$PATH"')
    assert res.stdout == "new|missing|missing|/missing"


async def test_concurrent_updates_and_close(opened):
    _, shell = opened
    first = asyncio.create_task(shell.run("sleep .1; export A=first"))
    await asyncio.sleep(0.03)
    second = asyncio.create_task(shell.run('export B="$A-second"'))
    await asyncio.gather(first, second)
    assert (await shell.run('printf %s "$B"')).stdout == "first-second"
    pending = asyncio.create_task(shell.run("sleep .1; true"))
    await asyncio.sleep(0.03)
    await shell.close()
    assert (await pending).exit_code == 0
    assert not Path(shell._root).exists()
    with pytest.raises(RuntimeError, match="closed"):
        await shell.run("true")


@pytest.mark.parametrize("command", ["exec /bin/true", "trap - EXIT; exit 3", "kill -9 $$"])
async def test_unsaved_snapshot_keeps_previous_state(opened, command):
    _, shell = opened
    await shell.run("export KEEP=kept")
    res = await shell.run(f"export LOST=1; cd /; {command}")
    assert "state not saved" in res.stderr
    res = await shell.run('printf "%s|%s|%s" "$KEEP" "${LOST-unset}" "$PWD"')
    assert res.stdout.split("|")[:2] == ["kept", "unset"] and res.stdout.split("|")[2] != "/"


async def test_user_exit_trap_runs_and_state_is_saved(opened):
    _, shell = opened
    res = await shell.run("trap 'echo bye' EXIT; export T=1")
    assert res.stdout == "bye\n" and "not saved" not in res.stderr
    assert (await shell.run('printf %s "$T"')).stdout == "1"


async def test_background_job_and_stdin_do_not_block(opened):
    _, shell = opened
    start = time.monotonic()
    res = await shell.run("sleep 3 & echo started")
    assert res.stdout == "started\n" and time.monotonic() - start < 2
    res = await shell.run("cat; read -r line; echo rc=$?", timeout=5)
    assert res.stdout == "rc=1\n"


async def test_output_is_capped_in_sandbox(tmp_path):
    backend = LocalBackend()
    shell = SnapshotShell(backend, cwd=str(tmp_path), output_limit=1000)
    res = await shell.run("head -c 5000 /dev/zero | tr '\\0' a; head -c 3000 /dev/zero | tr '\\0' b >&2")
    assert res.stdout.startswith("a" * 1000) and "first 1000 of 5000 bytes" in res.stdout
    assert res.stderr.startswith("b" * 1000) and "first 1000 of 3000 bytes" in res.stderr
    await shell.close()


async def test_missing_cwd_skips_command_then_falls_back_home(opened, tmp_path):
    _, shell = opened
    work = tmp_path / "work"
    work.mkdir()
    await shell.run(f"cd {work}; export KEEP=kept")
    work.rmdir()
    with pytest.raises(SnapshotStateError, match="not executed") as exc:
        await shell.run("printf must-not-run")
    assert not exc.value.result.stdout
    res = await shell.run('printf "%s|%s" "$PWD" "$KEEP"')
    assert res.stdout == f"{Path.home()}|kept"
    work.mkdir()
    await shell.run(f"cd {work}")
    work.rmdir()
    with pytest.raises(SnapshotStateError):
        await shell.run("true")


async def test_missing_state_does_not_run_user_command(opened):
    _, shell = opened
    Path(shell._root, "state").unlink()
    with pytest.raises(SnapshotStateError) as exc:
        await shell.run("printf must-not-run")
    assert not exc.value.result.stdout
    assert (await shell.run("printf ran")).stdout == "ran"


async def test_timeout_preserves_partial_output_and_keeps_handle(opened):
    _, shell = opened
    await shell.run("export KEEP=kept")
    root = shell._root
    result = await SandboxShell(shell).run("printf before; printf err >&2; export KEEP=lost; sleep 5", timeout=0.1)
    assert result.timed_out and result.stdout == "before" and "err" in result.stderr
    assert (await shell.run('printf %s "$KEEP"')).stdout == "kept"
    assert shell._root == root


async def test_transport_timeout_fences_stale_runner(opened):
    backend, shell = opened
    await shell.run("export A=1")
    old_root = shell._root
    backend.fault = "detach"
    with pytest.raises(ShellTimeoutError, match="remote state unknown"):
        await shell.run("sleep .3; export B=1")
    res = await shell.run('printf "%s|%s" "$A" "${B-unset}"')
    assert res.stdout == "1|unset" and shell._root != old_root
    await asyncio.gather(*backend.detached)
    assert (await shell.run('printf %s "${B-unset}"')).stdout == "unset"
    assert not Path(old_root).exists() or not Path(old_root, "state").exists()


async def test_lost_rpc_is_a_timeout_and_state_survives(opened):
    backend, shell = opened
    await shell.run("export A=1")
    backend.fault = "timeout"
    result = await SandboxShell(shell).run("export B=1", timeout=5)
    assert result.timed_out
    assert (await shell.run('printf "%s|%s" "$A" "${B-unset}"')).stdout == "1|unset"


async def test_cancellation_fences_and_next_command_recovers(opened):
    _, shell = opened
    await shell.run("export A=1")
    old_root = shell._root
    pending = asyncio.create_task(shell.run("sleep .2", timeout=0.3))
    await asyncio.sleep(0.05)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert (await shell.run('printf %s "$A"')).stdout == "1"
    assert shell._root != old_root and not Path(old_root).exists()


async def test_rotation_with_lost_reply_is_retried_without_losing_state(opened):
    backend, shell = opened
    await shell.run("export A=1")
    old_root = shell._root
    backend.fault = "timeout"
    with pytest.raises(ShellTimeoutError):
        await shell.run("true")
    backend.fault = "lost"
    with pytest.raises(ShellTimeoutError):
        await shell.run("true")
    res = await shell.run('printf %s "${A-unset}"')
    assert res.stdout == "1" and "reinitialized" not in res.stderr
    leftovers = [old_root + ".dead", *shell._stale_roots]
    await shell.close()
    assert not any(Path(p).exists() for p in leftovers)


async def test_lost_state_is_reinitialized_with_notice(opened, tmp_path):
    backend, shell = opened
    await shell.run("export A=1; cd /")
    Path(shell._root, "state").unlink()
    backend.fault = "timeout"
    with pytest.raises(ShellTimeoutError):
        await shell.run("true")
    res = await shell.run('printf "%s|%s|%s" "${A-unset}" "$INHERITED" "$PWD"')
    assert res.stdout == f"unset|base|{tmp_path}" and "reinitialized" in res.stderr
    assert "reinitialized" not in (await shell.run("true")).stderr


async def test_transport_error_depends_on_liveness(opened):
    backend, shell = opened
    await shell.run("export A=1")
    backend.fault = ConnectionError("reset")
    with pytest.raises(SnapshotStateError, match="remote state unknown"):
        await shell.run("true")
    backend.fault = ConnectionError("reset during rotation")
    with pytest.raises(SnapshotStateError, match="rotation"):
        await shell.run("true")
    assert (await shell.run('printf %s "$A"')).stdout == "1"
    backend.fault, backend.alive = ConnectionError("gone"), False
    with pytest.raises(ConnectionError):
        await shell.run("true")
    with pytest.raises(RuntimeError, match="broken"):
        await shell.run("true")


async def test_large_payload_and_output_do_not_collide_with_control(opened):
    _, shell = opened
    value = "x" * 150_000
    result = await shell.run(f"printf '%s' '{value}'; printf '\\nUASH1 9 9 9 9 9 9\\n' >&2")
    assert result.stdout == value and result.stderr == "\nUASH1 9 9 9 9 9 9\n"
    assert result.exit_code == 0
    assert sorted(p.name for p in Path(shell._root).iterdir()) == ["runner.sh", "state"]


async def test_exported_rc_is_not_overwritten_by_protocol(opened):
    _, shell = opened
    await shell.run("export rc=original __uash_rc=user; exit 7")
    res = await shell.run('printf "%s|%s" "$rc" "$__uash_rc"')
    assert res.stdout == "original|user"


async def test_replaced_sandbox_rejects_old_handle(tmp_path):
    backend = LocalBackend()
    backend._runtime = object()
    shell = SnapshotShell(backend, cwd=str(tmp_path), generation_attr="_runtime")
    assert (await shell.run("printf ok")).stdout == "ok"
    backend._runtime = object()
    with pytest.raises(RuntimeError, match="replaced"):
        await shell.run("true")
    with pytest.raises(RuntimeError, match="broken"):
        await shell.run("true")
    await shell.close()
    assert not Path(shell._root).exists()


async def test_interactive_init_applies_bashrc_and_survives_rotation(tmp_path):
    (tmp_path / ".bashrc").write_text(
        "case $- in *i*) ;; *) return;; esac\nexport FROM_RC=1\ngreet() { echo \"hi $1\"; }\nalias ll='echo aliased'\n"
    )
    backend = LocalBackend()
    shell = SnapshotShell(
        backend, env={"HOME": str(tmp_path)}, cwd=str(tmp_path), interactive=True, prelude="export PRE=1"
    )
    probe = 'greet x; ll; printf "%s|%s|%s" "$FROM_RC" "$PRE" "$PWD"'
    assert (await shell.run(probe)).stdout == f"hi x\naliased\n1|1|{tmp_path}"
    await shell.run("export FROM_RC=2")
    backend.fault = "timeout"
    with pytest.raises(ShellTimeoutError):
        await shell.run("true")
    assert (await shell.run(probe)).stdout == f"hi x\naliased\n2|1|{tmp_path}"
    await shell.close()
