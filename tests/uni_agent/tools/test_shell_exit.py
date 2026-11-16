import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from uni_agent.sandbox import LocalSandbox
from uni_agent.tools.base import Toolbox
from uni_agent.tools.shell import ShellTool, TmuxShell

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.level0,
    pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux required"),
]


@pytest_asyncio.fixture
async def shell() -> AsyncIterator[TmuxShell]:
    channel = TmuxShell(LocalSandbox(), env={"UNI_AGENT_EXIT_TEST_ENV": "configured"})
    await channel.start()
    try:
        # Match providers where the private server exits with its last pane,
        # independent of this machine's tmux configuration.
        result = await channel.backend.exec(channel._tmux("set-window-option", "remain-on-exit", "off"))
        assert result.exit_code == 0
        assert (await channel.run("true")).exit_code == 0
        yield channel
    finally:
        await channel.close()


@pytest.mark.asyncio
async def test_normal_commands_keep_cwd_exports_and_existing_exit_trap(shell: TmuxShell, tmp_path: Path):
    result = await shell.run(f"cd {tmp_path}; export UNI_AGENT_EXIT_TEST_VALUE=persistent")
    assert (result.exit_code, result.timed_out) == (0, False)
    result = await shell.run('printf "%s\\n%s\\n" "$PWD" "$UNI_AGENT_EXIT_TEST_VALUE"')
    assert result.stdout.splitlines() == [str(tmp_path), "persistent"]
    await shell.run("trap 'printf prior' EXIT")
    result = await shell.run("trap -p EXIT; printf ordinary")
    assert "printf prior" in result.stdout and result.stdout.endswith("ordinary")
    result = await shell.run("trap -p EXIT")
    assert "printf prior" in result.stdout


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.asyncio
async def test_exit_returns_actual_result_and_next_command_starts_new_shell(
    shell: TmuxShell, tmp_path: Path, exit_code: int
):
    before = await shell.run(f'cd {tmp_path}; export UNI_AGENT_EXIT_TEST_VALUE=old; printf "%s" "$$"')
    result = await shell.run(f"printf before-exit; printf before-exit-error >&2; exit {exit_code}", timeout=10)
    assert (result.exit_code, result.stdout, result.timed_out) == (exit_code, "before-exit", False)
    # Interactive bash may append its own "exit" diagnostic; retain it too.
    assert result.stderr.startswith("before-exit-error")
    assert result.duration < 5
    after = await shell.run(
        'printf "%s\\n%s\\n%s\\n" "$$" "$UNI_AGENT_EXIT_TEST_ENV" "${UNI_AGENT_EXIT_TEST_VALUE-unset}"'
    )
    assert after.exit_code == 0 and after.command_id == result.command_id + 1
    pid, configured, value = after.stdout.splitlines()
    assert pid != before.stdout and configured == "configured" and value == "unset"
    assert not shell._closed


@pytest.mark.asyncio
async def test_timeout_still_interrupts_command_without_restarting_shell(shell: TmuxShell):
    before = await shell.run('printf "%s" "$$"')
    result = await shell.run("sleep 30", timeout=0.2)
    assert result.timed_out is True
    assert not shell._closed
    after = await shell.run('printf "%s" "$$"')
    assert after.stdout == before.stdout and after.exit_code == 0


@pytest.mark.asyncio
async def test_large_command_executes_through_same_capture_protocol(shell: TmuxShell):
    result = await shell.run("# long input\n" * 20_000 + "printf complete", timeout=10)
    assert (result.exit_code, result.stdout, result.timed_out) == (0, "complete", False)


@pytest.mark.asyncio
async def test_nonzero_command_and_child_exit_keep_parent_shell(shell: TmuxShell):
    before = await shell.run('printf "%s" "$$"')
    result = await shell.run("false")
    assert result.exit_code == 1 and not shell._closed
    result = await shell.run('(exit 7); printf "%s" "$?"')
    assert result.exit_code == 0 and result.stdout == "7" and not shell._closed
    after = await shell.run('printf "%s" "$$"')
    assert after.stdout == before.stdout


@pytest.mark.parametrize("command", ["kill -KILL $$", "trap - EXIT; exit 7", "exec bash -c 'exit 7'"])
@pytest.mark.asyncio
async def test_missing_exit_capture_does_not_infer_success_or_restart(shell: TmuxShell, command: str):
    with pytest.raises(RuntimeError, match="send_keys failed"):
        await shell.run(command, timeout=1)
    assert not shell._closed
    with pytest.raises(RuntimeError, match="failed to inject command"):
        await shell.run("printf unexpected-recovery", timeout=0.2)


@pytest.mark.asyncio
async def test_user_exit_trap_is_not_overridden_or_claimed_as_captured(shell: TmuxShell, tmp_path: Path):
    output = tmp_path / "user-exit-trap"
    await shell.run(f"trap 'printf ran > {output}' EXIT")
    with pytest.raises(RuntimeError, match="send_keys failed"):
        await shell.run("exit 7", timeout=1)
    assert output.read_text() == "ran" and not shell._closed


@pytest.mark.asyncio
async def test_retained_dead_pane_without_capture_remains_unknown(shell: TmuxShell):
    assert (await shell.backend.exec(shell._tmux("set-window-option", "remain-on-exit", "on"))).exit_code == 0
    result = await shell.run("kill -KILL $$", timeout=0.2)
    assert result.timed_out and result.exit_code is None and not shell._closed
    after = await shell.run("printf unexpected-recovery", timeout=0.2)
    assert after.timed_out and after.exit_code is None and after.stdout == "" and not shell._closed


@pytest.mark.asyncio
async def test_toolbox_observes_actual_exit_and_continues_next_call():
    toolbox = Toolbox([ShellTool(LocalSandbox(), env_vars={"UNI_AGENT_EXIT_TEST_ENV": "configured"})])
    try:
        result = await toolbox.call("shell", {"command": "printf final-output; exit 7"}, timeout=10)
        assert result.status == "ok" and result.text is not None
        assert "[exit code: 7]" in result.text and "final-output" in result.text
        result = await toolbox.call("shell", {"command": 'printf "%s" "$UNI_AGENT_EXIT_TEST_ENV"'}, timeout=10)
        assert result.status == "ok" and result.text is not None
        assert "[exit code: 0]" in result.text and "configured" in result.text
    finally:
        await toolbox.close()
