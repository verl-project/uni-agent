import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from uni_agent.sandbox.base import ExecResult
from uni_agent.tasks.r2e_gym import reward, task

PASSING_OUTPUT = "=== short test summary info ===\nPASSED r2e_tests/test_fix.py::test_fix\n"
METADATA = {"instance_id": "repo__abc", "expected_output_json": '{"test_fix": "PASSED"}'}
_ABSOLUTE_PATH = re.compile(r"(?<![\w./$-])/(root|testbed|tmp|usr|opt|r2e_tests|run_tests\.sh)(?=[/\s\"';]|$)")


def _in_fake_root(script: str, root: Path) -> str:
    """Rewrite the sandbox's absolute paths to live under ``root`` so the scripts run on the host."""
    return _ABSOLUTE_PATH.sub(lambda match: f"{root}/{match.group(1)}", script)


def _run(script: str, root: Path) -> subprocess.CompletedProcess:
    # HOME keeps the setup script's ``git config --global`` out of the host's git config.
    env = {"HOME": str(root), "PATH": os.environ["PATH"]}
    return subprocess.run(
        ["bash", "-c", _in_fake_root(script, root)], capture_output=True, text=True, check=False, env=env
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_r2e_reward_restores_held_out_tests_before_running_them():
    sandbox = SimpleNamespace(
        write_file=AsyncMock(),
        exec_shell=AsyncMock(
            side_effect=[ExecResult(exit_code=0, stdout="", stderr=""), ExecResult(0, PASSING_OUTPUT, "")]
        ),
    )

    result = await reward.compute_reward(METADATA, sandbox, eval_timeout=60.0, eval_assets=b"archive")

    sandbox.write_file.assert_awaited_once_with(reward.EVAL_ASSETS_PATH, b"archive")
    restore_call, test_call = sandbox.exec_shell.await_args_list
    assert restore_call.args == (reward._RESTORE_SCRIPT,)
    assert "bash /root/run_tests.sh" in test_call.args[0]
    assert result["resolved"] is True


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_r2e_reward_raises_when_restoring_held_out_tests_fails():
    sandbox = SimpleNamespace(
        write_file=AsyncMock(),
        exec_shell=AsyncMock(return_value=ExecResult(exit_code=1, stdout="", stderr="No space left on device")),
    )

    with pytest.raises(RuntimeError, match="No space left on device"):
        await reward.compute_reward(METADATA, sandbox, eval_timeout=60.0, eval_assets=b"archive")

    sandbox.exec_shell.assert_awaited_once()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.skipif(shutil.which("tar") is None, reason="needs tar")
def test_r2e_setup_hides_held_out_tests_and_restore_brings_them_back(tmp_path):
    (tmp_path / "tmp").mkdir()
    testbed = tmp_path / "testbed"
    (testbed / "r2e_tests").mkdir(parents=True)
    (testbed / "r2e_tests" / "test_fix.py").write_text("def test_fix(): pass\n")
    (tmp_path / "run_tests.sh").write_text("echo held-out runner\n")
    (tmp_path / "root").mkdir()
    # Some images ship /root/run_tests.sh as a symlink to /run_tests.sh, which the setup removes.
    (tmp_path / "root" / "run_tests.sh").symlink_to(tmp_path / "run_tests.sh")

    setup = _run(task._SETUP_SCRIPT, tmp_path)
    assert setup.returncode == 0, setup.stderr

    archive = tmp_path / "tmp" / Path(reward.EVAL_ASSETS_PATH).name
    assert archive.is_file()
    for exposed in ("run_tests.sh", "root/run_tests.sh", "testbed/r2e_tests"):
        assert not (tmp_path / exposed).exists() and not (tmp_path / exposed).is_symlink()

    # The agent tampers with the workspace copy; the restored copy comes from the archive.
    (testbed / "r2e_tests").mkdir()
    (testbed / "r2e_tests" / "test_fix.py").write_text("def test_fix(): assert True  # tampered\n")

    restore = _run(reward._RESTORE_SCRIPT, tmp_path)
    assert restore.returncode == 0, restore.stderr

    runner = tmp_path / "root" / "run_tests.sh"
    assert runner.is_file() and not runner.is_symlink()
    assert runner.read_text() == "echo held-out runner\n"
    assert (testbed / "r2e_tests").resolve() == (tmp_path / "root" / "r2e_tests").resolve()
    assert (testbed / "r2e_tests" / "test_fix.py").read_text() == "def test_fix(): pass\n"
    assert not archive.exists()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.skipif(shutil.which("tar") is None, reason="needs tar")
def test_r2e_setup_removes_symlink_targets_outside_testbed(tmp_path):
    (tmp_path / "tmp").mkdir()
    (tmp_path / "testbed").mkdir()
    (tmp_path / "testbed" / "conftest.py").write_text("# repository file\n")
    (tmp_path / "run_tests.sh").write_text("echo held-out runner\n")
    held_out = tmp_path / "opt" / "r2e_tests"
    held_out.mkdir(parents=True)
    (held_out / "test_fix.py").write_text("def test_fix(): pass\n")
    (tmp_path / "opt" / "helper.py").write_text("HELPER = 1\n")
    (held_out / "helper.py").symlink_to(tmp_path / "opt" / "helper.py")
    (held_out / "conftest.py").symlink_to(tmp_path / "testbed" / "conftest.py")
    (tmp_path / "usr" / "lib").mkdir(parents=True)
    (tmp_path / "usr" / "lib" / "plugin.py").write_text("# runtime environment\n")
    (held_out / "plugin.py").symlink_to(tmp_path / "usr" / "lib" / "plugin.py")
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "r2e_tests").symlink_to(held_out)

    setup = _run(task._SETUP_SCRIPT, tmp_path)
    assert setup.returncode == 0, setup.stderr

    assert not held_out.exists()
    assert not (tmp_path / "opt" / "helper.py").exists()
    assert not (tmp_path / "root" / "r2e_tests").is_symlink()
    assert (tmp_path / "testbed" / "conftest.py").read_text() == "# repository file\n"
    assert (tmp_path / "usr" / "lib" / "plugin.py").read_text() == "# runtime environment\n"

    restore = _run(reward._RESTORE_SCRIPT, tmp_path)
    assert restore.returncode == 0, restore.stderr
    restored = tmp_path / "root" / "r2e_tests"
    assert (restored / "test_fix.py").read_text() == "def test_fix(): pass\n"
    assert (restored / "helper.py").read_text() == "HELPER = 1\n"


def _git(testbed: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "-C", str(testbed), *args],
        check=True,
        capture_output=True,
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.skipif(shutil.which("tar") is None, reason="needs tar")
def test_r2e_setup_follows_symlinks_inside_symlinked_directories(tmp_path):
    (tmp_path / "tmp").mkdir()
    (tmp_path / "testbed").mkdir()
    (tmp_path / "run_tests.sh").write_text("echo held-out runner\n")
    held_out = tmp_path / "opt" / "r2e_tests"
    held_out.mkdir(parents=True)
    (held_out / "test_fix.py").write_text("def test_fix(): pass\n")
    group = tmp_path / "opt" / "group"
    group.mkdir()
    secret = tmp_path / "srv" / "held_out.py"
    secret.parent.mkdir()
    secret.write_text("SECRET = 1\n")
    # r2e_tests/fixtures -> /opt/group, and /opt/group/data.py -> /srv/held_out.py: two levels of links.
    (group / "data.py").symlink_to(secret)
    (held_out / "fixtures").symlink_to(group)
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "r2e_tests").symlink_to(held_out)

    setup = _run(task._SETUP_SCRIPT, tmp_path)
    assert setup.returncode == 0, setup.stderr

    assert not held_out.exists()
    assert not group.exists()
    assert not secret.exists()

    restore = _run(reward._RESTORE_SCRIPT, tmp_path)
    assert restore.returncode == 0, restore.stderr
    assert (tmp_path / "root" / "r2e_tests" / "fixtures" / "data.py").read_text() == "SECRET = 1\n"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.skipif(shutil.which("tar") is None or shutil.which("git") is None, reason="needs tar and git")
def test_r2e_setup_removes_untracked_symlink_targets_inside_testbed(tmp_path):
    (tmp_path / "tmp").mkdir()
    testbed = tmp_path / "testbed"
    testbed.mkdir()
    (testbed / "module.py").write_text("VALUE = 1\n")
    _git(testbed, "init", "-q")
    _git(testbed, "add", "module.py")
    _git(testbed, "commit", "-q", "-m", "init")
    (tmp_path / "run_tests.sh").write_text("echo held-out runner\n")
    held_out = testbed / "held_out"
    held_out.mkdir()
    (held_out / "test_fix.py").write_text("def test_fix(): pass\n")
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "r2e_tests").symlink_to(held_out)

    setup = _run(task._SETUP_SCRIPT, tmp_path)
    assert setup.returncode == 0, setup.stderr

    assert not held_out.exists()
    assert (testbed / "module.py").read_text() == "VALUE = 1\n"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.skipif(shutil.which("tar") is None or shutil.which("git") is None, reason="needs tar and git")
def test_r2e_setup_fails_when_held_out_tests_resolve_to_tracked_files(tmp_path):
    (tmp_path / "tmp").mkdir()
    testbed = tmp_path / "testbed"
    (testbed / "suite").mkdir(parents=True)
    (testbed / "suite" / "test_fix.py").write_text("def test_fix(): pass\n")
    _git(testbed, "init", "-q")
    _git(testbed, "add", "suite")
    _git(testbed, "commit", "-q", "-m", "init")
    (tmp_path / "run_tests.sh").write_text("echo held-out runner\n")
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "r2e_tests").symlink_to(testbed / "suite")

    setup = _run(task._SETUP_SCRIPT, tmp_path)

    assert setup.returncode != 0
    assert "cannot hide held-out R2E-Gym tests" in setup.stderr
    assert (testbed / "suite" / "test_fix.py").read_text() == "def test_fix(): pass\n"
