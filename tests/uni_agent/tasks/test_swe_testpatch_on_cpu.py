import shlex
import subprocess
import sys

import pytest

from uni_agent.tasks.swe_bench import reward


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "cpu@example.invalid")
    git(tmp_path, "config", "user.name", "CPU test")
    (tmp_path / "test_modified.py").write_text("def test_existing():\n    assert True\n")
    (tmp_path / "candidate.py").write_text("original = True\n")
    git(tmp_path, "add", "--", "test_modified.py", "candidate.py")
    git(tmp_path, "commit", "-m", "base")
    return tmp_path, git(tmp_path, "rev-parse", "HEAD").strip()


def make_patch(repo):
    (repo / "test_modified.py").write_text("def test_existing():\n    assert 1 == 1\n")
    (repo / "test added.py").write_text("def test_official():\n    assert True\n")
    git(repo, "add", "--", "test_modified.py", "test added.py")
    patch = git(repo, "diff", "--cached")
    git(repo, "reset", "--hard", "HEAD")
    return patch


def run_eval(repo, base, patch, monkeypatch, test_files=("test added.py", "test_modified.py")):
    monkeypatch.setitem(
        reward.MAP_REPO_VERSION_TO_SPECS["psf/requests"],
        "cpu",
        {
            "test_cmd": shlex.join([sys.executable, "-m", "pytest", "-rA", *test_files]),
        },
    )
    monkeypatch.setattr(reward, "get_test_directives", lambda instance: [])
    commands = reward._make_eval_script_list(
        {"repo": "psf/requests", "version": "cpu", "base_commit": base},
        {},
        "testbed",
        str(repo),
        base,
        patch,
    )
    # Run the actual generated patch/test/reset section; sandbox activation is remote-only.
    start = next(
        i for i, command in enumerate(commands) if command.startswith("git checkout ") or command == "echo 'skip reset'"
    )
    script = "\n".join(["set -uxo pipefail", *commands[start:]])
    return subprocess.run(["bash", "-c", script], cwd=repo, text=True, capture_output=True)


@pytest.mark.parametrize("collision", ["untracked", "ignored", "staged", "committed"])
def test_added_collision_restores_official_tests_and_preserves_other_candidate_files(repo, monkeypatch, collision):
    path, base = repo
    patch = make_patch(path)
    (path / "test added.py").write_text('raise AssertionError("candidate collision")\n')
    (path / "test_modified.py").write_text('raise AssertionError("candidate modified test")\n')
    (path / "candidate.py").write_text("candidate_change = True\n")
    (path / "unrelated.txt").write_text("candidate untracked\n")
    if collision == "ignored":
        (path / ".gitignore").write_text("test added.py\nunrelated.txt\n")
    elif collision in ("staged", "committed"):
        git(path, "add", "--", "test added.py")
        if collision == "committed":
            git(path, "commit", "-m", "candidate collision")
    before = subprocess.run(["git", "apply", "-"], cwd=path, input=patch, text=True, capture_output=True)
    assert before.returncode != 0
    assert "already exists" in before.stderr

    result = run_eval(path, base, patch, monkeypatch)

    assert result.returncode == 0
    assert "PASSED test added.py::test_official" in result.stdout
    assert "PASSED test_modified.py::test_existing" in result.stdout
    assert reward.START_TEST_OUTPUT in result.stderr and reward.END_TEST_OUTPUT in result.stderr
    assert (path / "test_modified.py").read_text() == "def test_existing():\n    assert True\n"
    assert not (path / "test added.py").exists()
    assert (path / "candidate.py").read_text() == "candidate_change = True\n"
    assert (path / "unrelated.txt").read_text() == "candidate untracked\n"


def test_apply_failure_stops_before_formal_test_markers(repo, monkeypatch):
    path, base = repo
    patch = make_patch(path).replace("assert True", "assert absent_base_context", 1)
    # Corrupt the modified-file context while retaining a valid unified patch.
    patch = patch.replace("-    assert True", "-    assert missing_context")
    result = run_eval(path, base, patch, monkeypatch)
    assert result.returncode != 0
    assert "patch does not apply" in result.stderr
    assert reward.START_TEST_OUTPUT not in result.stderr
    assert reward.END_TEST_OUTPUT not in result.stderr
    assert "test session starts" not in result.stdout


def test_legitimate_test_failure_keeps_end_marker_and_final_reset(repo, monkeypatch):
    path, base = repo
    patch = make_patch(path).replace("+    assert True", "+    assert False")
    result = run_eval(path, base, patch, monkeypatch)
    assert result.returncode == 0  # Existing shell contract: final test reset succeeds.
    assert "FAILED test added.py::test_official" in result.stdout
    assert reward.START_TEST_OUTPUT in result.stderr and reward.END_TEST_OUTPUT in result.stderr
    assert (path / "test_modified.py").read_text() == "def test_existing():\n    assert True\n"


def test_added_only_patch_cleans_collision_without_modified_reset(repo, monkeypatch):
    path, base = repo
    patch = "".join(str(file) for file in reward.PatchSet(make_patch(path)) if file.is_added_file)
    (path / "test added.py").write_text('raise AssertionError("collision")\n')
    result = run_eval(path, base, patch, monkeypatch)
    assert result.returncode == 0
    assert "PASSED test added.py::test_official" in result.stdout
    assert "PASSED test_modified.py::test_existing" in result.stdout


def test_deleted_test_is_restored_before_patch_and_after_testing(repo, monkeypatch):
    path, base = repo
    git(path, "rm", "--", "test_modified.py")
    patch = git(path, "diff", "--cached")
    git(path, "reset", "--hard", base)
    (path / "test_remaining.py").write_text("def test_remaining():\n    assert True\n")
    (path / "test_modified.py").unlink()

    result = run_eval(path, base, patch, monkeypatch, ("test_remaining.py",))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED test_remaining.py::test_remaining" in result.stdout
    assert (path / "test_modified.py").read_text() == "def test_existing():\n    assert True\n"


@pytest.mark.parametrize("staged", [False, True])
def test_renamed_test_cleans_destination_and_restores_source(repo, monkeypatch, staged):
    path, base = repo
    git(path, "mv", "--", "test_modified.py", "test_renamed.py")
    patch = git(path, "diff", "--cached")
    git(path, "reset", "--hard", base)
    (path / "test_renamed.py").write_text('raise AssertionError("candidate collision")\n')
    if staged:
        git(path, "add", "--", "test_renamed.py")

    result = run_eval(path, base, patch, monkeypatch, ("test_renamed.py",))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED test_renamed.py::test_existing" in result.stdout
    assert (path / "test_modified.py").read_text() == "def test_existing():\n    assert True\n"
    assert not (path / "test_renamed.py").exists()


def test_modified_test_pathspec_metacharacters_do_not_reset_other_candidate_files(repo, monkeypatch):
    path, _ = repo
    (path / "test[1].py").write_text("def test_official():\n    assert True\n")
    (path / "test1.py").write_text("original = True\n")
    git(path, "add", "--", "test[1].py", "test1.py")
    git(path, "commit", "-m", "base literal paths")
    base = git(path, "rev-parse", "HEAD").strip()
    (path / "test[1].py").write_text("def test_official():\n    assert 1 == 1\n")
    patch = git(path, "diff")
    (path / "test[1].py").write_text('raise AssertionError("candidate modified test")\n')
    (path / "test1.py").write_text("candidate_change = True\n")

    result = run_eval(path, base, patch, monkeypatch, ("test_modified.py",))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED test_modified.py::test_existing" in result.stdout
    assert (path / "test[1].py").read_text() == "def test_official():\n    assert True\n"
    assert (path / "test1.py").read_text() == "candidate_change = True\n"
