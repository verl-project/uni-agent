import json
import os
import shlex
import subprocess
import sys

import pytest

from uni_agent.tasks.swe_bench import reward


def command(instance):
    commands = reward._make_eval_script_list(instance, {}, "testbed", "/testbed", instance["base_commit"], "")
    return commands[commands.index(f": '{reward.START_TEST_OUTPUT}'") + 1]


def test_sphinx_reporting_preserves_official_command_and_other_repositories(monkeypatch):
    directives = ["sphinx/testing/util.py", "tests/test_ext_napoleon.py"]
    monkeypatch.setattr(reward, "get_test_directives", lambda instance: directives)
    for version, specs in reward.MAP_REPO_VERSION_TO_SPECS["sphinx-doc/sphinx"].items():
        instance = {"repo": "sphinx-doc/sphinx", "version": version, "base_commit": "0" * 40}
        assert command(instance) == 'PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -rA" ' + " ".join(
            [specs["test_cmd"], *directives]
        )
    instance = {"repo": "psf/requests", "version": "2.2", "base_commit": "0" * 40}
    assert command(instance) == " ".join(
        [reward.MAP_REPO_VERSION_TO_SPECS["psf/requests"]["2.2"]["test_cmd"], *directives]
    )


@pytest.mark.parametrize("outcome", ["pass", "fail", "skip"])
def test_sphinx_reporting_real_tox_pytest_and_official_parser(tmp_path, monkeypatch, outcome):
    # Official 27ac testenv passenv/commands contract; local Python 3.12,
    # no Sphinx install or sandbox. current-env deliberately uses this interpreter.
    (tmp_path / "tox.ini").write_text(
        "[tox]\nminversion=2.4.0\n[testenv]\nskip_install=true\n"
        "passenv=PYTEST_ADDOPTS\ncommands=pytest --durations 25 {posargs}\n"
    )
    test_file = tmp_path / "test_reporting.py"
    body = {"pass": "assert True", "fail": "assert False", "skip": 'pytest.skip("required skipped")'}[outcome]
    test_file.write_text(
        "import os\nimport pytest\nclass TestRequired:\n    def test_required(self):\n"
        '        assert os.environ["PYTEST_ADDOPTS"].split() in '
        '(["--strict-markers"], ["--strict-markers", "-rA"])\n        ' + body + "\n"
    )
    test_cmd = shlex.quote(sys.executable) + " -m tox --current-env -epy312 -v --"
    monkeypatch.setitem(reward.MAP_REPO_VERSION_TO_SPECS["sphinx-doc/sphinx"], "cpu", {"test_cmd": test_cmd})
    monkeypatch.setattr(reward, "get_test_directives", lambda instance: ["test_reporting.py"])
    instance = {
        "instance_id": "sphinx-reporting-cpu",
        "repo": "sphinx-doc/sphinx",
        "version": "cpu",
        "base_commit": "0" * 40,
        "FAIL_TO_PASS": json.dumps(["test_reporting.py::TestRequired::test_required"]),
        "PASS_TO_PASS": "[]",
    }
    env = dict(os.environ, PYTEST_ADDOPTS="--strict-markers")
    node = "test_reporting.py::TestRequired::test_required"
    if outcome == "pass":
        original = subprocess.run(
            ["bash", "-c", test_cmd + " test_reporting.py"],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
        )
        assert original.returncode == 0, original.stdout + original.stderr
        assert "1 passed" in original.stdout
        original_output = f"{reward.START_TEST_OUTPUT}\n{original.stdout}\n{reward.END_TEST_OUTPUT}"
        original_status, original_found = reward._get_logs_eval(instance, original_output)
        assert original_found and original_status.get(node) != "PASSED"
        assert reward._get_eval_report(instance, original_output)["resolved"] is False
    result = subprocess.run(["bash", "-c", command(instance)], cwd=tmp_path, env=env, text=True, capture_output=True)
    output = f"{reward.START_TEST_OUTPUT}\n{result.stdout}\n{result.stderr}\n{reward.END_TEST_OUTPUT}"
    status, found = reward._get_logs_eval(instance, output)
    assert found
    if outcome == "pass":
        assert result.returncode == 0, output
        assert status[node] == "PASSED", output
    elif outcome == "fail":
        assert result.returncode != 0, output
        assert status[node] == "FAILED", output
    else:
        assert result.returncode == 0, output
        assert "SKIPPED" in output
        # Folded skip summaries do not identify the canonical node; never promote
        # a missing required test to PASSED merely because pytest exited zero.
        assert status.get(node) != "PASSED", output
    assert reward._get_eval_report(instance, output)["resolved"] is (outcome == "pass")
