"""Regression coverage for Ray's relocated UV working directories."""

import pytest

from tests.uni_agent.test_uv_gpu import assert_equivalent_environments

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def environment(project):
    return {
        "project": project,
        "prefix": f"{project}/.venv",
        "python_version": (3, 12, 16),
        "lock_sha256": "same-lock",
        "policy_sha256": "same-policy",
        "versions": {"torch": "2.13.0", "ray": "2.59.0"},
        "cuda": "13.0",
    }


def test_relocated_environment_with_same_lock_is_valid():
    driver = environment("/checkout/uni-agent")
    worker = environment("/tmp/ray/working_dir_files/ray-package")
    assert_equivalent_environments(driver, worker)


@pytest.mark.parametrize("field", ["lock_sha256", "policy_sha256", "python_version", "versions", "cuda"])
def test_environment_drift_is_rejected(field):
    driver = environment("/checkout/uni-agent")
    worker = environment("/tmp/ray/working_dir_files/ray-package")
    worker[field] = "different"
    with pytest.raises(AssertionError, match=field):
        assert_equivalent_environments(driver, worker)


def test_ambient_environment_is_rejected():
    driver = environment("/checkout/uni-agent")
    worker = environment("/tmp/ray/working_dir_files/ray-package")
    worker["prefix"] = "/opt/ambient/venv"
    with pytest.raises(AssertionError, match="prefix"):
        assert_equivalent_environments(driver, worker)
