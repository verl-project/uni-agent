"""CPU checks for importing the pinned verl's root-only UV policy."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

pytestmark = [pytest.mark.cpu, pytest.mark.level0]
ROOT = Path(__file__).resolve().parents[2]


def run_generator(root, *args):
    return subprocess.run(
        [sys.executable, str(root / "scripts/refresh_uv.py"), *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "verl").mkdir()
    shutil.copyfile(ROOT / "scripts/refresh_uv.py", tmp_path / "scripts/refresh_uv.py")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "example"\nrequires-python = ">=3.10"\n')
    (tmp_path / "verl/pyproject.toml").write_text(
        """[tool.uv]
environments = ["sys_platform == 'linux'"]
override-dependencies = ["example==1.0"]
conflicts = [[{extra = "cpu"}, {extra = "gpu"}]]
[[tool.uv.index]]
name = "wheels"
url = "https://example.com/simple"
explicit = true
[[tool.uv.dependency-metadata]]
name = "native-package"
requires-dist = []
[tool.uv.sources.example]
index = "wheels"
"""
    )
    return tmp_path


def test_policy_roundtrip_preserves_member_ownership(workspace):
    upstream_path = workspace / "verl/pyproject.toml"
    upstream_text = upstream_path.read_text()
    upstream = tomllib.loads(upstream_text)["tool"]["uv"]
    result = run_generator(workspace)
    assert result.returncode == 0, result.stderr
    generated = tomllib.loads((workspace / "pyproject.toml").read_text())
    assert generated["project"]["requires-python"] == ">=3.10"
    policy = generated["tool"]["uv"]
    for key in ("environments", "override-dependencies", "dependency-metadata", "index"):
        assert policy[key] == upstream[key]
    assert "conflicts" not in policy
    assert policy["sources"] == {"verl": {"workspace": True}}
    assert policy["workspace"] == {"members": ["verl"]}
    assert upstream_path.read_text() == upstream_text


def test_check_detects_drift_without_writing(workspace):
    assert run_generator(workspace).returncode == 0
    path = workspace / "pyproject.toml"
    original = path.read_text()
    upstream = workspace / "verl/pyproject.toml"
    upstream.write_text(upstream.read_text().replace("example==1.0", "example==2.0"))
    result = run_generator(workspace, "--check")
    assert result.returncode != 0
    assert "stale" in result.stderr
    assert path.read_text() == original
    assert run_generator(workspace).returncode == 0
    updated = path.read_text()
    assert "example==2.0" in updated
    assert run_generator(workspace, "--check").returncode == 0
    assert path.read_text() == updated


def test_checked_in_policy_matches_pinned_verl():
    result = run_generator(ROOT, "--check")
    assert result.returncode == 0, result.stderr
