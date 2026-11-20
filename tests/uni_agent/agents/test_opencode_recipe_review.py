"""Offline recipe contracts; no Ray cluster, model, or dataset download."""

import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.level0]
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/blackbox_recipes/opencode_swe_task/run_train.sh"


@pytest.mark.parametrize("key", ["prefix suffix", "prefix'quote\"suffix", "prefix\\suffix", "prefix\nsuffix"])
@pytest.mark.parametrize("mode", ["job", "local"])
def test_preview_never_calls_ray_and_redacts_before_quoting(tmp_path, monkeypatch, key, mode):
    monkeypatch.delenv("FILTER_GROUPS", raising=False)
    # A fake timeout records even a failed Ray status probe: dry run must not call it.
    marker = tmp_path / "called"
    probe = tmp_path / "timeout"
    probe.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\nexit 1\n")
    probe.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPT), "+ray_kwargs.ray_init.runtime_env.env_vars.E2B_API_KEY=override-secret"],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PRINT_CMD_ONLY": "1",
            "E2B_API_KEY": key,
            "RAY_SUBMIT_MODE": mode,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert not marker.exists()
    assert all(part not in result.stdout + result.stderr for part in ["prefix", "suffix", "override-secret"])
    arguments = shlex.split(result.stdout.splitlines()[-1])
    keys = [arg for arg in arguments if "E2B_API_KEY=" in arg]
    assert len(keys) == 2 and all(arg.endswith("E2B_API_KEY=<redacted>") for arg in keys)
    assert "algorithm.filter_groups.enable=False" in arguments


def test_live_job_mode_still_requires_ray(tmp_path):
    probe = tmp_path / "timeout"
    probe.write_text("#!/bin/sh\nexit 1\n")
    probe.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PRINT_CMD_ONLY": "0",
            "E2B_API_KEY": "dummy",
            "RAY_SUBMIT_MODE": "job",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "No running Ray cluster" in result.stderr


@pytest.fixture
def preprocess(monkeypatch):
    # Exercise selection with an in-memory dataset, without installing the
    # optional Hugging Face package or weakening other tests' imports.
    dependency = ModuleType("datasets")
    dependency.load_dataset = lambda *a, **kw: pytest.fail("unexpected dataset load")
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "datasets", dependency)
        spec = importlib.util.spec_from_file_location(
            "review_preprocess", ROOT / "uni_agent/tasks/swe_rebench/preprocess.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class Dataset:
    def __init__(self, ids):
        self.ids = list(ids)
        self.column_names = ["instance_id"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, key):
        assert key == "instance_id"
        return self.ids

    def select(self, indices):
        return Dataset([self.ids[i] for i in indices])

    def map(self, process, remove_columns):
        return self


@pytest.mark.parametrize("kwargs", [{"data_files": []}, {"instance_ids": []}, {"instance_ids": [" "]}])
def test_empty_input_rejected_before_load(preprocess, kwargs):
    with pytest.raises(ValueError):
        preprocess.build_swe_rebench(**kwargs)


def test_selection_order_missing_and_duplicate_ids(preprocess):
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return Dataset(["b", "a", "c", "c"])

    preprocess.load_dataset = load
    assert preprocess.build_swe_rebench(data_files=["local.parquet"], instance_ids=["a", "b", "a"]).ids == ["a", "b"]
    assert calls[0] == (("parquet",), {"data_files": ["local.parquet"], "split": "train"})
    with pytest.raises(ValueError, match="not found"):
        preprocess.build_swe_rebench(instance_ids=["missing"])
    with pytest.raises(ValueError, match="duplicate"):
        preprocess.build_swe_rebench(instance_ids=["c"])
    assert preprocess.build_swe_rebench(max_instances=1).ids == ["b"]


@pytest.mark.parametrize("selection", ["", ",,,"])
def test_empty_cli_selection_never_downloads(tmp_path, selection):
    (tmp_path / "datasets.py").write_text("def load_dataset(*a, **kw): raise AssertionError('must not download')\n")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "uni_agent/tasks/swe_rebench/preprocess.py"),
            "--instance-ids",
            selection,
            "--local-save-dir",
            str(tmp_path / "out"),
        ],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0 and "instance_ids must contain" in result.stderr
    assert "must not download" not in result.stderr
