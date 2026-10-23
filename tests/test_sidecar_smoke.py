"""Verify failed smoke runs clean their containers and never report success."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SIDECAR_DIR = Path(__file__).resolve().parents[1] / "examples" / "sidecar"


@pytest.fixture
def smoke(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SIDECAR_DIR))
    build_spec = importlib.util.spec_from_file_location("sidecar_build_for_smoke", SIDECAR_DIR / "build.py")
    build = importlib.util.module_from_spec(build_spec)
    build_spec.loader.exec_module(build)
    monkeypatch.setitem(sys.modules, "build", build)
    smoke_spec = importlib.util.spec_from_file_location("sidecar_smoke", SIDECAR_DIR / "smoke.py")
    module = importlib.util.module_from_spec(smoke_spec)
    smoke_spec.loader.exec_module(module)
    request, expected = tmp_path / "input.txt", tmp_path / "expected.txt"
    request.write_text("abc")
    expected.write_text("sample-output")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke.py",
            "my-tool",
            "--image",
            "my-tool-sidecar:test",
            "--stdin-file",
            str(request),
            "--expected-stdout-file",
            str(expected),
        ],
    )
    return module


@pytest.mark.parametrize("failed_stage", ["create", "run"])
def test_failed_probe_or_partial_creation_cleans_both_containers(smoke, monkeypatch, capsys, failed_stage):
    live = set()
    removed = []
    names = {}

    def docker(argv, **kwargs):
        operation = argv[1]
        if operation in {"create", "run"}:
            name = argv[argv.index("--name") + 1]
            names[operation] = name
            live.add(name)
            if operation == failed_stage:
                # The daemon can create a resource before the client reports failure.
                raise subprocess.CalledProcessError(23, argv, stderr="probe failed")
        elif operation == "rm":
            removed.append(argv[-1])
            live.discard(argv[-1])
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(smoke.subprocess, "run", docker)
    with pytest.raises(subprocess.CalledProcessError) as error:
        smoke.main()
    assert error.value.returncode == 23
    assert live == set()
    assert removed == [f"{names['create']}-runtime", names["create"]]
    assert '"smoke": "passed"' not in capsys.readouterr().out


def test_cleanup_failure_prevents_success_and_still_cleans_other_container(smoke, monkeypatch, capsys):
    removed = []
    names = {}

    def docker(argv, **kwargs):
        operation = argv[1]
        if operation in {"create", "run"}:
            names[operation] = argv[argv.index("--name") + 1]
        elif operation == "rm":
            removed.append(argv[-1])
            if argv[-1] == names["run"]:
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr="permission denied")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(smoke.subprocess, "run", docker)
    with pytest.raises(RuntimeError, match="Container cleanup failed: .*permission denied"):
        smoke.main()
    assert removed == [names["run"], names["create"]]
    assert '"smoke": "passed"' not in capsys.readouterr().out
