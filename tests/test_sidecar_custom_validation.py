"""Exercise custom sidecar checks without a Docker daemon or service."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.sidecar.validation import check_result, load_check

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def custom_recipe(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    manifest = tmp_path / "recipes.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recipes": {
                    "custom-tool": {
                        "context": ".",
                        "dockerfile": "Dockerfile",
                        "image": "custom-tool",
                        "tag": "{TOOL_VERSION}",
                        "platform": "linux/amd64",
                        "mount_target": "/opt/custom-tool",
                        "entrypoint": [
                            sys.executable,
                            "-c",
                            "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())",
                        ],
                        "probe_argv": [sys.executable, "--version"],
                        "version_arg": "TOOL_VERSION",
                        "build_args": {"BASE_IMAGE": "debian:bookworm-slim", "TOOL_VERSION": "1.0"},
                    }
                },
            }
        )
    )
    request, expected = tmp_path / "input.bin", tmp_path / "expected.bin"
    request.write_bytes(b"sample\xff\n")
    expected.write_bytes(request.read_bytes())
    return manifest, request, expected


def test_custom_recipe_requires_a_functional_check(custom_recipe):
    manifest, request, _ = custom_recipe
    with pytest.raises(ValueError, match="custom recipes need"):
        load_check("custom-tool", recipe_file=manifest)
    with pytest.raises(ValueError, match="must be supplied together"):
        load_check("custom-tool", recipe_file=manifest, stdin_file=request)


@pytest.mark.parametrize("matches", [True, False])
def test_generic_check_runs_real_entrypoint_and_compares_bytes(custom_recipe, monkeypatch, tmp_path, capsys, matches):
    manifest, request, expected = custom_recipe
    if not matches:
        expected.write_bytes(b"wrong-output")
    check = load_check("custom-tool", recipe_file=manifest, stdin_file=request, expected_stdout_file=expected)
    case = tmp_path / "case.json"
    case.write_text(check.files["case.json"])
    spec = importlib.util.spec_from_file_location(
        "generic_sidecar_check",
        ROOT / "examples/sidecar/check_entrypoint.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "Path", lambda value: tmp_path / "testbed" if value == "/testbed" else Path(value))
    monkeypatch.setattr(sys, "argv", ["check_entrypoint.py", "--case", str(case)])
    if matches:
        module.main()
        assert check_result(capsys.readouterr().out)["recipe"] == "custom-tool"
    else:
        with pytest.raises(RuntimeError, match="stdout mismatch"):
            module.main()


def test_tool_owned_check_receives_resolved_contract(custom_recipe, tmp_path):
    manifest, _, _ = custom_recipe
    script = tmp_path / "checks.py"
    script.write_text("print('tool-specific check')\n")
    check = load_check("custom-tool", recipe_file=manifest, check_script=script)
    assert check.files["probe.py"] == script.read_text()
    assert json.loads(check.files["case.json"])["mount_target"] == "/opt/custom-tool"
    assert check.command("/tmp/check")[-2:] == ["--case", "/tmp/check/case.json"]


@pytest.mark.parametrize("output", ["", "{}", '{"result":"failed"}'])
def test_check_cannot_pass_on_exit_status_alone(output):
    with pytest.raises(RuntimeError, match="Entrypoint check"):
        check_result(output)


def test_custom_smoke_mounts_profile_and_uses_custom_target(custom_recipe, tmp_path, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(ROOT / "examples/sidecar"))
    spec = importlib.util.spec_from_file_location("custom_smoke", ROOT / "examples/sidecar/smoke.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    manifest, request, expected = custom_recipe
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke.py",
            "custom-tool",
            "--recipe-file",
            str(manifest),
            "--image",
            "custom-tool:1.0",
            "--stdin-file",
            str(request),
            "--expected-stdout-file",
            str(expected),
        ],
    )
    commands = []

    def docker(command, **kwargs):
        commands.append(command)
        if command[1] == "run":
            assert "none" == command[command.index("--network") + 1]
            assert any("dst=/opt/custom-tool,readonly" in value for value in command)
            case_mount = next(value for value in command if "dst=/sidecar-check,readonly" in value)
            case_dir = Path(case_mount.split("src=", 1)[1].split(",dst=", 1)[0])
            assert json.loads((case_dir / "case.json").read_text())["recipe"] == "custom-tool"
        return subprocess.CompletedProcess(command, 0, stdout='{"result":"passed"}\n', stderr="")

    monkeypatch.setattr(smoke.subprocess, "run", docker)
    smoke.main()
    assert [command[1] for command in commands] == ["create", "cp", "run", "rm", "rm"]
    assert '"smoke": "passed"' in capsys.readouterr().out
