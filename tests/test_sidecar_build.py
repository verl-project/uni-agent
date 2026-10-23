"""Public custom-recipe build behavior without a Docker daemon or network."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from examples.sidecar import build


def test_describe_resolves_manifest_paths_and_explicit_version(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build.shutil, "which", lambda _: None)
    assert build.main(["my-tool", "--version", "1.37.0", "--no-cache", "--pull", "--describe"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["image_ref"] == "my-tool-sidecar:1.37.0"
    assert result["build_args"]["TOOL_VERSION"] == "1.37.0"
    assert Path(result["context"]).is_dir()
    assert "--no-cache" in result["build_command"] and "--pull" in result["build_command"]


def test_environment_does_not_change_build_manifest(monkeypatch, capsys):
    for name in ("TOOL_VERSION", "TOOL_IMAGE", "TOOL_TAG", "BASE_IMAGE"):
        monkeypatch.setenv(name, "unrelated-value")
    assert build.main(["my-tool", "--describe"]) == 0
    recipe = json.loads(capsys.readouterr().out)
    assert recipe["image_ref"] == "my-tool-sidecar:1.36.1"
    assert recipe["build_args"]["BASE_IMAGE"] == "busybox:1.36.1-musl"


@pytest.mark.parametrize("include_new_key", [False, True])
def test_custom_pip_index_maps_to_the_declared_install_argument(tmp_path, capsys, include_new_key):
    recipe = build.load_recipe("my-tool")
    recipe["build_args"]["PIP_INDEX_URL"] = "https://legacy.example/simple"
    if include_new_key:
        recipe["build_args"]["INSTALL_PIP_INDEX_URL"] = ""
    manifest = tmp_path / "recipes.json"
    manifest.write_text(json.dumps({"schema_version": 1, "recipes": {"custom": recipe}}))
    assert (
        build.main(
            [
                "custom",
                "--recipe-file",
                str(manifest),
                "--pip-index",
                "https://mirror.example/simple",
                "--describe",
            ]
        )
        == 0
    )
    configured = json.loads(capsys.readouterr().out)
    key = "INSTALL_PIP_INDEX_URL" if include_new_key else "PIP_INDEX_URL"
    assert configured["build_args"][key] == "https://mirror.example/simple"


@pytest.mark.parametrize("push", [False, True])
def test_registry_push_requires_explicit_opt_in(monkeypatch, push):
    calls = []
    monkeypatch.setattr(build.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(build.subprocess, "run", lambda command, check: calls.append(command))
    argv = ["my-tool", "--registry", "registry.example:5000/team", "--tag", "release-1"]
    if push:
        argv.append("--push")
    assert build.main(argv) == 0
    assert len(calls) == (2 if push else 1)
    assert "registry.example:5000/team/my-tool-sidecar:release-1" in calls[0]
    if push:
        assert calls[1] == ["docker", "push", "registry.example:5000/team/my-tool-sidecar:release-1"]


def test_failed_build_never_pushes(monkeypatch):
    calls = []

    def fail(command, check):
        calls.append(command)
        raise subprocess.CalledProcessError(17, command)

    monkeypatch.setattr(build.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(build.subprocess, "run", fail)
    assert build.main(["my-tool", "--registry", "registry.example/team", "--push"]) == 17
    assert len(calls) == 1


@pytest.mark.parametrize(
    "options",
    [
        ["--push"],
        ["--build-arg", "API_KEY=private-value"],
        ["--base-image", "https://person:private-value@registry.example"],
        ["--registry", "https://registry.example"],
        ["--image", "name:tag"],
    ],
)
def test_invalid_arguments_fail_without_echoing_sensitive_values(options, capsys):
    with pytest.raises(SystemExit) as error:
        build.main(["my-tool", "--dry-run", *options])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out and "private-value" not in captured.err


def test_manifest_paths_are_relative_to_custom_file_and_allow_spaces(tmp_path, monkeypatch, capsys):
    context = tmp_path / "custom tool"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\n")
    recipe = build.load_recipe("my-tool")
    recipe.update(context="custom tool", dockerfile="custom tool/Dockerfile", image="custom-tool")
    manifest = tmp_path / "recipes.json"
    manifest.write_text(json.dumps({"schema_version": 1, "recipes": {"custom": recipe}}))
    monkeypatch.chdir(Path(__file__).parents[1])
    assert (
        build.main(
            [
                "custom",
                "--recipe-file",
                str(manifest),
                "--build-arg",
                "TOOL_VERSION=2.0",
                "--describe",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["image_ref"] == "custom-tool:2.0"
    assert result["build_command"][-1] == str(context)


def test_malformed_manifest_is_a_cli_error(tmp_path):
    recipe = build.load_recipe("my-tool")
    recipe["build_args"] = []
    manifest = tmp_path / "recipes.json"
    manifest.write_text(json.dumps({"schema_version": 1, "recipes": {"custom": recipe}}))
    with pytest.raises(SystemExit, match="2"):
        build.main(["custom", "--recipe-file", str(manifest), "--describe"])
