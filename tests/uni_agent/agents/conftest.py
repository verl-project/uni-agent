"""Version selection for explicitly enabled OpenCode CLI probes."""

import os
import subprocess

import pytest


@pytest.fixture(scope="module", autouse=True)
def verify_selected_opencode_release(request):
    modules = {
        "test_opencode_cli_probe.py",
        "test_opencode_process_cleanup.py",
        "test_opencode_task_smoke.py",
        "test_opencode_gateway_integration.py",
        "test_opencode_sidecar_environment.py",
    }
    if request.node.name not in modules:
        return
    variable = "OPENCODE_SIDECAR_BIN" if "sidecar" in request.node.name else "OPENCODE_BIN"
    cli = os.environ.get(variable)
    if not cli:
        return
    expected = os.environ.get("OPENCODE_VERSION", "1.19.3")
    result = subprocess.run([cli, "--version"], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected, (result.stdout, expected)
