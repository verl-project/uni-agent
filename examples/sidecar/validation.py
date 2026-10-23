"""Prepare one tool-owned entrypoint check for Docker or a remote sandbox."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .build import load_recipe
else:
    from build import load_recipe


def add_check_arguments(parser):
    parser.add_argument("--recipe-file", type=Path, help="Custom build manifest used for this sidecar")
    parser.add_argument("--stdin-file", type=Path, help="Input bytes for the entrypoint check")
    parser.add_argument("--expected-stdout-file", type=Path, help="Expected stdout bytes, including final newlines")
    parser.add_argument("--check-script", type=Path, help="Tool-owned Python check; accepts --case CASE_JSON")


@dataclass
class EntrypointCheck:
    recipe: dict
    files: dict[str, str]

    def command(self, directory: str) -> list[str]:
        return ["python3", f"{directory}/probe.py", "--case", f"{directory}/case.json"]


def load_check(name, *, recipe_file=None, stdin_file=None, expected_stdout_file=None, check_script=None):
    recipe = load_recipe(name, recipe_file)
    if bool(stdin_file) != bool(expected_stdout_file):
        raise ValueError("--stdin-file and --expected-stdout-file must be supplied together")
    scripts = Path(__file__).parent
    if stdin_file is None and check_script is None:
        raise ValueError("custom recipes need --stdin-file and --expected-stdout-file, or --check-script")
    case = {
        "recipe": name,
        "mount_target": recipe["mount_target"],
        "entrypoint": recipe["entrypoint"],
        "probe_argv": recipe["probe_argv"],
    }
    if stdin_file is not None:
        case["stdin_base64"] = base64.b64encode(Path(stdin_file).read_bytes()).decode("ascii")
        case["expected_stdout_base64"] = base64.b64encode(Path(expected_stdout_file).read_bytes()).decode("ascii")
    script = Path(check_script) if check_script else scripts / "check_entrypoint.py"
    return EntrypointCheck(
        recipe,
        {"probe.py": script.read_text(encoding="utf-8"), "case.json": json.dumps(case)},
    )


def check_result(output):
    try:
        result = json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Entrypoint check did not return a final JSON result") from exc
    if not isinstance(result, dict) or result.get("result") != "passed":
        raise RuntimeError("Entrypoint check did not report result=passed")
    return result
