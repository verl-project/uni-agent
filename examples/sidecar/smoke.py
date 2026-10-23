"""Validate a sidecar in a fresh Docker container with no network or credentials."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

if __package__:
    from .validation import add_check_arguments, check_result, load_check
else:
    from validation import add_check_arguments, check_result, load_check


def run(argv, **kwargs):
    try:
        return subprocess.run(argv, check=True, text=True, **kwargs)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr, end="")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", help="Recipe name in the manifest")
    parser.add_argument("--image", required=True, help="Built sidecar image reference")
    parser.add_argument("--runtime-image", default="python:3.11-slim-bookworm")
    add_check_arguments(parser)
    args = parser.parse_args()
    try:
        check = load_check(
            args.recipe,
            recipe_file=args.recipe_file,
            stdin_file=args.stdin_file,
            expected_stdout_file=args.expected_stdout_file,
            check_script=args.check_script,
        )
    except (ValueError, TypeError, OSError, KeyError) as exc:
        parser.error(str(exc))
    recipe = check.recipe
    mount = recipe["mount_target"]
    name = f"sidecar-check-{uuid.uuid4().hex[:12]}"
    runtime = f"{name}-runtime"
    with tempfile.TemporaryDirectory(prefix="sidecar-check-") as directory:
        payload = Path(directory) / "payload"
        payload.mkdir()
        try:
            run(["docker", "create", "--name", name, args.image, "/unused"], capture_output=True)
            run(["docker", "cp", f"{name}:/.", str(payload)])
            checks = Path(directory) / "checks"
            checks.mkdir()
            for filename, content in check.files.items():
                (checks / filename).write_text(content, encoding="utf-8")
            result = run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    runtime,
                    "--platform",
                    recipe["platform"],
                    "--network",
                    "none",
                    "--mount",
                    f"type=bind,src={payload},dst={mount},readonly",
                    "--mount",
                    f"type=bind,src={checks},dst=/sidecar-check,readonly",
                    "--entrypoint",
                    "python3",
                    args.runtime_image,
                    *check.command("/sidecar-check")[1:],
                ],
                timeout=180,
                capture_output=True,
            )
            check_result(result.stdout)
            print(result.stdout, end="")
        finally:
            # Remove only this invocation's resources, including an interrupted docker run.
            errors = []
            for container in (runtime, name):
                result = subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True, text=True)
                if result.returncode and "No such container" not in result.stderr:
                    errors.append(f"{container}: {result.stderr.strip()}")
            if errors:
                raise RuntimeError("Container cleanup failed: " + "; ".join(errors))
    print(json.dumps({"recipe": args.recipe, "image": args.image, "smoke": "passed", "cleanup": "passed"}))


if __name__ == "__main__":
    main()
