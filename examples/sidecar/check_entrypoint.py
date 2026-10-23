"""Check a custom sidecar's declared probe and stdin/stdout contract."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case.read_text(encoding="utf-8"))
    workdir = Path("/testbed")
    workdir.mkdir(exist_ok=True)
    for key in ("probe_argv", "entrypoint"):
        argv = case[key]
        result = subprocess.run(
            argv,
            input=base64.b64decode(case["stdin_base64"]) if key == "entrypoint" else b"",
            capture_output=True,
            check=False,
            timeout=90,
            cwd=workdir,
        )
        if result.returncode:
            raise RuntimeError(f"{key} failed with exit code {result.returncode}: {result.stderr[-2000:]!r}")
    expected = base64.b64decode(case["expected_stdout_base64"])
    if result.stdout != expected:
        raise RuntimeError(
            f"Entrypoint stdout mismatch: expected {len(expected)} bytes, received {len(result.stdout)} bytes"
        )
    print(
        json.dumps(
            {
                "recipe": case["recipe"],
                "entrypoint": "passed",
                "entrypoint_path": case["entrypoint"][0],
                "dependency_probe": "passed",
                "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
                "result": "passed",
            }
        )
    )


if __name__ == "__main__":
    main()
