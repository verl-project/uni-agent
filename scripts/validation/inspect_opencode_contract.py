"""Record exact CLI/help/config behavior without model requests or version overrides."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from uni_agent.agents.base import ModelConfig
from uni_agent.agents.opencode import OpenCodeConfig
from uni_agent.agents.opencode.agent import _KEY_ENV, _runtime_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cli", required=True)
parser.add_argument("--expected", choices=["1.18.18", "1.19.3"], required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
config = _runtime_config(
    OpenCodeConfig(
        model=ModelConfig(
            base_url="http://127.0.0.1:9/v1",
            model_name="synthetic-policy",
            api_key="contract-dummy",
        )
    )
)
rows = []
with tempfile.TemporaryDirectory(prefix="ua-contract-") as tmp:
    env = {
        **os.environ,
        _KEY_ENV: "contract-dummy",
        "HOME": tmp,
        "TMPDIR": tmp,
        "OPENCODE_CONFIG": "",
        "OPENCODE_CONFIG_DIR": str(Path(tmp) / "config/opencode"),
        "OPENCODE_CONFIG_CONTENT": json.dumps(config),
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_DISABLE_MODELS_FETCH": "1",
        "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
    }
    for kind in ["CONFIG", "DATA", "CACHE", "STATE"]:
        env[f"XDG_{kind}_HOME"] = str(Path(tmp) / kind.lower())
    for command in [["--version"], ["run", "--help"], ["debug", "config"]]:
        result = subprocess.run([args.cli, *command], cwd=tmp, env=env, text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
        rows.append(dict(command=command, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr))
assert rows[0]["stdout"].strip() == args.expected
required = ["--pure", "--format", "--model", "--agent", "--dir", "--title"]
advertised = {flag: flag in rows[1]["stdout"] + rows[1]["stderr"] for flag in required}
parsed = json.loads(rows[2]["stdout"])
fields = ["model", "small_model", "permission", "compaction", "snapshot", "autoupdate"]
comparison = {
    field: {"expected": config[field], "resolved": parsed.get(field), "equal": parsed.get(field) == config[field]}
    for field in fields
}
args.output.write_text(
    json.dumps(
        {
            "version": args.expected,
            "production_guard_unchanged": True,
            "commands": rows,
            "required_flags": required,
            "advertised_flags": advertised,
            "config_comparison": comparison,
            "input_config_keys": sorted(config),
            "resolved_config_keys": sorted(parsed),
        },
        indent=2,
    )
    + "\n"
)
