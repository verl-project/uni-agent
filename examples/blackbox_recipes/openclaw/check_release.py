"""Validate only the plugin against the installed release, outside the checkout.

Invoke with the dedicated environment Python, -I and an absolute script path.
The upstream checkout is not added to sys.path; only the OpenClaw namespace
extension and recipe test fixtures are loaded from the working source.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys

import uni_agent
import uni_agent.agents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    version = importlib.metadata.version("uni-agent")
    if version != "0.1.0rc1":
        raise RuntimeError(f"Expected release 0.1.0rc1, got {version}")
    root = Path(__file__).resolve().parents[3]
    package = Path(uni_agent.__file__).resolve()
    if root in package.parents:
        raise RuntimeError("Imported checkout instead of installed release; use python -I")
    uni_agent.agents.__path__.append(str(root / "uni_agent/agents"))
    from importlib import import_module
    import_module("uni_agent.agents.openclaw.agent")  # register only the new extension

    fixtures = root / "examples/blackbox_recipes/openclaw"
    sys.path.insert(0, str(fixtures))
    sys.argv = ["smoke_task.py", str(args.output)]
    runpy.run_path(str(fixtures / "smoke_task.py"), run_name="__main__")
    (args.output / "baseline.json").write_text(json.dumps({
        "version": version, "package": str(package),
        "plugin_source": str(root / "uni_agent/agents/openclaw"),
        "scope": "installed-release Task/Sandbox and new plugin; scripted model only",
    }, indent=2))


if __name__ == "__main__":
    main()
