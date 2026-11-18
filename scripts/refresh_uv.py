#!/usr/bin/env python3
"""Import root-only uv settings from the pinned verl pyproject (Python 3.12).

uv natively supports the verl dependency through tool.uv.workspace/sources;
this script is not needed to reference or install a workspace member. However,
workspace membership does not inherit all tool.uv settings: overrides, for
example, are read only from the workspace root, ignoring member declarations.
See https://docs.astral.sh/uv/reference/settings/#override-dependencies.

The root settings could be maintained by hand. This repository instead copies
the pinned verl policy with this maintenance helper to avoid configuration
drift. Run it after bumping verl; --check detects drift without changing files.
Normal locked installs do not need to run this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tomllib

MARKER = "# BEGIN GENERATED VERL UV POLICY"


def value(item):
    """Serialize the string/bool/list/table values used by uv configuration."""
    if isinstance(item, str):
        return json.dumps(item, ensure_ascii=False)
    if isinstance(item, bool):
        return str(item).lower()
    if isinstance(item, int | float):
        return str(item)
    if isinstance(item, list):
        return "[" + ", ".join(value(part) for part in item) + "]"
    if isinstance(item, dict):
        return "{ " + ", ".join(f"{json.dumps(key)} = {value(part)}" for key, part in item.items()) + " }"
    raise TypeError(f"Unsupported TOML value: {type(item)}")


def table(name, settings, array=False):
    header = f"[[{name}]]" if array else f"[{name}]"
    lines = [header]
    children = []
    for key, item in settings.items():
        field = json.dumps(key)
        if isinstance(item, dict):
            children.append(table(f"{name}.{field}", item))
        elif isinstance(item, list) and item and all(isinstance(part, dict) for part in item):
            children.extend(table(f"{name}.{field}", part, array=True) for part in item)
        elif isinstance(item, list) and item:
            lines.append(f"{field} = [")
            lines.extend(f"    {value(part)}," for part in item)
            lines.append("]")
        else:
            lines.append(f"{field} = {value(item)}")
    return "\n".join(lines) + "\n" + "".join("\n" + child for child in children)


def write_generated(path, content, check=False):
    old = path.read_text()
    updated = old.split(MARKER)[0].rstrip() + "\n\n" + MARKER + "\n" + content
    if old == updated:
        return
    if check:
        raise SystemExit(f"{path}: generated uv policy is stale; run scripts/refresh_uv.py")
    path.write_text(updated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    upstream = tomllib.loads((root / "verl/pyproject.toml").read_text())["tool"]["uv"]
    # Conflicts/sources remain on the verl member. Do not create a second,
    # independent backend-conflict dimension on Uni-Agent's extras.
    inherited = {
        key: upstream[key]
        for key in ("environments", "override-dependencies", "dependency-metadata", "index")
        if key in upstream
    }
    policy = table("tool.uv", inherited)
    policy += "\n" + table("tool.uv.workspace", {"members": ["verl"]})
    policy += "\n" + table("tool.uv.sources", {"verl": {"workspace": True}})
    write_generated(root / "pyproject.toml", policy, args.check)


if __name__ == "__main__":
    main()
