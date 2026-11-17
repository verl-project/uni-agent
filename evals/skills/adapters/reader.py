"""Read-only MCP server for a single staged evidence workspace."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

TEXT = {"type": "string"}
MAX_FILE = 8 * 1024 * 1024
MAX_FILES = 2000


def tool(name, description, fields):
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": fields,
            "required": list(fields),
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False},
    }


TOOLS = [
    tool("inventory", "List staged paths only.", {}),
    tool(
        "read",
        "Read a bounded page from a staged UTF-8 file.",
        {"path": TEXT, "offset": {"type": "integer", "minimum": 0}},
    ),
    tool(
        "search", "Search a literal substring in the original scene; returns paths and line numbers.", {"literal": TEXT}
    ),
    tool("check", "Run the one experimenter-approved diagnostic helper on the complete scene, if available.", {}),
]


class Reader:
    def __init__(self, root, helper=None, sha=None):
        self.root = root.resolve(strict=True)
        self.files = {}
        self.helper = None
        for folder in ("scene", "package"):
            base = self.root / folder
            if not base.exists():
                continue
            for path in [base, *base.rglob("*")]:
                if (
                    path.is_symlink()
                    or getattr(path.lstat(), "st_file_attributes", 0) & 1024
                    or not path.resolve().is_relative_to(self.root)
                ):
                    raise ValueError("Unexpected linked input")
                if path.is_file():
                    if path.stat().st_size > MAX_FILE or len(self.files) >= MAX_FILES:
                        raise ValueError("Input exceeds reader limits")
                    self.files[path.relative_to(self.root).as_posix()] = path
        if helper:
            key = "package/scripts/" + helper
            if key in self.files:
                if self.files[key].suffix != ".py" or hashlib.sha256(self.files[key].read_bytes()).hexdigest() != sha:
                    raise ValueError("Helper pin mismatch")
                self.helper = self.files[key]

    def call(self, name, args):
        if name == "inventory":
            return "\n".join(sorted(self.files))
        if name == "read":
            p = self.files[args["path"]]
            offset = args["offset"]
            if type(offset) is not int or offset < 0:
                raise ValueError("Invalid offset")
            data = p.read_text(encoding="utf-8-sig")
            return data[offset : offset + 16000] + (
                f"\n[more; offset {offset + 16000}]" if len(data) > offset + 16000 else ""
            )
        if name == "search":
            literal = args["literal"]
            hits = []
            if not isinstance(literal, str) or not literal or len(literal) > 1000:
                raise ValueError("Invalid literal")
            for key, path in sorted(self.files.items()):
                if not key.startswith("scene/"):
                    continue
                with path.open(encoding="utf-8-sig", errors="replace") as f:
                    for number, line in enumerate(f, 1):
                        pos = line.find(literal)
                        if pos >= 0:
                            hits.append(
                                f"{key}:{number}: "
                                + line[max(0, pos - 80) : pos + min(len(literal), 150) + 90].rstrip()
                            )
                        if len(hits) == 50:
                            return "\n".join(hits) + "\n[more hits omitted]"
            return "\n".join(hits) or "[no matches]"
        if name == "check":
            if not self.helper:
                raise ValueError("No approved helper installed")
            env = {
                k: v
                for k, v in os.environ.items()
                if k.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "LANG"}
            }
            output = subprocess.run(
                [sys.executable, "-I", "-B", str(self.helper), str(self.root / "scene")],
                cwd=self.root,
                env=env,
                capture_output=True,
                timeout=12,
            )
            if output.returncode:
                raise ValueError("Approved helper exited with error")
            return output.stdout.decode("utf-8", "replace")[:24000]
        raise ValueError("Unsupported tool")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--helper")
    p.add_argument("--sha256")
    args = p.parse_args()
    reader = Reader(args.root, args.helper, args.sha256)
    for line in sys.stdin.buffer:
        req = json.loads(line)
        if "id" not in req:
            continue
        resp = {"jsonrpc": "2.0", "id": req["id"]}
        method = req["method"]
        params = req.get("params", {})
        if method == "initialize":
            result = {
                "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                "serverInfo": {"name": "evidence", "version": "1"},
                "capabilities": {"tools": {}},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            try:
                result = {
                    "content": [{"type": "text", "text": reader.call(params["name"], params.get("arguments", {}))}]
                }
            except (KeyError, ValueError, OSError, subprocess.SubprocessError):
                result = {"content": [{"type": "text", "text": "Unavailable or invalid request"}], "isError": True}
        elif method == "ping":
            result = {}
        elif method in ("resources/list", "resources/templates/list", "prompts/list"):
            result = {
                {
                    "resources/list": "resources",
                    "resources/templates/list": "resourceTemplates",
                    "prompts/list": "prompts",
                }[method]: []
            }
        else:
            result = None
            resp["error"] = {"code": -32601, "message": "Method not found"}
        if result is not None:
            resp["result"] = result
        sys.stdout.buffer.write(json.dumps(resp, ensure_ascii=True).encode() + b"\n")
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
