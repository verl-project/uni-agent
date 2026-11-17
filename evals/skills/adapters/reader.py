"""Read-only MCP server for a single staged evidence workspace."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TEXT = {"type": "string"}
MAX_FILE = 8 * 1024 * 1024
MAX_FILES = 2000


def tool(name, description, fields, optional=()):
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": fields,
            "required": [key for key in fields if key not in optional],
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
    tool(
        "check",
        "Run the approved helper on the whole scene. Optionally supply a literal ticket marker for --match.",
        {"literal": {"type": "string", "minLength": 1, "maxLength": 1000}},
        optional=("literal",),
    ),
]


CHECKPOINT_TOOL = tool(
    "submit_evidence",
    "Submit a provisional evidence finding without receiving correctness feedback. Use the public answer contract.",
    {"finding": {"type": "object"}},
)


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
            if not isinstance(args, dict) or set(args) - {"literal"}:
                raise ValueError("Unknown helper arguments")
            argv = [sys.executable, "-I", "-B", str(self.helper), str(self.root / "scene")]
            if "literal" in args:
                literal = args["literal"]
                if not isinstance(literal, str) or not 1 <= len(literal) <= 1000 or "\0" in literal:
                    raise ValueError("Invalid literal")
                argv.append("--match=" + literal)
            env = {
                k: v
                for k, v in os.environ.items()
                if k.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "LANG"}
            }
            output = subprocess.run(
                argv,
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
    p.add_argument("--journal", type=Path)
    p.add_argument("--start-ns", type=int)
    args = p.parse_args()
    if (args.journal is None) != (args.start_ns is None):
        p.error("journal and start-ns must be set together")
    submissions = 0
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
            result = {"tools": TOOLS + ([CHECKPOINT_TOOL] if args.journal else [])}
        elif method == "tools/call":
            try:
                if args.journal and params["name"] == "submit_evidence":
                    finding = params["arguments"]["finding"]
                    ns = time.monotonic_ns() - args.start_ns
                    if not isinstance(finding, dict) or len(json.dumps(finding)) > 32000 or submissions >= 32:
                        raise ValueError("Invalid checkpoint")
                    with args.journal.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"ns": ns, "finding": finding}) + "\n")
                    submissions += 1
                    text = "Recorded. No correctness feedback."
                else:
                    text = reader.call(params["name"], params.get("arguments", {}))
                result = {"content": [{"type": "text", "text": text}]}
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
