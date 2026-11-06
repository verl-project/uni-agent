"""Small helpers for R2E-Gym dataset rows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _range_patch(value: Mapping[str, Any]) -> str:
    start = value["start"]
    length = value.get("length")
    return str(start) if length is None else f"{start},{length}"


def _parsed_commit_patch(parsed_content: Mapping[str, Any]) -> str:
    """Rebuild the output of R2E-Gym's ``ParsedCommit.get_patch()``."""
    patch: list[str] = []
    for file_diff in parsed_content.get("file_diffs", []):
        header = file_diff["header"]
        path = header["file"]["path"]
        if not path.endswith(".py"):
            continue

        patch.append(f"diff --git a/{path} b/{path}\n")
        if misc_line := header.get("misc_line"):
            patch.append(f"{misc_line}\n")

        if index_line := file_diff.get("index_line"):
            mode = index_line.get("mode") or ""
            suffix = f" {mode}" if mode else ""
            patch.append(f"index {index_line['old_commit_hash']}..{index_line['new_commit_hash']}{suffix}\n")

        if file_diff.get("is_binary_file"):
            patch.append(f"{file_diff['binary_line']}\n")

        minus_file = file_diff.get("minus_file")
        plus_file = file_diff.get("plus_file")
        if minus_file and plus_file:
            patch.append(f"--- {minus_file['path']}\n")
            patch.append(f"+++ {plus_file['path']}\n")

        for hunk in file_diff.get("hunks", []):
            descriptor = hunk["descriptor"]
            descriptor_line = f"@@ -{_range_patch(descriptor['old_range'])} +{_range_patch(descriptor['new_range'])} @@"
            if section := descriptor.get("section"):
                descriptor_line += f" {section}"
            patch.append(f"{descriptor_line}\n")

            prefixes = {"context": " ", "added": "+", "deleted": "-", "note": "\\ "}
            for diff_line in hunk["line_group"].get("all_lines", []):
                line_type = diff_line["type"]
                try:
                    prefix = prefixes[line_type]
                except KeyError as exc:
                    raise ValueError(f"unknown R2E-Gym diff line type: {line_type!r}") from exc
                patch.append(f"{prefix}{diff_line['content']}\n")

    return "".join(patch)


def get_gold_patch(example: Mapping[str, Any]) -> str:
    """Read a ready patch or reconstruct it from serialized commit metadata."""
    patch = example.get("patch")
    if isinstance(patch, str) and patch.strip():
        return patch

    parsed_content = example.get("parsed_commit_content")
    if parsed_content is None:
        raise ValueError("R2E-Gym row has neither 'patch' nor 'parsed_commit_content'")
    if isinstance(parsed_content, str):
        parsed_content = json.loads(parsed_content)
    if not isinstance(parsed_content, Mapping):
        raise ValueError("R2E-Gym 'parsed_commit_content' must be a JSON object")
    return _parsed_commit_patch(parsed_content)
