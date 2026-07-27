#!/usr/bin/env python3

import os
import re
import sys
from dataclasses import dataclass

ALLOWED_AREAS = (
    "agents",
    "framework",
    "gateway",
    "logging",
    "sandbox",
    "tasks",
    "tools",
    "training",
    "app",
    "docs",
    "examples",
    "ci",
    "build",
    "deps",
    "misc",
)
ALLOWED_TYPES = (
    "feat",
    "fix",
    "refactor",
    "perf",
    "test",
    "docs",
    "chore",
    "revert",
)

TITLE_FORMAT = "[area] type: summary"
TITLE_EXAMPLE = "[agents, sandbox] feat: add isolated harness execution"

TITLE_PATTERN = re.compile(
    r"^(?:\[(?P<series_index>[1-9]\d*)/(?P<series_total>[1-9]\d*|N)\])?"
    r"(?P<breaking>\[BREAKING\])?"
    r"\[(?P<areas>[a-z]+(?:, [a-z]+)*)\] "
    r"(?P<change_type>[a-z]+): "
    r"(?P<summary>\S(?:.*\S)?)$"
)


class PRTitleError(ValueError):
    """Raised when a pull request title does not follow the project convention."""


@dataclass(frozen=True)
class PRTitle:
    areas: tuple[str, ...]
    change_type: str
    summary: str
    breaking: bool = False
    series_index: int | None = None
    series_total: int | str | None = None


def parse_pr_title(title: str) -> PRTitle:
    if not title:
        raise PRTitleError(f"PR title is empty. Expected '{TITLE_FORMAT}'.")
    if title != title.strip():
        raise PRTitleError("PR title must not have leading or trailing whitespace.")

    match = TITLE_PATTERN.fullmatch(title)
    if match is None:
        raise PRTitleError(f"Expected '{TITLE_FORMAT}' with lowercase areas and type. Example: '{TITLE_EXAMPLE}'.")

    areas = tuple(match.group("areas").split(", "))
    unknown_areas = tuple(area for area in areas if area not in ALLOWED_AREAS)
    if unknown_areas:
        raise PRTitleError(f"Unknown area(s): {', '.join(unknown_areas)}. Allowed areas: {', '.join(ALLOWED_AREAS)}.")

    duplicate_areas = tuple(dict.fromkeys(area for area in areas if areas.count(area) > 1))
    if duplicate_areas:
        raise PRTitleError(f"Duplicate area(s): {', '.join(duplicate_areas)}.")

    change_type = match.group("change_type")
    if change_type not in ALLOWED_TYPES:
        raise PRTitleError(f"Unknown type '{change_type}'. Allowed types: {', '.join(ALLOWED_TYPES)}.")

    series_index_text = match.group("series_index")
    series_total_text = match.group("series_total")
    series_index = int(series_index_text) if series_index_text is not None else None
    if series_total_text is None:
        series_total: int | str | None = None
    elif series_total_text == "N":
        series_total = series_total_text
    else:
        series_total = int(series_total_text)

    if isinstance(series_total, int) and series_index is not None and series_index > series_total:
        raise PRTitleError(f"Stacked PR index {series_index} cannot be greater than total {series_total}.")

    return PRTitle(
        areas=areas,
        change_type=change_type,
        summary=match.group("summary"),
        breaking=match.group("breaking") is not None,
        series_index=series_index,
        series_total=series_total,
    )


def _escape_workflow_command(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main() -> int:
    title = os.environ.get("PR_TITLE", "")
    try:
        parsed = parse_pr_title(title)
    except PRTitleError as error:
        print(f"::error title=Invalid PR title::{_escape_workflow_command(str(error))}")
        return 1

    flags = []
    if parsed.breaking:
        flags.append("breaking")
    if parsed.series_index is not None:
        flags.append(f"stack {parsed.series_index}/{parsed.series_total}")
    suffix = f" ({', '.join(flags)})" if flags else ""
    print(f"PR title is valid: areas={', '.join(parsed.areas)}, type={parsed.change_type}{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
