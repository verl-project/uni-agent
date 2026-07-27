#!/usr/bin/env python3

import json
import os
import re
import sys


class PRBodyLoadError(RuntimeError):
    pass


class PRDescriptionError(ValueError):
    pass


SUMMARY_HEADING = "## Summary"
HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
MARKDOWN_PREFIX_PATTERN = re.compile(r"^\s*(?:(?:[-*+]|\d+[.)]|>)\s*)+")
PLACEHOLDER_VALUES = {"todo", "tbd", "n/a", "none", "..."}


def load_pr_body(event_path: str) -> str:
    try:
        with open(event_path, encoding="utf-8") as event_file:
            payload = json.load(event_file)
        body = payload.get("pull_request", {}).get("body", "") or ""
        if not isinstance(body, str):
            raise TypeError("pull_request.body must be a string or null")
        return body
    except (OSError, TypeError, AttributeError, json.JSONDecodeError) as error:
        raise PRBodyLoadError(f"Failed to read PR body from {event_path}: {error}") from error


def extract_section(body: str, heading: str) -> str:
    heading_pattern = re.compile(rf"^{re.escape(heading)}[ \t]*$", re.MULTILINE)
    heading_match = heading_pattern.search(body)
    if heading_match is None:
        raise PRDescriptionError(f"PR description must keep the '{heading}' section from the template.")

    section_start = heading_match.end()
    next_heading = re.search(r"^##\s+\S.*$", body[section_start:], re.MULTILINE)
    section_end = section_start + next_heading.start() if next_heading is not None else len(body)
    return body[section_start:section_end]


def _plain_markdown(content: str) -> str:
    without_comments = HTML_COMMENT_PATTERN.sub("", content)
    lines = []
    for line in without_comments.splitlines():
        plain_line = MARKDOWN_PREFIX_PATTERN.sub("", line).strip()
        if plain_line:
            lines.append(plain_line)
    return " ".join(lines)


def check_pr_description(body: str) -> None:
    if not body.strip():
        raise PRDescriptionError("PR description is empty.")

    summary = _plain_markdown(extract_section(body, SUMMARY_HEADING))
    normalized_summary = summary.casefold().strip(" .!?:;_-")
    if not summary or normalized_summary in PLACEHOLDER_VALUES:
        raise PRDescriptionError("Fill in '## Summary' with the problem, solution, and reason for the change.")
    if not any(character.isalnum() for character in summary):
        raise PRDescriptionError("'## Summary' must contain a meaningful description.")


def _escape_workflow_command(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main() -> int:
    event_path = os.getenv("GITHUB_EVENT_PATH")
    if not event_path:
        print("::error title=Invalid PR description::GITHUB_EVENT_PATH is not set.")
        return 1

    try:
        pr_body = load_pr_body(event_path)
        check_pr_description(pr_body)
    except (PRBodyLoadError, PRDescriptionError) as error:
        print(f"::error title=Invalid PR description::{_escape_workflow_command(str(error))}")
        return 1

    print("PR description contains a completed Summary section.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
