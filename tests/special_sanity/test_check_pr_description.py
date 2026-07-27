import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from check_pr_description import (
    PRBodyLoadError,
    PRDescriptionError,
    check_pr_description,
    load_pr_body,
    main,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class CheckPRDescriptionTest(unittest.TestCase):
    def test_accepts_completed_summary(self):
        body = """\
## Summary

<!-- Template guidance that should be ignored. -->
Fixes a race when sibling rollout sessions finish out of order.

## Changes

- Preserve each session result.
"""

        check_pr_description(body)

    def test_accepts_non_english_summary(self):
        check_pr_description("## Summary\n\n修复多个 Gateway session 并发结束时的状态覆盖问题。\n")

    def test_rejects_empty_body(self):
        with self.assertRaisesRegex(PRDescriptionError, "empty"):
            check_pr_description("")

    def test_rejects_missing_summary_heading(self):
        with self.assertRaisesRegex(PRDescriptionError, "must keep"):
            check_pr_description("## Changes\n\n- Add a feature.\n")

    def test_rejects_untouched_template_summary(self):
        body = """\
## Summary

<!-- Explain the problem, solution, and reason for the change. -->

## Changes

-
"""

        with self.assertRaisesRegex(PRDescriptionError, "Fill in"):
            check_pr_description(body)

    def test_rejects_repository_template_without_edits(self):
        template = (REPOSITORY_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")

        with self.assertRaisesRegex(PRDescriptionError, "Fill in"):
            check_pr_description(template)

    def test_rejects_placeholder_summary(self):
        for placeholder in ("TODO", "TODO.", "TBD", "N/A", "None", "...", "- TODO"):
            with self.subTest(placeholder=placeholder), self.assertRaises(PRDescriptionError):
                check_pr_description(f"## Summary\n\n{placeholder}\n")

    def test_rejects_punctuation_only_summary(self):
        with self.assertRaisesRegex(PRDescriptionError, "meaningful"):
            check_pr_description("## Summary\n\n--- !!!\n")


class LoadPRBodyTest(unittest.TestCase):
    def test_loads_body_from_github_event(self):
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "event.json"
            event_path.write_text(
                json.dumps({"pull_request": {"body": "## Summary\n\nA fix."}}),
                encoding="utf-8",
            )

            self.assertEqual(load_pr_body(str(event_path)), "## Summary\n\nA fix.")

    def test_wraps_invalid_event_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "event.json"
            event_path.write_text("{", encoding="utf-8")

            with self.assertRaises(PRBodyLoadError):
                load_pr_body(str(event_path))


class MainTest(unittest.TestCase):
    def test_returns_zero_for_completed_description(self):
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "event.json"
            event_path.write_text(
                json.dumps({"pull_request": {"body": "## Summary\n\nPreserve sandbox exit codes."}}),
                encoding="utf-8",
            )
            output = io.StringIO()
            with (
                mock.patch.dict(
                    os.environ,
                    {"GITHUB_EVENT_PATH": str(event_path)},
                    clear=True,
                ),
                contextlib.redirect_stdout(output),
            ):
                exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertIn("completed Summary", output.getvalue())

    def test_returns_one_when_event_path_is_missing(self):
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            contextlib.redirect_stdout(output),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertTrue(output.getvalue().startswith("::error title=Invalid PR description::"))


if __name__ == "__main__":
    unittest.main()
