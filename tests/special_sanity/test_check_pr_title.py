import contextlib
import io
import os
import unittest
from pathlib import Path
from unittest import mock

from check_pr_title import (
    ALLOWED_AREAS,
    ALLOWED_TYPES,
    PRTitleError,
    main,
    parse_pr_title,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ParsePRTitleTest(unittest.TestCase):
    def test_accepts_architecture_areas(self):
        parsed = parse_pr_title("[agents, sandbox] feat: add isolated harness execution")

        self.assertEqual(parsed.areas, ("agents", "sandbox"))
        self.assertEqual(parsed.change_type, "feat")
        self.assertEqual(parsed.summary, "add isolated harness execution")
        self.assertFalse(parsed.breaking)

    def test_accepts_breaking_change(self):
        parsed = parse_pr_title("[BREAKING][tasks, docs] refactor: replace task config schema")

        self.assertTrue(parsed.breaking)
        self.assertEqual(parsed.areas, ("tasks", "docs"))

    def test_accepts_numeric_stack_marker(self):
        parsed = parse_pr_title("[12/20][gateway] perf: reduce trajectory materialization overhead")

        self.assertEqual(parsed.series_index, 12)
        self.assertEqual(parsed.series_total, 20)

    def test_accepts_open_ended_stack_marker(self):
        parsed = parse_pr_title("[2/N][framework] test: cover failed sibling rollout handling")

        self.assertEqual(parsed.series_index, 2)
        self.assertEqual(parsed.series_total, "N")

    def test_accepts_all_supported_types(self):
        for change_type in (
            "feat",
            "fix",
            "refactor",
            "perf",
            "test",
            "docs",
            "chore",
            "revert",
        ):
            with self.subTest(change_type=change_type):
                parsed = parse_pr_title(f"[misc] {change_type}: describe the change")
                self.assertEqual(parsed.change_type, change_type)

    def test_rejects_noncanonical_titles(self):
        invalid_titles = (
            "",
            "[Agents] feat: add an agent",
            "[agents,sandbox] feat: omit comma spacing",
            "[agents] feature: use an unsupported type",
            "[agents] fix:no space after colon",
            "[0/N][gateway] fix: use a zero stack index",
            "[1/n][gateway] fix: use a lowercase stack total",
            "[BREAKING] [tasks] feat: add space between prefixes",
            "[1/N] [gateway] refactor: add space between prefixes",
            " [agents] feat: include leading whitespace",
            "[agents] feat: include trailing whitespace ",
        )

        for title in invalid_titles:
            with self.subTest(title=title), self.assertRaises(PRTitleError):
                parse_pr_title(title)

    def test_rejects_removed_pre_refactor_area(self):
        with self.assertRaisesRegex(PRTitleError, "Unknown area.*core"):
            parse_pr_title("[core] fix: use an obsolete area")

    def test_rejects_duplicate_areas(self):
        with self.assertRaisesRegex(PRTitleError, "Duplicate area.*agents"):
            parse_pr_title("[agents, agents] fix: repeat an area")

    def test_rejects_stack_index_greater_than_total(self):
        with self.assertRaisesRegex(PRTitleError, "cannot be greater"):
            parse_pr_title("[3/2][gateway] refactor: split protocol adapters")


class MainTest(unittest.TestCase):
    def test_returns_zero_for_valid_title(self):
        output = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {"PR_TITLE": "[sandbox] fix: preserve command exit codes"},
                clear=True,
            ),
            contextlib.redirect_stdout(output),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertIn("PR title is valid", output.getvalue())

    def test_returns_one_with_github_annotation_for_invalid_title(self):
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"PR_TITLE": "fix sandbox"}, clear=True),
            contextlib.redirect_stdout(output),
        ):
            exit_code = main()

        self.assertEqual(exit_code, 1)
        self.assertTrue(output.getvalue().startswith("::error title=Invalid PR title::"))


class PRTemplateTest(unittest.TestCase):
    def test_documents_every_allowed_area_and_type(self):
        template = (REPOSITORY_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")

        for value in (*ALLOWED_AREAS, *ALLOWED_TYPES):
            with self.subTest(value=value):
                self.assertIn(f"`{value}`", template)


if __name__ == "__main__":
    unittest.main()
