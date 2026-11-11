"""Exercise str_replace_editor on UTF-8 files through its public toolbox interface."""

from pathlib import Path

import pytest
import pytest_asyncio

from uni_agent.sandbox import LocalSandbox
from uni_agent.tools import Toolbox
from uni_agent.tools.edit_file import MAX_RESPONSE_LEN

pytestmark = [pytest.mark.cpu, pytest.mark.level0, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def editor():
    async with (
        LocalSandbox() as sandbox,
        Toolbox.from_specs([{"name": "str_replace_editor"}], sandbox=sandbox) as tools,
    ):

        async def call(path: Path, command: str, **arguments) -> str:
            result = await tools.call("str_replace_editor", {"path": str(path), "command": command, **arguments})
            assert result.status == "ok", result.text
            assert result.text is not None
            return result.text

        yield call


async def assert_undo_restores(editor, path, original):
    assert "undone successfully" in await editor(path, "undo_edit")
    assert path.read_bytes() == original


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r"], ids=["lf", "crlf", "cr"])
@pytest.mark.parametrize("terminated", [True, False], ids=["final-newline", "no-final-newline"])
async def test_replace_makefile_comment_preserves_recipes(editor, tmp_path, ending, terminated):
    path = tmp_path / "Makefile"
    eof = ending if terminated else b""
    original = (
        ending.join(
            [
                b"# Build the release archive.",
                b".PHONY: package",
                b"",
                b"package:",
                b"\t@mkdir -p dist",
                b"\t@tar -czf dist/source.tar.gz src/",
            ]
        )
        + eof
    )
    expected = (
        ending.join(
            [
                b"# Package the source tree.",
                b".PHONY: package",
                b"",
                b"package:",
                b"\t@mkdir -p dist",
                b"\t@tar -czf dist/source.tar.gz src/",
            ]
        )
        + eof
    )
    path.write_bytes(original)

    result = await editor(
        path, "str_replace", old_str="# Build the release archive.", new_str="# Package the source tree."
    )

    assert "has been edited" in result
    assert path.read_bytes() == expected
    await assert_undo_restores(editor, path, original)


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r"], ids=["lf", "crlf", "cr"])
async def test_replace_lines_copied_from_numbered_view(editor, tmp_path, ending):
    path = tmp_path / "backup.sh"
    original = ending.join(
        [
            b"#!/bin/sh",
            b"set -eu",
            "# 归档当前工作目录".encode(),
            b'if [ -d "$source" ]; then',
            b'\tmkdir -p "$backup"',
            b'\tcp -R "$source" "$backup/latest"',
            b"fi",
            b"",
        ]
    )
    expected = ending.join(
        [
            b"#!/bin/sh",
            b"set -eu",
            "# 归档当前工作目录".encode(),
            b'if [ -d "$source" ]; then',
            b'\tmkdir -p "$backup/daily"',
            b'\tcp -R "$source" "$backup/daily/latest"',
            b'\tprintf "%s\\n" "backup complete"',
            b"fi",
            b"",
        ]
    )
    path.write_bytes(original)

    view = await editor(path, "view", view_range=[5, 6])
    rows = view.split("\n")[1:-1]
    assert rows == ['     5\t\tmkdir -p "$backup"', '     6\t\tcp -R "$source" "$backup/latest"']
    assert path.read_bytes() == original
    copied = "\n".join(row.split("\t", 1)[1] for row in rows)
    replacement = """\
\tmkdir -p "$backup/daily"
\tcp -R "$source" "$backup/daily/latest"
\tprintf "%s\\n" "backup complete"\
"""
    result = await editor(path, "str_replace", old_str=copied, new_str=replacement)

    assert "has been edited" in result
    assert "\r" not in result
    assert path.read_bytes() == expected
    await assert_undo_restores(editor, path, original)


async def test_tab_and_space_indentation_select_different_functions(editor, tmp_path):
    path = tmp_path / "limits.py"
    original = b"""\
def batch_limit():
\t# Read once per batch.
\treturn 30

def request_limit():
        return 30
"""
    expected = b"""\
def batch_limit():
\t# Read once per batch.
\treturn 30

def request_limit():
        return 15
"""
    path.write_bytes(original)

    rejected = await editor(path, "str_replace", old_str="        # Read once per batch.", new_str="        # Cache.")
    assert "No replacement was performed" in rejected and "tabs" in rejected
    assert path.read_bytes() == original
    assert "No edit history" in await editor(path, "undo_edit")

    result = await editor(path, "str_replace", old_str="        return 30", new_str="        return 15")
    assert "has been edited" in result
    assert path.read_bytes() == expected


async def test_repeated_ini_settings_need_section_context_in_mixed_file(editor, tmp_path):
    path = tmp_path / "deploy.ini"
    original = b"""\
[staging]
retries = 3
timeout = 30

[production]\r
retries = 3\r
timeout = 30\r
"""
    expected = b"""\
[staging]
retries = 1
timeout = 10

[production]\r
retries = 3\r
timeout = 30\r
"""
    path.write_bytes(original)

    view = await editor(path, "view", view_range=[2, 3])
    copied = "\n".join(row.split("\t", 1)[1] for row in view.split("\n")[1:-1])
    assert copied == "retries = 3\ntimeout = 30"
    result = await editor(path, "str_replace", old_str=copied, new_str="retries = 1\ntimeout = 10")
    assert "Multiple occurrences" in result
    assert path.read_bytes() == original
    assert "No edit history" in await editor(path, "undo_edit")

    result = await editor(
        path, "str_replace", old_str="[staging]\n" + copied, new_str="[staging]\nretries = 1\ntimeout = 10"
    )
    assert "has been edited" in result
    assert path.read_bytes() == expected
    await assert_undo_restores(editor, path, original)


async def test_replacement_reuses_each_line_ending_inside_mixed_block(editor, tmp_path):
    path = tmp_path / "report.ini"
    original = (
        "# 导出配置\r\n"
        "[format]\r\n"
        "type = csv\n"
        "header = true\r"
        "encoding = utf-8\r\n"
        "[destination]\n"
        "path = exports/report.csv"
    ).encode()
    expected = (
        "# 导出配置\r\n"
        "[format]\r\n"
        "type = csv\n"
        "header = false\r"
        "encoding = utf-8\r"
        "delimiter = comma\r\n"
        "[destination]\n"
        "path = exports/report.csv"
    ).encode()
    path.write_bytes(original)

    result = await editor(
        path,
        "str_replace",
        old_str="type = csv\nheader = true\nencoding = utf-8",
        new_str="type = csv\nheader = false\nencoding = utf-8\ndelimiter = comma",
    )
    assert "has been edited" in result
    assert path.read_bytes() == expected
    await assert_undo_restores(editor, path, original)


async def test_deleting_sql_comment_between_cr_and_lf_keeps_blank_line(editor, tmp_path):
    path = tmp_path / "orders.sql"
    original = (
        b"SELECT order_id, total\r\n"
        b"FROM orders\r"
        b"-- temporary filter placeholder\n"
        b"WHERE total > 0\r\n"
        b"ORDER BY order_id;\r\n"
    )
    expected = b"SELECT order_id, total\r\nFROM orders\r\n\nWHERE total > 0\r\nORDER BY order_id;\r\n"
    path.write_bytes(original)

    result = await editor(path, "str_replace", old_str="-- temporary filter placeholder", new_str="")
    assert "has been edited" in result
    assert path.read_bytes() == expected
    view = await editor(path, "view", view_range=[3, 4])
    assert view.split("\n")[1:-1] == ["     3\t", "     4\tWHERE total > 0"]
    await assert_undo_restores(editor, path, original)


@pytest.mark.parametrize("recipe_ending", [b"\n", b"\r\n", b"\r"], ids=["mixed-lf", "crlf", "mixed-cr"])
async def test_insert_makefile_recipes_uses_ending_at_insertion_point(editor, tmp_path, recipe_ending):
    path = tmp_path / "Makefile"
    # The header and clean target are CRLF. Only the publish block uses recipe_ending.
    original = (
        b"# Release targets\r\n"
        b".PHONY: publish clean\r\n"
        + recipe_ending.join([b"publish:", b"\t@echo published", b"", b""])
        + b"clean:\r\n\t@rm -f dist/manifest\r\n"
    )
    expected = (
        b"# Release targets\r\n"
        b".PHONY: publish clean\r\n"
        + recipe_ending.join(
            [b"publish:", b"\t@mkdir -p dist", b"\t@cp manifest dist/manifest", b"\t@echo published", b"", b""]
        )
        + b"clean:\r\n\t@rm -f dist/manifest\r\n"
    )
    path.write_bytes(original)

    result = await editor(path, "insert", insert_line=3, new_str="\t@mkdir -p dist\n\t@cp manifest dist/manifest")
    assert "has been edited" in result
    assert path.read_bytes() == expected
    await assert_undo_restores(editor, path, original)


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r"], ids=["lf", "crlf", "cr"])
@pytest.mark.parametrize(
    ("last_line", "terminated"),
    [(b"idna==3.10", True), (b"idna==3.10", False), (b"", True)],
    ids=["final-newline", "no-final-newline", "last-line-blank"],
)
async def test_insert_after_last_requirement(editor, tmp_path, ending, last_line, terminated):
    path = tmp_path / "requirements.txt"
    eof = ending if terminated else b""
    original = ending.join([b"# HTTP client dependencies", b"httpx==0.28.1", last_line]) + eof
    expected = (
        ending.join([b"# HTTP client dependencies", b"httpx==0.28.1", last_line, b"# Test dependencies", b"pytest>=8"])
        + ending
    )
    path.write_bytes(original)

    expected_rows = ["     1\t# HTTP client dependencies", "     2\thttpx==0.28.1", "     3\t" + last_line.decode()]
    assert (await editor(path, "view")).split("\n")[1:-1] == expected_rows
    assert (await editor(path, "view", view_range=[3, 3])).split("\n")[1:-1] == expected_rows[2:3]
    past_end = len(expected_rows) + 1
    assert "Invalid" in await editor(path, "view", view_range=[past_end, -1])
    assert "Invalid" in await editor(path, "insert", insert_line=past_end, new_str="pytest>=8")
    assert path.read_bytes() == original
    assert "No edit history" in await editor(path, "undo_edit")

    result = await editor(path, "insert", insert_line=3, new_str="# Test dependencies\npytest>=8")
    assert "has been edited" in result
    assert path.read_bytes() == expected
    view = await editor(path, "view", view_range=[4, -1])
    expected_tail = ["     4\t# Test dependencies", "     5\tpytest>=8"]
    assert view.split("\n")[1:-1] == expected_tail
    await assert_undo_restores(editor, path, original)


@pytest.mark.parametrize("ending", ["\r\n", "\n", "\r"], ids=["crlf", "lf", "cr"])
@pytest.mark.parametrize(
    ("original_lines", "insert_line", "new_text", "expected_lines"),
    [
        (["苹果", "香蕉", ""], 0, "牛奶", ["牛奶", "苹果", "香蕉", ""]),
        (["苹果", "香蕉", ""], 1, "牛奶", ["苹果", "牛奶", "香蕉", ""]),
        (["苹果", "香蕉", ""], 2, "牛奶", ["苹果", "香蕉", "牛奶", ""]),
        (["苹果", "香蕉"], 2, "牛奶", ["苹果", "香蕉", "牛奶", ""]),
        (["苹果", "香蕉", "", ""], 3, "牛奶", ["苹果", "香蕉", "", "牛奶", ""]),
        (["苹果", "香蕉", ""], 2, "牛奶\n", ["苹果", "香蕉", "牛奶", "", ""]),
    ],
    ids=["prepend", "middle", "append", "append-unterminated", "after-blank", "insert-with-newline"],
)
async def test_insert_after_a_shopping_list_line(
    editor, tmp_path, ending, original_lines, insert_line, new_text, expected_lines
):
    path = tmp_path / "shopping.txt"
    original = ending.join(original_lines).encode()
    expected = ending.join(expected_lines).encode()
    path.write_bytes(original)

    result = await editor(path, "insert", insert_line=insert_line, new_str=new_text)
    assert "has been edited" in result
    assert path.read_bytes() == expected
    await assert_undo_restores(editor, path, original)


async def test_final_crlf_does_not_create_a_third_shopping_list_line(editor, tmp_path):
    path = tmp_path / "shopping.txt"
    original = "苹果\r\n香蕉\r\n".encode()
    path.write_bytes(original)
    assert (await editor(path, "view")).split("\n")[1:-1] == ["     1\t苹果", "     2\t香蕉"]
    assert "Invalid" in await editor(path, "view", view_range=[3, 3])
    assert "Invalid" in await editor(path, "insert", insert_line=3, new_str="牛奶")
    assert path.read_bytes() == original
    assert "No edit history" in await editor(path, "undo_edit")


async def test_insert_first_rules_into_empty_gitignore(editor, tmp_path):
    path = tmp_path / ".gitignore"
    original = b""
    expected = b"# Python build output\n__pycache__/\n*.pyc\ndist/\n"
    path.write_bytes(original)

    assert (await editor(path, "view")).split("\n")[1:-1] == []
    assert "Invalid" in await editor(path, "view", view_range=[1, 1])
    assert "Invalid" in await editor(path, "insert", insert_line=1, new_str="dist/")
    assert path.read_bytes() == original
    result = await editor(path, "insert", insert_line=0, new_str="# Python build output\n__pycache__/\n*.pyc\ndist/")
    assert "has been edited" in result
    assert path.read_bytes() == expected
    await assert_undo_restores(editor, path, original)


async def test_clipped_build_log_ends_at_a_complete_crlf(editor, tmp_path):
    path = tmp_path / "build.log"
    # A long artifact listing puts the first CR right at the raw truncation boundary.
    listing = "INFO archive contents: " + ", ".join(f"dist/module-{n:04d}.whl" for n in range(1000))
    first_line = listing[: MAX_RESPONSE_LEN - 1]
    original = (first_line + "\r\nINFO archive uploaded\r\n").encode()
    expected_prefix = f"     1\t{first_line}\n     2\t<response clipped>"
    path.write_bytes(original)

    view = await editor(path, "view")
    assert view.partition("\n")[2].startswith(expected_prefix)
    assert "\r" not in view
    assert "INFO archive uploaded" not in view
    assert path.read_bytes() == original
