"""``str_replace_editor`` must keep the formatting of a file: tabs, line endings, the final newline."""

import asyncio
from types import SimpleNamespace

import pytest

from uni_agent.tools.edit_file import EditFileTool

PATH = "/testbed/Makefile"
MAKEFILE = "# before\nall:\n\t@echo OK\n"
MAKEFILE_CRLF = "# before\r\nall:\r\n\t@echo OK\r\n"


class _MemorySandbox:
    """Just enough sandbox for the editor: one file that exists and is not a directory."""

    def __init__(self, content: str) -> None:
        self.files = {PATH: content.encode("utf-8")}

    async def exec(self, argv, **kwargs):
        # `test -e path` succeeds, `test -d path` fails: the path is a file.
        return SimpleNamespace(exit_code=1 if argv[:2] == ["test", "-d"] else 0, stdout="", stderr="")

    async def read_file(self, path: str) -> bytes:
        return self.files[path]

    async def write_file(self, path: str, content) -> None:
        self.files[path] = content.encode("utf-8") if isinstance(content, str) else content


def _call(tool: EditFileTool, **args) -> str:
    return asyncio.run(tool.run({"path": PATH, **args})).text


def _edit(content: str, **args) -> tuple[_MemorySandbox, EditFileTool, str]:
    sandbox = _MemorySandbox(content)
    tool = EditFileTool(sandbox)
    return sandbox, tool, _call(tool, **args)


# ----- view -----


@pytest.mark.cpu
@pytest.mark.level0
def test_view_shows_tabs_as_they_are():
    _, _, text = _edit(MAKEFILE, command="view")
    assert "\t@echo OK" in text
    assert "        @echo OK" not in text


@pytest.mark.cpu
@pytest.mark.level0
def test_view_does_not_show_the_carriage_return_of_crlf_lines():
    sandbox, _, text = _edit(MAKEFILE_CRLF, command="view")
    assert "\r" not in text
    assert sandbox.files[PATH] == MAKEFILE_CRLF.encode("utf-8")


# ----- str_replace -----


@pytest.mark.cpu
@pytest.mark.level0
def test_str_replace_keeps_tabs_of_untouched_lines():
    sandbox, _, _ = _edit(MAKEFILE, command="str_replace", old_str="# before", new_str="# after")
    assert sandbox.files[PATH] == b"# after\nall:\n\t@echo OK\n"


@pytest.mark.cpu
@pytest.mark.level0
def test_str_replace_matches_and_writes_tabs_exactly():
    sandbox, _, result = _edit(MAKEFILE, command="str_replace", old_str="\t@echo OK", new_str="\t@echo DONE\n\t@true")
    assert "has been edited" in result
    assert sandbox.files[PATH] == b"# before\nall:\n\t@echo DONE\n\t@true\n"


@pytest.mark.cpu
@pytest.mark.level0
def test_str_replace_does_not_confuse_spaces_with_tabs():
    """Spaces in old_str must not match a tab-indented line, even if both look the same when tabs are expanded."""
    content = "\tvalue = 1\n        value = 1\n"
    sandbox, _, _ = _edit(content, command="str_replace", old_str="        value = 1", new_str="        value = 2")
    assert sandbox.files[PATH] == b"\tvalue = 1\n        value = 2\n"


@pytest.mark.cpu
@pytest.mark.level0
def test_str_replace_explains_a_tab_versus_space_mismatch_and_changes_nothing():
    sandbox, tool, result = _edit(MAKEFILE, command="str_replace", old_str="        @echo OK", new_str="x")
    assert "No replacement was performed" in result
    assert "tabs" in result
    assert sandbox.files[PATH] == MAKEFILE.encode("utf-8")
    assert "No edit history" in _call(tool, command="undo_edit")


@pytest.mark.cpu
@pytest.mark.level0
def test_str_replace_keeps_crlf_line_endings():
    sandbox, _, result = _edit(
        MAKEFILE_CRLF, command="str_replace", old_str="all:\n\t@echo OK", new_str="all:\n\t@echo DONE\n\t@true"
    )
    assert "has been edited" in result
    assert sandbox.files[PATH] == b"# before\r\nall:\r\n\t@echo DONE\r\n\t@true\r\n"


@pytest.mark.cpu
@pytest.mark.level0
def test_str_replace_keeps_a_missing_final_newline():
    sandbox, _, _ = _edit("a\nb", command="str_replace", old_str="b", new_str="c")
    assert sandbox.files[PATH] == b"a\nc"


# ----- insert -----


@pytest.mark.cpu
@pytest.mark.level0
def test_insert_keeps_tabs_of_existing_and_inserted_lines():
    sandbox, _, _ = _edit(MAKEFILE, command="insert", insert_line=2, new_str="\t@echo FIRST")
    assert sandbox.files[PATH] == b"# before\nall:\n\t@echo FIRST\n\t@echo OK\n"


@pytest.mark.cpu
@pytest.mark.level0
def test_insert_keeps_crlf_line_endings():
    sandbox, _, _ = _edit(MAKEFILE_CRLF, command="insert", insert_line=1, new_str="# one\n# two")
    assert sandbox.files[PATH] == b"# before\r\n# one\r\n# two\r\nall:\r\n\t@echo OK\r\n"


# ----- undo_edit -----


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "content",
    [MAKEFILE, MAKEFILE_CRLF, "# before\nall:\n\t@echo OK", "# before\r\nall:\r\n\t@echo OK"],
    ids=["tabs", "crlf", "no-final-newline", "crlf-no-final-newline"],
)
@pytest.mark.parametrize(
    "edit",
    [
        {"command": "str_replace", "old_str": "# before", "new_str": "# after"},
        {"command": "insert", "insert_line": 1, "new_str": "x"},
    ],
    ids=["str_replace", "insert"],
)
def test_undo_restores_the_exact_bytes(content, edit):
    sandbox = _MemorySandbox(content)
    tool = EditFileTool(sandbox)
    _call(tool, **edit)
    assert sandbox.files[PATH] != content.encode("utf-8")

    result = _call(tool, command="undo_edit")

    assert "undone successfully" in result
    assert sandbox.files[PATH] == content.encode("utf-8")
