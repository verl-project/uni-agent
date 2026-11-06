"""``str_replace_editor`` must not rewrite the tabs of lines it does not edit."""

import asyncio
from types import SimpleNamespace

import pytest

from uni_agent.tools.edit_file import EditFileTool

PATH = "/testbed/Makefile"
MAKEFILE = "# before\nall:\n\t@echo OK\n"


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


def _run(sandbox: _MemorySandbox, **args):
    tool = EditFileTool(sandbox)
    return tool, asyncio.run(tool.run({"path": PATH, **args}))


@pytest.mark.cpu
@pytest.mark.level0
def test_str_replace_keeps_tabs_of_untouched_lines():
    sandbox = _MemorySandbox(MAKEFILE)
    _run(sandbox, command="str_replace", old_str="# before", new_str="# after")
    assert sandbox.files[PATH] == b"# after\nall:\n\t@echo OK\n"


@pytest.mark.cpu
@pytest.mark.level0
def test_insert_keeps_tabs_of_existing_lines():
    sandbox = _MemorySandbox(MAKEFILE)
    _run(sandbox, command="insert", insert_line=0, new_str="# added")
    assert sandbox.files[PATH] == b"# added\n# before\nall:\n\t@echo OK\n"


@pytest.mark.cpu
@pytest.mark.level0
def test_undo_restores_original_tabs():
    sandbox = _MemorySandbox(MAKEFILE)
    tool = EditFileTool(sandbox)
    asyncio.run(tool.run({"path": PATH, "command": "str_replace", "old_str": "# before", "new_str": "# after"}))
    asyncio.run(tool.run({"path": PATH, "command": "undo_edit"}))
    assert sandbox.files[PATH] == MAKEFILE.encode("utf-8")


@pytest.mark.cpu
@pytest.mark.level0
def test_str_replace_still_matches_text_copied_from_the_tab_expanded_view():
    """The view shows tabs as spaces, so an old_str copied from it must still match."""
    sandbox = _MemorySandbox(MAKEFILE)
    expanded_recipe = "        @echo OK"  # the tab, expanded to 8 columns
    _, result = _run(sandbox, command="str_replace", old_str=expanded_recipe, new_str="        @echo DONE")
    assert "has been edited" in result.text
    assert b"DONE" in sandbox.files[PATH]
