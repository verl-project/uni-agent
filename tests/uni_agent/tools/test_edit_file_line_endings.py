"""Line-ending tests for ``str_replace_editor`` on the local sandbox."""

import asyncio

import pytest

from uni_agent.sandbox.local import LocalSandbox
from uni_agent.tools.edit_file import EditFileTool

NEWLINES = pytest.mark.parametrize("newline", ("\r\n", "\r", "\n"), ids=("crlf", "cr", "lf"))


def _lines(newline: str, *lines: str) -> bytes:
    return "".join(line + newline for line in lines).encode()


def _run(path, *calls):
    tool = EditFileTool(LocalSandbox())

    async def run():
        return [await tool.run({"path": str(path), **call}) for call in calls]

    return asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.level0
@NEWLINES
def test_str_replace_keeps_line_endings_of_the_file(tmp_path, newline):
    path = tmp_path / "file.txt"
    path.write_bytes(_lines(newline, "alpha", "beta", "gamma"))

    _run(path, {"command": "str_replace", "old_str": "beta", "new_str": "BETA\nbeta2"})

    assert path.read_bytes() == _lines(newline, "alpha", "BETA", "beta2", "gamma")


@pytest.mark.cpu
@pytest.mark.level0
@NEWLINES
def test_insert_keeps_line_endings_of_the_file(tmp_path, newline):
    path = tmp_path / "file.txt"
    path.write_bytes(_lines(newline, "alpha", "beta"))

    _run(path, {"command": "insert", "insert_line": 1, "new_str": "inserted"})

    assert path.read_bytes() == _lines(newline, "alpha", "inserted", "beta")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "original",
    (b"alpha\r\nbeta\r\ngamma\r\n", b"alpha\rbeta\rgamma\r", b"alpha\r\nbeta\ngamma\r\n"),
    ids=("crlf", "cr", "mixed"),
)
@pytest.mark.parametrize(
    "edit",
    (
        {"command": "str_replace", "old_str": "beta", "new_str": "BETA"},
        {"command": "insert", "insert_line": 1, "new_str": "inserted"},
    ),
    ids=("str_replace", "insert"),
)
def test_undo_restores_the_original_bytes(tmp_path, original, edit):
    path = tmp_path / "file.txt"
    path.write_bytes(original)

    _, undo = _run(path, edit, {"command": "undo_edit"})

    assert path.read_bytes() == original
    assert "\r" not in undo.text


@pytest.mark.cpu
@pytest.mark.level0
def test_each_undo_restores_the_bytes_before_its_edit(tmp_path):
    path = tmp_path / "file.txt"
    original = b"alpha\r\nbeta\r\ngamma\r\n"
    path.write_bytes(original)
    tool = EditFileTool(LocalSandbox())

    async def run():
        await tool.run({"command": "str_replace", "path": str(path), "old_str": "beta", "new_str": "BETA"})
        after_first_edit = path.read_bytes()
        await tool.run({"command": "insert", "path": str(path), "insert_line": 0, "new_str": "header"})
        await tool.run({"command": "undo_edit", "path": str(path)})
        assert path.read_bytes() == after_first_edit == b"alpha\r\nBETA\r\ngamma\r\n"
        await tool.run({"command": "undo_edit", "path": str(path)})
        assert path.read_bytes() == original

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.level0
def test_view_of_a_crlf_file_shows_no_carriage_returns(tmp_path):
    path = tmp_path / "file.txt"
    path.write_bytes(b"alpha\r\nbeta\r\n")

    (view,) = _run(path, {"command": "view"})

    assert "\r" not in view.text
    assert "     2\tbeta" in view.text
