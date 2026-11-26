"""``str_replace_editor`` messages and sandbox round trips over the exec-only file floor."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
import pytest_asyncio

from uni_agent.sandbox import LocalSandbox
from uni_agent.sandbox.base import _WRITE_FILE_B64_CHUNK_SIZE, ExecResult, Sandbox
from uni_agent.tools import Toolbox

pytestmark = [pytest.mark.cpu, pytest.mark.level0, pytest.mark.asyncio]


class ExecFloorSandbox(LocalSandbox):
    """Local host commands, but files go through the base64-over-exec floor (like GKE)."""

    read_file = Sandbox.read_file
    write_file = Sandbox.write_file

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    async def exec(self, argv, **kwargs) -> ExecResult:
        self.calls.append(list(argv))
        return await super().exec(argv, **kwargs)


class CountingLocalSandbox(LocalSandbox):
    """Native host read/write channel; only exec round trips are counted."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []
        self.reads = 0

    async def exec(self, argv, **kwargs) -> ExecResult:
        self.calls.append(list(argv))
        return await super().exec(argv, **kwargs)

    async def read_file(self, path: str) -> bytes:
        self.reads += 1
        return await super().read_file(path)


def _make_editor(sandbox_cls):
    @pytest_asyncio.fixture
    async def fixture():
        async with (
            sandbox_cls() as sandbox,
            Toolbox.from_specs([{"name": "str_replace_editor"}], sandbox=sandbox) as tools,
        ):

            async def call(path: Path | str, command: str, status: str = "ok", **arguments) -> tuple[str, int]:
                sandbox.calls.clear()
                result = await tools.call("str_replace_editor", {"path": str(path), "command": command, **arguments})
                assert result.status == status, result.text
                assert result.text is not None
                return result.text, len(sandbox.calls)

            call.sandbox = sandbox
            yield call

    return fixture


floor_editor = _make_editor(ExecFloorSandbox)
native_editor = _make_editor(CountingLocalSandbox)


async def test_view_file_message_and_single_exec(floor_editor, tmp_path):
    path = tmp_path / "a.py"
    path.write_text("one\ntwo\nthree\n")
    text, execs = await floor_editor(path, "view")
    assert text == f"Here's the result of running `cat -n` on {path}:\n     1\tone\n     2\ttwo\n     3\tthree\n"
    assert execs == 1  # was 4: test -e, test -d, test -d, base64

    text, execs = await floor_editor(path, "view", view_range=[2, 3])
    assert text == f"Here's the result of running `cat -n` on {path}:\n     2\ttwo\n     3\tthree\n"
    assert execs == 1

    text, _ = await floor_editor(path, "view", view_range=[5, 6])
    assert text == (
        "Invalid `view_range`: [5, 6]. Its first element `5` should be within the range of lines of the file: [1, 3]"
    )


async def test_view_directory_message_and_two_execs(floor_editor, tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("")
    (tmp_path / ".hidden").write_text("")
    text, execs = await floor_editor(tmp_path, "view")
    assert text.startswith(
        f"Here's the files and directories up to 2 levels deep in `{tmp_path}`, excluding hidden items:\n"
    )
    assert f"{tmp_path}/pkg/m.py" in text
    assert ".hidden" not in text
    assert execs == 2  # was 4: test -e, test -d, test -d, find

    text, execs = await floor_editor(tmp_path, "view", view_range=[1, 2])
    assert text == "The `view_range` parameter is not allowed when `path` points to a directory."
    assert execs == 1


async def test_missing_path_messages(floor_editor, tmp_path):
    missing = tmp_path / "missing.py"
    for command, extra in [
        ("view", {}),
        ("str_replace", {"old_str": "a", "new_str": "b"}),
        ("insert", {"insert_line": 0, "new_str": "x"}),
        ("undo_edit", {}),
    ]:
        text, execs = await floor_editor(missing, command, **extra)
        assert text == f"The path `{missing}` does not exist. Please provide a valid path."
        assert execs == 1

    text, _ = await floor_editor(tmp_path, "str_replace", old_str="a", new_str="b")
    assert text == f"The path `{tmp_path}` is a directory and only the `view` command can be used on directories"


async def test_create_messages_and_execs(floor_editor, tmp_path):
    path = tmp_path / "new.py"
    text, execs = await floor_editor(path, "create", file_text="hello\n")
    assert text == f"File created successfully at: {path}"
    assert path.read_text() == "hello\n"
    assert execs == 2  # was 3: test -e, test -d parent, write

    text, execs = await floor_editor(path, "create", file_text="again\n")
    assert text == f"File already exists at: `{path}`. Cannot overwrite files using command `create`."
    assert execs == 1

    orphan = tmp_path / "nope" / "x.py"
    text, execs = await floor_editor(orphan, "create", file_text="x")
    assert text == f"The parent directory {orphan.parent} does not exist. Please create it first."
    assert execs == 1

    text, execs = await floor_editor(f"{tmp_path}/dir/", "create", file_text="x")
    assert text == f"The path `{tmp_path}/dir/` ends with `/`. The `create` command requires a file path."
    assert execs == 0


async def test_str_replace_messages_and_execs(floor_editor, tmp_path):
    path = tmp_path / "s.py"
    path.write_text("a = 1\nb = 2\na = 1\nc = 3\n")

    text, execs = await floor_editor(path, "str_replace", old_str="zzz", new_str="y")
    assert text == f"No replacement was performed, old_str `zzz` did not appear verbatim in {path}."
    assert execs == 1

    text, execs = await floor_editor(path, "str_replace", old_str="a = 1", new_str="y")
    assert text == (
        "No replacement was performed. Multiple occurrences of old_str `a = 1` "
        "(first two start on lines 1 and 3). Please ensure it is unique"
    )
    assert execs == 1

    text, execs = await floor_editor(path, "str_replace", old_str="b = 2", new_str="b = 20")
    assert text == (
        f"The file {path} has been edited. Here's the result of running `cat -n` on a snippet of {path}:\n"
        "     1\ta = 1\n     2\tb = 20\n     3\ta = 1\n     4\tc = 3\n"
        "Review the changes and make sure they are as expected. Edit the file again if necessary."
    )
    assert path.read_text() == "a = 1\nb = 20\na = 1\nc = 3\n"
    assert execs == 2  # was 4: test -e, test -d, base64, write

    text, execs = await floor_editor(path, "undo_edit")
    assert text == (
        f"Last edit to {path} undone successfully. Here's the result of running `cat -n` on {path}:\n"
        "     1\ta = 1\n     2\tb = 2\n     3\ta = 1\n     4\tc = 3\n"
    )
    assert path.read_text() == "a = 1\nb = 2\na = 1\nc = 3\n"
    assert execs == 2  # was 3: test -e, test -d, write

    text, execs = await floor_editor(path, "undo_edit")
    assert text == f"No edit history found for {path}."
    assert execs == 1


async def test_insert_message_and_execs(floor_editor, tmp_path):
    path = tmp_path / "i.py"
    path.write_text("x\ny\n")
    text, execs = await floor_editor(path, "insert", insert_line=1, new_str="mid")
    assert text == (
        f"The file {path} has been edited. Here's the result of running `cat -n` on a snippet of the edited file:\n"
        "     1\tx\n     2\tmid\n     3\ty\n"
        "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). "
        "Edit the file again if necessary."
    )
    assert path.read_text() == "x\nmid\ny\n"
    assert execs == 2  # was 4

    text, execs = await floor_editor(path, "insert", insert_line=9, new_str="z")
    assert text == "Invalid `insert_line` parameter: 9. It should be within the range of lines of the file: [0, 3]"
    assert execs == 1


async def test_non_utf8_file_error(floor_editor, tmp_path):
    binary = tmp_path / "b.bin"
    binary.write_bytes(b"\xff\xfe\x00")
    text, execs = await floor_editor(binary, "view", status="error")
    assert f"The file `{binary}` is not valid UTF-8 text." in text
    assert execs == 1


async def test_large_file_round_trips(floor_editor, tmp_path):
    path = tmp_path / "big.txt"
    body = "".join(f"line {i:06d}\n" for i in range(20000))  # ~240 KB: chunked write
    path.write_text(body)
    text, execs = await floor_editor(path, "str_replace", old_str="line 010000\n", new_str="LINE\n")
    assert "has been edited" in text
    assert path.read_text() == body.replace("line 010000\n", "LINE\n")
    payload = len(base64.b64encode(path.read_bytes()))
    chunks = -(-payload // _WRITE_FILE_B64_CHUNK_SIZE)
    assert execs == 1 + chunks + 1  # probe+read, chunk stages, decode (rm folded in)


async def test_native_read_file_is_not_bypassed(native_editor, tmp_path):
    path = tmp_path / "n.py"
    path.write_text("p\nq\n")
    text, execs = await native_editor(path, "str_replace", old_str="q", new_str="r")
    assert "has been edited" in text
    assert path.read_text() == "p\nr\n"
    assert execs == 1  # probe only; read and write use the provider's native channel
    assert native_editor.sandbox.reads == 1
