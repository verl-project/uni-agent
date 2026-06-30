"""``str_replace_editor``: view / create / edit files over the sandbox data plane.

A host-side file editor (view / create / str_replace / insert / undo_edit) driven
entirely through the sandbox data plane:

* File I/O goes through ``sandbox.read_file`` / ``sandbox.write_file`` instead of
  the local filesystem, so the same tool drives any provider (local, Modal, ...).
* **Undo history lives on the tool instance** (``self._history``), held by the
  :class:`~uni_agent.tools.base.Toolbox` for the rollout -- there is no
  ``/root/editor_state.json`` written into the container.

The five commands and their messages are kept stable so the model sees consistent
behaviour across rollouts. Optional ``filemap`` / window-expansion niceties are
intentionally dropped to keep this layer dependency-free; because the tool runs in
the harness they can be re-added later without touching the task image.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .base import Observation, Tool, ToolError, register_tool

if TYPE_CHECKING:
    from ..sandbox import SandboxBackend

SNIPPET_LINES = 4
MAX_RESPONSE_LEN = 16000
TRUNCATED_MESSAGE = (
    "<response clipped><NOTE>To save on context only part of this file has been shown to you. "
    "You should retry this tool after you have searched inside the file "
    "with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
)

DESCRIPTION = """
Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
""".strip()  # noqa: E501


class StrReplaceEditorArguments(BaseModel):
    command: str = Field(
        description="The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",  # noqa: E501
        json_schema_extra={"enum": ["view", "create", "str_replace", "insert", "undo_edit"]},
    )
    path: str = Field(description="Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.")
    file_text: str = Field(
        default=None, description="Required parameter of `create` command, with the content of the file to be created."
    )
    old_str: str = Field(
        default=None,
        description="Required parameter of `str_replace` command containing the string in `path` to replace.",
    )
    new_str: str = Field(
        default=None,
        description="Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",  # noqa: E501
    )
    insert_line: int = Field(
        default=None,
        description="Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",  # noqa: E501
    )
    view_range: list[int] = Field(
        default=None,
        description="Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",  # noqa: E501
    )


def _maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN) -> str:
    if not truncate_after or len(content) <= truncate_after:
        return content
    return content[:truncate_after] + TRUNCATED_MESSAGE


def _make_output(file_content: str, file_descriptor: str, init_line: int = 1) -> str:
    """``cat -n`` style numbering (1-based ``init_line``), truncated if huge."""
    file_content = _maybe_truncate(file_content).expandtabs()
    lines = file_content.split("\n")
    numbered = "\n".join(f"{i + init_line:6}\t{line}" for i, line in enumerate(lines))
    return f"Here's the result of running `cat -n` on {file_descriptor}:\n{numbered}\n"


@register_tool("str_replace_editor")
class EditFileTool(Tool):
    name = "str_replace_editor"
    description = DESCRIPTION
    args_model = StrReplaceEditorArguments

    def __init__(self, sandbox: SandboxBackend, **kwargs: Any) -> None:
        super().__init__(sandbox, **kwargs)
        # remote path -> stack of previous contents (newest last); the editor's
        # undo state, kept here in the harness rather than inside the container.
        self._history: dict[str, list[str]] = defaultdict(list)

    # ----- data-plane helpers -----
    @staticmethod
    async def _exists(sandbox: SandboxBackend, path: str) -> bool:
        return (await sandbox.exec(["test", "-e", path])).exit_code == 0

    @staticmethod
    async def _is_dir(sandbox: SandboxBackend, path: str) -> bool:
        return (await sandbox.exec(["test", "-d", path])).exit_code == 0

    @staticmethod
    async def _read(sandbox: SandboxBackend, path: str) -> str:
        data = await sandbox.read_file(path)
        return data.decode("utf-8", errors="replace")

    # ----- dispatch -----
    async def run(self, args: dict[str, Any]) -> Observation:
        return Observation(text=await self._apply(args))

    async def _apply(self, args: dict[str, Any]) -> str:
        sandbox = self.sandbox
        command = args.get("command")
        path = args.get("path")
        if not command:
            raise ToolError("Parameter `command` is required.")
        if not path:
            raise ToolError("Parameter `path` is required.")

        invalid = await self._validate_path(command, sandbox, path)
        if invalid is not None:
            return invalid

        if command == "view":
            return await self._view(sandbox, path, args.get("view_range"))
        if command == "create":
            return await self._create(sandbox, path, args.get("file_text"))
        if command == "str_replace":
            return await self._str_replace(sandbox, path, args.get("old_str"), args.get("new_str"))
        if command == "insert":
            return await self._insert(sandbox, path, args.get("insert_line"), args.get("new_str"))
        if command == "undo_edit":
            return await self._undo_edit(sandbox, path)
        return (
            f"Unrecognized command `{command}`. "
            "The allowed commands for the str_replace_editor are: "
            "`view`, `create`, `str_replace`, `insert`, `undo_edit`"
        )

    async def _validate_path(self, command: str, sandbox: SandboxBackend, path: str) -> str | None:
        exists = await self._exists(sandbox, path)
        is_dir = await self._is_dir(sandbox, path) if exists else False
        if not exists and command != "create":
            return f"The path `{path}` does not exist. Please provide a valid path."
        if exists and command == "create":
            return f"File already exists at: `{path}`. Cannot overwrite files using command `create`."
        if is_dir and command != "view":
            return f"The path `{path}` is a directory and only the `view` command can be used on directories"
        return None

    # ----- commands -----
    async def _view(self, sandbox: SandboxBackend, path: str, view_range: list[int] | None) -> str:
        if await self._is_dir(sandbox, path):
            if view_range:
                return "The `view_range` parameter is not allowed when `path` points to a directory."
            res = await sandbox.exec(["find", path, "-maxdepth", "2", "-not", "-path", "*/.*"])
            if res.exit_code != 0:
                return f"Failed to list directory `{path}`: {res.stderr.strip()}"
            return (
                f"Here's the files and directories up to 2 levels deep in `{path}`, "
                f"excluding hidden items:\n{res.stdout}\n"
            )

        file_content = await self._read(sandbox, path)
        init_line = 1
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                return "Invalid `view_range`. It should be a list of two integers."
            file_lines = file_content.split("\n")
            n_lines = len(file_lines)
            init_line, final_line = view_range
            if init_line < 1 or init_line > n_lines:
                return (
                    f"Invalid `view_range`: {view_range}. Its first element `{init_line}` should be "
                    f"within the range of lines of the file: {[1, n_lines]}"
                )
            if final_line > n_lines:
                return (
                    f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should be "
                    f"smaller than the number of lines in the file: `{n_lines}`"
                )
            if final_line != -1 and final_line < init_line:
                return (
                    f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should be "
                    f"larger or equal than its first `{init_line}`"
                )
            final_line = n_lines if final_line == -1 else final_line
            file_content = "\n".join(file_lines[init_line - 1 : final_line])

        return _make_output(file_content, str(path), init_line=init_line)

    async def _create(self, sandbox: SandboxBackend, path: str, file_text: str | None) -> str:
        if file_text is None:
            return "Parameter `file_text` is required for command: create"
        parent = os.path.dirname(path.rstrip("/")) or "/"
        if not await self._is_dir(sandbox, parent):
            return f"The parent directory {parent} does not exist. Please create it first."
        await sandbox.write_file(path, file_text)
        self._history[path].append(file_text)
        return f"File created successfully at: {path}"

    async def _str_replace(
        self, sandbox: SandboxBackend, path: str, old_str: str | None, new_str: str | None
    ) -> str:
        if old_str is None:
            return "Parameter `old_str` is required for command: str_replace"
        file_content = (await self._read(sandbox, path)).expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str is not None else ""

        occurrences = file_content.count(old_str)
        if occurrences == 0:
            return f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
        if occurrences > 1:
            lines = [idx + 1 for idx, line in enumerate(file_content.split("\n")) if old_str in line]
            return (
                f"No replacement was performed. Multiple occurrences of old_str `{old_str}` "
                f"in lines {lines}. Please ensure it is unique"
            )
        if new_str == old_str:
            return f"No replacement was performed, old_str `{old_str}` is the same as new_str `{new_str}`."

        new_file_content = file_content.replace(old_str, new_str)
        await sandbox.write_file(path, new_file_content)
        self._history[path].append(file_content)

        replacement_line = file_content.split(old_str)[0].count("\n")
        start_line = max(1, replacement_line - SNIPPET_LINES)
        end_line = min(
            replacement_line + SNIPPET_LINES + new_str.count("\n"),
            len(new_file_content.splitlines()),
        )
        snippet = "\n".join(new_file_content.split("\n")[start_line - 1 : end_line])

        success_msg = f"The file {path} has been edited. "
        success_msg += _make_output(snippet, f"a snippet of {path}", start_line)
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."
        return success_msg

    async def _insert(
        self, sandbox: SandboxBackend, path: str, insert_line: int | None, new_str: str | None
    ) -> str:
        if insert_line is None:
            return "Parameter `insert_line` is required for command: insert"
        if new_str is None:
            return "Parameter `new_str` is required for command: insert"

        file_text = (await self._read(sandbox, path)).expandtabs()
        new_str = new_str.expandtabs()
        file_lines = file_text.split("\n")
        n_lines = len(file_lines)
        insert_line -= 1  # convert to 0-based index

        if insert_line < 0 or insert_line > n_lines:
            return (
                f"Invalid `insert_line` parameter: {insert_line}. "
                f"It should be within the range of lines of the file: {[0, n_lines]}"
            )

        new_str_lines = new_str.split("\n")
        new_file_lines = file_lines[:insert_line] + new_str_lines + file_lines[insert_line:]
        snippet_lines = (
            file_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        await sandbox.write_file(path, "\n".join(new_file_lines))
        self._history[path].append(file_text)

        success_msg = f"The file {path} has been edited. "
        success_msg += _make_output(
            "\n".join(snippet_lines),
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += (
            "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). "
            "Edit the file again if necessary."
        )
        return success_msg

    async def _undo_edit(self, sandbox: SandboxBackend, path: str) -> str:
        if not self._history.get(path):
            return f"No edit history found for {path}."
        old_text = self._history[path].pop()
        await sandbox.write_file(path, old_text)
        return f"Last edit to {path} undone successfully. {_make_output(old_text, str(path))}"
