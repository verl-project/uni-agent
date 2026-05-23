# ruff: noqa: E501
"""Lark/Feishu CLI tool definition.

Thin wrapper around the official ``lark-cli`` binary
(https://github.com/larksuite/cli). The actual command string the model
emits is forwarded through the shell verbatim (see the ``lark-cli`` case in
``uni_agent/interaction/tools_manager.py``), so any feature ``lark-cli``
supports on a regular shell -- shortcuts, API commands, raw API, plus
heredocs, command substitution, pipes -- works here too.

The tool is registered under the name ``lark-cli`` (matching the upstream
binary). The containing Python package directory stays ``lark_cli`` because
hyphens are not valid in Python module names.

Authentication is expected to be done *outside* the container by the user
(``lark-cli auth login --recommend``). The local credential file is then
mounted/copied into the runtime environment.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Run a Lark / Feishu CLI command. The string you put in `command` is what
would follow `lark-cli` on the shell, and is executed *through* the shell --
heredocs, command substitution (`$(...)`), pipes and redirects all work.

`lark-cli` exposes Lark/Feishu (calendar, docs, IM, video conference, drive,
bitable, sheets, wiki, contact, mail, task, ...) through three layers:

1. **Shortcuts** (prefixed with `+`, e.g. `calendar +agenda`, `docs +fetch`)
   — high-level, zero-config commands for common scenarios.
2. **API Commands** — `<resource> <action>` style, parameters via `--params`
   (JSON). Mirrors the Lark Open Platform REST endpoints.
3. **Raw API** — `api <METHOD> <PATH> --params <json> --data <json>` for any
   of the 2500+ Lark OpenAPI endpoints.

Examples:
- command = "calendar +agenda"
- command = "docs +fetch --doc \\"https://bytedance.larkoffice.com/docx/...\\""
- command = "api GET /open-apis/calendar/v4/calendars/primary/events"
- command = "calendar events instance_view --params '{\\"calendar_id\\":\\"primary\\",\\"start_time\\":\\"1700000000\\",\\"end_time\\":\\"1700086400\\"}'"
- command = "docs +create --title \\"Draft\\" --markdown \\"$(cat /tmp/draft.md)\\""
  (use `execute_bash` first to write the file to /tmp)

Tips:
- For long markdown bodies, prefer the two-step pattern:
  1) `execute_bash` to write the body to `/tmp/x.md`
  2) `lark-cli docs +create --title ... --markdown "$(cat /tmp/x.md)"`
  This avoids escaping a multi-line string inside JSON.
- Most commands return JSON on stdout. Use `--format pretty|table|csv|ndjson`
  to switch formatting if needed.
- Send messages only with `--as bot`; user-identity send is not supported.
- If the system prompt lists ``lark-*`` Skills (lark-calendar, lark-doc,
  lark-im, lark-vc, ...), `cat` them on demand for the exact command
  vocabulary and recommended workflows.
""".strip()


class LarkCliArguments(BaseModel):
    command: str = Field(
        description=(
            "Arguments to pass to `lark-cli`, written exactly as you would on "
            "the shell. The command is executed through the shell, so `$(...)`, "
            "heredocs, pipes and redirects are all available. "
            "Examples: `calendar +agenda`, "
            '`docs +fetch --doc "https://bytedance.larkoffice.com/docx/..."`, '
            "`api GET /open-apis/calendar/v4/calendars`. "
            "Do NOT include the leading `lark-cli` token; the framework adds it."
        ),
    )


@register_tool("lark-cli")
class LarkCliTool(AbstractTool):
    @property
    def name(self) -> str:
        return "lark-cli"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "lark-cli"

    def get_tool_schema(self) -> dict:
        return self.build_tool_schema(
            description=DESCRIPTION,
            arguments_model=LarkCliArguments,
        )

    def get_install_command(self) -> str | None:
        return (
            "lark-cli --version >/dev/null 2>&1 || ( "
            "echo 'lark-cli not found in PATH. Install with: "
            "npm install -g @larksuite/cli && lark-cli auth login --recommend' >&2; "
            "exit 1 )"
        )
