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
- Refer to the bundled Skills (lark-calendar, lark-docs, lark-im, lark-vc, ...)
  for the exact command vocabulary and recommended workflows.
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

    @property
    def skills_dir(self) -> Path:
        """Local directory holding bundled SKILL.md files for this tool.

        Skills are NOT pushed into the container. They are loaded by the agent
        loop on the host side and injected into the model's system prompt
        (see ``get_skills`` below). This mirrors how Claude Code consumes the
        upstream `larksuite/cli` skills under ``~/.claude/skills/``.
        """
        return Path(__file__).parent / "skills"

    def get_skills(self) -> list[dict]:
        """Load bundled SKILL.md files for prompt injection.

        Returns a list of ``{"name": str, "description": str, "body": str}``
        dicts, one per skill, sorted by filename for determinism.
        Returns an empty list if the skills directory does not exist yet.
        """
        skills: list[dict] = []
        if not self.skills_dir.is_dir():
            return skills

        for skill_path in sorted(self.skills_dir.glob("*/SKILL.md")):
            name = skill_path.parent.name
            text = skill_path.read_text(encoding="utf-8")
            description, body = _split_skill_frontmatter(text)
            skills.append({"name": name, "description": description, "body": body})
        return skills

    def get_tool_schema(self) -> dict:
        return self.build_tool_schema(
            description=DESCRIPTION,
            arguments_model=LarkCliArguments,
        )

    def get_install_command(self) -> str | None:
        # Best-effort install. Assumes the runtime has node/npm available.
        # Authentication (`lark-cli auth login`) must be performed beforehand
        # on the host, and ~/.config/larksuite (or equivalent) must already be
        # populated when the tool is invoked.
        return (
            "lark-cli --version >/dev/null 2>&1 "
            "|| (npm install -g @larksuite/cli "
            "&& chmod +x \"$(npm root -g)/@larksuite/cli/scripts/run.js\" 2>/dev/null || true)"
        )


def _split_skill_frontmatter(text: str) -> tuple[str, str]:
    """Parse a SKILL.md file into (description, body).

    Supports both Claude-Code-style YAML frontmatter
    (``---\\ndescription: ...\\n---\\n<body>``) and a fallback where the first
    non-empty line is treated as the description.
    """
    stripped = text.lstrip()
    if stripped.startswith("---"):
        end = stripped.find("\n---", 3)
        if end != -1:
            frontmatter = stripped[3:end].strip()
            body = stripped[end + 4 :].lstrip("\n")
            description = ""
            for line in frontmatter.splitlines():
                if line.lower().startswith("description:"):
                    description = line.split(":", 1)[1].strip().strip("'\"")
                    break
            return description, body
    first_line, _, rest = text.partition("\n")
    return first_line.strip("# ").strip(), rest.lstrip("\n")
