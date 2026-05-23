"""SkillsManager: discover skills and build the system-prompt manifest.

Mirrors Claude Code / Qwen Code's progressive disclosure model:

- The system prompt only sees a lightweight **manifest** (skill name +
  description + path to its ``SKILL.md``).
- Skill bodies live as real files on disk. The model reads them on demand
  using ``execute_bash`` (e.g. ``cat /opt/uni-agent/skills/<name>/SKILL.md``).
- Each ``SKILL.md`` may reference sibling files (``reference.md``,
  ``examples.md``, ``scripts/...``); those are pushed into the container
  alongside ``SKILL.md`` and read lazily by the model.

Skills are fully **user-managed**: drop the directories you want exposed
into a single host-side dir (e.g. ``~/.uni-agent/skills/``) and point
``SkillsManagerConfig.skills_dir`` at it. Tools themselves bundle no
skills. Path resolution and per-deployment transfer are owned by
``AgentEnv`` (see ``env.install_skills``): host deployments keep skills
in place, container deployments copy them under ``/opt/uni-agent/skills/<name>/``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from uni_agent.skills.base import Skill, parse_skill_md


class SkillsManagerConfig(BaseModel):
    """Configuration for the SkillsManager."""

    skills_dir: Path | None = Field(
        default=None,
        description=(
            "Host-side directory holding ``<skill-name>/SKILL.md`` subdirs. "
            "All discovered skills come from here -- tools ship no skills "
            "of their own. ``~`` is expanded. Example: ``~/.uni-agent/skills``."
        ),
    )
    model_config = ConfigDict(extra="forbid")


class SkillsManager:
    """Skill discovery and manifest assembly. Install is owned by ``AgentEnv``."""

    def __init__(self, config: SkillsManagerConfig, skills: list[Skill]):
        self.config = config
        self.skills: list[Skill] = sorted(skills, key=lambda s: s.name)
        # Populated by ``AgentEnv.install_skills``: skill name -> path inside
        # the runtime (the original host source dir for host runtimes;
        # ``/opt/uni-agent/skills/<name>`` for container runtimes). Empty
        # until install_skills runs; ``build_manifest`` falls back to the
        # host source dir so it stays correct even if called early.
        self.runtime_paths: dict[str, Path] = {}

    @classmethod
    def from_config(cls, config: SkillsManagerConfig) -> SkillsManager:
        """Scan ``config.skills_dir`` for ``<name>/SKILL.md`` subdirs."""
        skills: list[Skill] = []
        if config.skills_dir is not None:
            root = Path(config.skills_dir).expanduser()
            skills = _scan_skills_root(root)
        return cls(config=config, skills=skills)

    def build_manifest(self) -> str:
        """System-prompt block listing every discovered skill.

        XML layout aligned with Claude Code's upstream Agent Skills
        formatter (and OpenClaw's port of it) so models trained against
        either system find it familiar. Empty string when no skills were
        discovered so callers can do ``if manifest: ...``.

        Reads ``self.runtime_paths`` (populated by
        ``AgentEnv.install_skills``) for each skill's in-runtime path,
        falling back to ``skill.source_dir`` if install_skills hasn't run
        yet -- correct for host runtimes, an acceptable best-effort
        otherwise.
        """
        if not self.skills:
            return ""

        lines = [
            "The following skills provide specialized instructions for specific tasks.",
            "Use the read tool to load a skill's file when the task matches its description.",
            "When a skill file references a relative path, resolve it against the skill "
            "directory (parent of SKILL.md / dirname of the path) and use that absolute path "
            "in tool commands.",
            "",
            "<available_skills>",
        ]
        for s in self.skills:
            skill_md = self.runtime_paths.get(s.name, s.source_dir) / "SKILL.md"
            desc = s.description.strip() or "(no description)"
            lines.append("  <skill>")
            lines.append(f"    <name>{_xml_escape(s.name)}</name>")
            lines.append(f"    <description>{_xml_escape(desc)}</description>")
            lines.append(f"    <location>{_xml_escape(skill_md.as_posix())}</location>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)


def _scan_skills_root(root: Path) -> list[Skill]:
    """Find every ``<root>/<name>/SKILL.md`` and parse it."""
    if not root.is_dir():
        return []
    found: list[Skill] = []
    for skill_md in sorted(root.glob("*/SKILL.md")):
        if not skill_md.is_file():
            continue
        found.append(parse_skill_md(skill_md))
    return found


def _xml_escape(value: str) -> str:
    """Escape the five XML special characters in a manifest field."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
