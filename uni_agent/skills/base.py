"""Skill abstraction: one ``SKILL.md`` + its sibling files on disk.

A skill is a *directory* on the host filesystem containing:

- ``SKILL.md`` (required): YAML frontmatter (``name``, ``description``) plus
  markdown body. Auto-read by the model on demand via ``cat``.
- Any number of sibling files (``reference.md``, ``examples.md``,
  ``scripts/...``, ``templates/...``): NOT pre-loaded; SKILL.md references
  them by relative path and the model decides when to read or execute them.

This module only parses skills off disk; pushing them into the container
and assembling the manifest prompt live in ``manager.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    """One discovered skill directory.

    Attributes:
        name: canonical identifier, falls back to the directory name if
            no ``name:`` field is present in frontmatter.
        description: from ``description:`` in frontmatter. Should describe
            *what* the skill is and *when* to use it -- this is what the
            model sees in the manifest.
        body: SKILL.md markdown body after the frontmatter block. Loaded
            on demand by the model, not pushed into the system prompt.
        source_dir: host-side directory holding SKILL.md and its siblings.
        origin: short tag describing where the skill came from (e.g.
            ``"tool:lark-cli"``, ``"user"``), used purely for logging /
            ordering in the manifest.
    """

    name: str
    description: str
    body: str
    source_dir: Path
    origin: str = ""

    @property
    def skill_md_path(self) -> Path:
        return self.source_dir / "SKILL.md"


def parse_skill_md(skill_md_path: Path, origin: str = "") -> Skill:
    """Load and parse a single ``SKILL.md`` file from disk."""
    text = skill_md_path.read_text(encoding="utf-8")
    description, name_override, body = _split_frontmatter(text)
    name = name_override or skill_md_path.parent.name
    return Skill(
        name=name,
        description=description,
        body=body,
        source_dir=skill_md_path.parent,
        origin=origin,
    )


def _split_frontmatter(text: str) -> tuple[str, str, str]:
    """Parse the YAML frontmatter at the top of ``text``.

    Returns ``(description, name, body)``. Missing fields come back as ``""``.
    Only ``name`` and ``description`` are recognised today; ``paths:`` and
    other Qwen / Claude fields are silently ignored for forward compatibility.

    Uses ``yaml.safe_load`` so block scalars (``description: |``), quoted
    values containing colons, and other valid YAML constructs all work.
    """
    import yaml

    stripped = text.lstrip()
    if not stripped.startswith("---"):
        # No frontmatter -- treat the whole file as body, no metadata.
        return "", "", text.lstrip("\n")

    end = stripped.find("\n---", 3)
    if end == -1:
        # Malformed frontmatter (no closing ---); treat as body, log nothing.
        return "", "", text.lstrip("\n")

    frontmatter = stripped[3:end].strip()
    body = stripped[end + 4 :].lstrip("\n")

    try:
        meta = yaml.safe_load(frontmatter)
    except yaml.YAMLError:
        meta = None
    if not isinstance(meta, dict):
        meta = {}

    description = str(meta.get("description") or "").strip()
    name = str(meta.get("name") or "").strip()
    return description, name, body
