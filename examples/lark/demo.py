# ruff: noqa: E501
"""Demo: run the ``lark-cli`` tool + user-managed skills on a local runtime.

Mirrors ``examples/agent_env/demo.py`` but:

- uses the ``local_native`` deployment (pexpect bash session on the host,
  no container) so the already-authenticated ``lark-cli`` Just Works;
- installs the ``lark-cli`` tool (the install step is now an *assertion*
  -- you must have ``npm install -g @larksuite/cli`` + ``lark-cli auth
  login`` done beforehand);
- points ``SkillsManagerConfig.skills_dir`` at a host-side directory you
  curate (here: ``~/.agents/skills`` since that's where ``npx skills add
  larksuite/cli -y -g`` lays out the 27 lark-* SKILL.md packs);
- prints the skills manifest that would be injected into the system
  prompt and ``cat``s one SKILL.md to show progressive disclosure;
- runs a couple of ``lark-cli`` commands through the env.

Prereqs (on the host, all one-time):

1. ``npm install -g @larksuite/cli``
2. ``lark-cli auth login --recommend``
3. (Optional, for skills) ``npx skills add larksuite/cli -y -g`` -- drops
   ~27 ``lark-*`` SKILL.md dirs into ``~/.agents/skills/``. Substitute
   any other dir you want; just point ``skills_dir`` at it below.

Run:

    python examples/lark/demo.py
"""

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uni_agent.interaction import AgentEnv, AgentEnvConfig
from uni_agent.skills import SkillsManager, SkillsManagerConfig
from uni_agent.tools import ToolConfig

# --- env config ------------------------------------------------------------
run_id = str(uuid.uuid4())

deployment_config = {
    "type": "local_native",
    "startup_timeout": 60.0,
}

# /usr/local/bin usually needs sudo. Override to a user-writable dir.
tool_install_dir = Path(os.getenv("LARK_TOOL_INSTALL_DIR", str(Path.home() / ".uni-agent" / "bin")))
tool_install_dir.mkdir(parents=True, exist_ok=True)

env_config = AgentEnvConfig(
    deployment=deployment_config,
    tool_install_dir=tool_install_dir,
    env_variables={"NO_COLOR": "1", "TERM": "dumb"},
)
env = AgentEnv(run_id=run_id, env_config=env_config)
env.start()

print(f"[env] local_native runtime ready (tool_install_dir={tool_install_dir})\n")

# --- tools -----------------------------------------------------------------
tools_config = [
    {"name": "execute_bash"},
    {"name": "lark-cli"},
]
tools = [ToolConfig(**tc).get_tool() for tc in tools_config]
env.install_tools(tools)

out = env.communicate("which lark-cli")
print(f"[tool] which lark-cli\n  -> {out.strip()}\n")

# --- skills (entirely user-managed) ----------------------------------------
# Drop the SKILL.md packs you want exposed under one directory; the tool
# itself ships none. Override with $LARK_SKILLS_DIR if you want a custom path.
skills_dir = Path(os.getenv("LARK_SKILLS_DIR", str(Path.home() / ".uni-agent" / "skills")))
skills_manager = SkillsManager.from_config(
    SkillsManagerConfig(skills_dir=skills_dir),
)
print(f"[skills] skills_dir={skills_dir} (exists={skills_dir.is_dir()})")
print(f"[skills] discovered {len(skills_manager.skills)} skill(s): "
      f"{[s.name for s in skills_manager.skills]}\n")

# For local_native runtime, env.install_skills is a no-op (skills are read
# in place from the host filesystem). For container runtimes it copies
# each skill dir to ``/opt/uni-agent/skills/<name>/``.
env.install_skills(skills_manager)

print("=" * 60)
print("  Skills manifest (this goes into the system prompt)")
print("=" * 60)
print(skills_manager.build_manifest())
print()

if skills_manager.skills:
    first = skills_manager.skills[0]
    skill_md = skills_manager.runtime_paths[first.name].as_posix() + "/SKILL.md"
    print(f"[skills] runtime path for {first.name!r} -> {skill_md}")
    print("[skills] first 6 lines of SKILL.md:\n")
    head = env.communicate(f"head -n 6 {skill_md}")
    print(head)

# --- run a couple of lark-cli commands -------------------------------------
print("=" * 60)
print("  lark-cli smoke test")
print("=" * 60)

print("\n[lark-cli] --version")
print(env.communicate("lark-cli --version").strip())

# Comment out if auth is not set up.
print("\n[lark-cli] calendar +agenda (today)")
print(env.communicate("lark-cli calendar +agenda 2>&1 | head -n 40").strip())

env.close()
print("\n[env] closed")
