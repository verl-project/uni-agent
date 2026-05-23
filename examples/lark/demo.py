# ruff: noqa: E501
"""Demo: run a reasoning lark agent on a local_native runtime.

Mirrors ``examples/search_arxiv/demo.py`` (full ``AgentInteraction``
reasoning loop) but tailored for lark:

- ``local_native`` deployment (pexpect bash on the host, no container) so
  the already-authenticated ``lark-cli`` Just Works -- no copy, no auth
  shuffle.
- Tools: ``execute_bash`` + ``lark-cli`` + ``finish``. The ``lark-cli``
  install step is now an *assertion* (you must have already done
  ``npm install -g @larksuite/cli`` and ``lark-cli auth login`` on this
  host); see ``uni_agent/tools/lark_cli/__init__.py``.
- Skills: user-managed. Drop the SKILL.md packs you want exposed under
  one directory and point ``SkillsManagerConfig.skills_dir`` at it.
  ``env.install_skills`` registers the runtime path of each skill on the
  ``SkillsManager``; ``interaction.inject_skills_manifest()`` then
  appends an XML manifest (``<available_skills>...``) into the system
  prompt so the model can ``cat`` the right SKILL.md on demand.

Prereqs (on the host, one-time):

1. ``npm install -g @larksuite/cli``
2. ``lark-cli auth login --recommend``
3. (Optional, for skills) ``npx skills add larksuite/cli -y -g`` or
   manually copy any ``<name>/SKILL.md`` packs under ``~/.uni-agent/skills/``.
   Override the directory with ``LARK_SKILLS_DIR=...``.
4. An OpenAI-compatible chat completion endpoint serving a tool-calling
   model (e.g. vLLM hosting Qwen3-Coder). Defaults to ``localhost:8000``.

vllm serve /mnt/hdfs/yyding/models/Qwen3.6-35B-A3B \
  --served-model-name Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --port 8000

Run:

    BASE_URL=http://localhost:8000/v1 \
    MODEL_NAME=Qwen/Qwen3.6-35B-A3B \
    python examples/lark/demo.py
"""

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uni_agent.interaction import (
    AgentEnv,
    AgentEnvConfig,
    AgentInteraction,
    OpenAICompatibleChatModel,
    ToolsManager,
    ToolsManagerConfig,
)
from uni_agent.skills import SkillsManager, SkillsManagerConfig
from uni_agent.tools import ToolConfig

# --- config ----------------------------------------------------------------
run_id = str(uuid.uuid4())

user_request = os.getenv(
    "LARK_USER_REQUEST",
    "帮我整理一下最近 7 天收到的邮件，私聊发到我自己的飞书。",
)

model_base_url = os.getenv("BASE_URL", "http://localhost:8000/v1")
model_api_key = os.getenv("API_KEY", "EMPTY")
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3.6-35B-A3B")

# /usr/local/bin usually needs sudo; default to a user-writable dir.
tool_install_dir = Path(os.getenv("LARK_TOOL_INSTALL_DIR", str(Path.home() / ".uni-agent" / "bin")))
tool_install_dir.mkdir(parents=True, exist_ok=True)

skills_dir = Path(os.getenv("LARK_SKILLS_DIR", str(Path.home() / ".uni-agent" / "skills")))

print("=" * 80)
print("Lark reasoning agent")
print("=" * 80)
print(f"Run ID:           {run_id}")
print(f"User request:     {user_request}")
print(f"Model endpoint:   {model_base_url}")
print(f"Model name:       {model_name}")
print(f"API key configured: {model_api_key != 'EMPTY'}")
print(f"Tool install dir: {tool_install_dir}")
print(f"Skills dir:       {skills_dir} (exists={skills_dir.is_dir()})")

# --- env -------------------------------------------------------------------
env_config = AgentEnvConfig(
    deployment={
        "type": "local_native",
        "startup_timeout": 60.0,
    },
    tool_install_dir=tool_install_dir,
    # NO_COLOR / TERM=dumb keep lark-cli from emitting OSC color queries
    # that pexpect would otherwise echo back into stdout as escape junk.
    env_variables={"NO_COLOR": "1", "TERM": "dumb"},
)
env = AgentEnv(run_id=run_id, env_config=env_config)

# --- tools -----------------------------------------------------------------
tools_manager = ToolsManager(
    ToolsManagerConfig(
        tools=[
            ToolConfig(name="execute_bash"),
            ToolConfig(name="lark-cli"),
            ToolConfig(name="finish"),
        ]
    )
)

# --- skills ----------------------------------------------------------------
skills_manager = SkillsManager.from_config(
    SkillsManagerConfig(skills_dir=skills_dir),
)
print(f"\n[skills] discovered {len(skills_manager.skills)} skill(s): {[s.name for s in skills_manager.skills]}\n")

# --- model -----------------------------------------------------------------
model = OpenAICompatibleChatModel(
    base_url=model_base_url,
    api_key=model_api_key,
    model_name=model_name,
    sampling_params={
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "top_k": 20,
        "repetition_penalty": 1.0,
    },
)
model.set_tools_schemas(tools_manager.tools_schemas)

# --- messages --------------------------------------------------------------
SYSTEM_PROMPT = """You are an agent that helps the user accomplish tasks by calling tools.

# Tools and skills

You have a fixed set of tools available (see the function-calling schema). \
You also have a library of *skills* -- task-specific instruction packs -- \
listed under <available_skills> in this system message. Each skill ships \
with a SKILL.md file describing its command vocabulary, parameters, edge \
cases, and formatting requirements for a particular domain.

When deciding how to act:

- Inspect <available_skills> first. If a skill's description matches the \
  user's intent, read its SKILL.md (e.g. `cat <location>`) BEFORE \
  invoking related commands. SKILL.md is the source of truth for that \
  domain -- do not guess command syntax, flags, or output formatting.
- Prefer the most specific skill over generic shell commands. Only fall \
  back to plain `execute_bash` when no skill applies.
- Run one command at a time, observe its output, then decide the next \
  step. Do not assume what an unread command will return.

# Tool-calling discipline

- Every assistant response MUST contain EXACTLY ONE tool call. Never \
  reply with plain text only, and never emit more than one tool call \
  per response.
- When the user's request is fully satisfied, call `finish` with a \
  short markdown summary of what was done and where the result lives \
  (e.g. "sent DM to self at 14:32").

# Identity

You are the **bot** in this system. The human you are talking to is the \
**user**. When a tool exposes an identity flag (e.g. `lark-cli ... --as \
{bot|user}`):

- Default to `--as bot` for any action that produces output *for* the \
  user (sending them a DM, posting to a chat the user reads, creating \
  files the user will consume, etc.). Sending "as user" in those cases \
  means impersonating the user, which is almost never what they want.
- Use `--as user` only when the action genuinely requires the user's \
  own identity / personal scope (e.g. reading the user's private \
  calendar, mailbox, drafts, drive, OKRs) or when the user explicitly \
  asks you to act as them.
- If unsure which identity to use, prefer `--as bot` first; fall back \
  to `--as user` only after a bot-identity attempt fails with a scope \
  or visibility error.

# Behavior

- Think briefly before each action; keep tool arguments focused on a \
  single concrete step.
- If a command fails with a permission / scope / authentication / \
  missing-credential error, do NOT retry or attempt to re-authenticate. \
  Stop, and call `finish` reporting the error so the user can fix \
  credentials manually.
- Side-effecting actions (sending messages, writing files, deleting \
  data, etc.) must stay within the scope of what the user explicitly \
  asked for. When in doubt, do the read-only version first and confirm \
  before writing.

# Tone

- Be concise. The user wants results, not narration.
- Match the user's language (e.g. reply in Chinese if they wrote in \
  Chinese).
- Final summaries: short, no preambles like "Sure!" or "Of course".
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_request},
]

# --- interaction -----------------------------------------------------------
interaction = AgentInteraction(
    run_id=run_id,
    env=env,
    model=model,
    tools_manager=tools_manager,
    messages=messages,
    skills_manager=skills_manager,
    action_timeout=60,
    max_turns=100,
)


# --- run -------------------------------------------------------------------
print("[1/5] Starting environment...")
env.start()

print("[2/5] Installing tools...")
env.install_tools(tools_manager.tools)
tool_check = env.communicate("which lark-cli execute_bash finish")
print(tool_check.strip())

print("\n[3/5] Installing skills + injecting manifest...")
env.install_skills(skills_manager)
interaction.inject_skills_manifest()

print("\n" + "=" * 80)
print("  Full prompt sent to the model (after manifest injection)")
print("=" * 80)
for i, msg in enumerate(interaction.messages):
    role = msg.get("role", "?")
    content = msg.get("content", "")
    header = f"[message #{i}] role={role}  ({len(content):,} chars)"
    print(f"\n{header}\n{'-' * len(header)}")
    print(content)
print("\n" + "=" * 80)

print("\n[4/5] Running interaction loop...")
result = interaction.run()
trajectory = result["trajectory"]

print("\n[5/5] Final status:")
last_step = trajectory[-1] if trajectory else None
if last_step is not None:
    print(f"  steps:        {len(trajectory)}")
    print(f"  exit_reason:  {last_step.exit_reason}")
    print(f"  done:         {last_step.done}")
else:
    print("  No step output found.")

print("\nFinal result:")
print(last_step.observation if last_step is not None else "(empty)")

env.close()
print("\n[env] closed")
