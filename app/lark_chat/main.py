# ruff: noqa: E501
"""Long-running Lark chat agent — entrypoint.

Bootstraps a single sandbox env + model client, then enters a loop that
consumes inbound IM events from ``lark-cli event consume
im.message.receive_v1`` and dispatches each non-self message to a
multi-step ``AgentInteraction`` run. Conversation history is persisted
per ``chat_id`` so each chat is a real ongoing conversation across
turns and process restarts.

See ``app/lark_chat/README.md`` for setup, env vars, and run examples.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.lark_chat import prompts  # noqa: E402
from app.lark_chat.listener import LarkEventListener, fetch_bot_open_id  # noqa: E402
from app.lark_chat.transcript import TranscriptStore  # noqa: E402
from uni_agent.interaction import (  # noqa: E402
    AgentEnv,
    AgentEnvConfig,
    AgentInteraction,
    OpenAICompatibleChatModel,
    ToolsManager,
    ToolsManagerConfig,
)
from uni_agent.skills import SkillsManager, SkillsManagerConfig  # noqa: E402
from uni_agent.tools import ToolConfig  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.local_native.yaml"

_DEFAULT_LOCAL_ATTACH_MEMORY_DIR = Path("/workspace/memory")
_DEFAULT_LOCAL_NATIVE_MEMORY_DIR = Path("~/.uni-agent/app/lark_chat/memory").expanduser()
_DEFAULT_LOCAL_NATIVE_TOOL_INSTALL_DIR = Path("~/.uni-agent/bin").expanduser()


@dataclass
class SwerexConfig:
    host: str
    port: int
    auth_token: str


@dataclass
class LocalAttachDeployment:
    """Bash session + ``lark-cli`` live in a user-managed Docker container;
    the host listener routes through ``docker exec -i <container>``."""

    container: str
    swerex: SwerexConfig
    post_setup_cmd: str | None = "cd /workspace"
    timeout: float = 300.0
    startup_timeout: float = 30.0


@dataclass
class LocalNativeDeployment:
    """Pexpect bash directly on the host; reuses host-side ``lark-cli``."""

    startup_timeout: float = 60.0
    tool_install_dir: Path = field(default_factory=lambda: _DEFAULT_LOCAL_NATIVE_TOOL_INSTALL_DIR)


DeploymentConfig = LocalAttachDeployment | LocalNativeDeployment


@dataclass
class ModelConfig:
    base_url: str
    name: str
    api_key: str
    sampling_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentLoopConfig:
    action_timeout: int = 60
    max_steps_per_turn: int = 20
    history_max_tokens: int = 128000
    """Compaction trigger; history is forwarded unchanged while it fits under this."""
    history_target_tokens: int = 32000
    """Post-compaction size. Must be ``<= history_max_tokens``."""

    def __post_init__(self) -> None:
        if self.history_target_tokens > self.history_max_tokens:
            raise ValueError(
                f"agent.history_target_tokens ({self.history_target_tokens}) must be "
                f"<= agent.history_max_tokens ({self.history_max_tokens})."
            )


@dataclass
class LarkChatConfig:
    """Runtime config for the Lark chat agent.

    Loaded from YAML via :meth:`load`. Secrets (``deployment.swerex.auth_token``
    for ``local_attach``, ``model.api_key``) may be left ``null`` in YAML and
    supplied through ``LOCAL_ATTACH_AUTH_TOKEN`` / ``API_KEY`` env vars instead.
    """

    deployment: DeploymentConfig
    memory_dir: Path
    model: ModelConfig
    tools: list[str]
    skills_dir: Path
    transcripts_dir: Path
    agent: AgentLoopConfig

    @classmethod
    def load(cls, path: Path) -> LarkChatConfig:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = yaml.safe_load(path.read_text()) or {}

        deployment_raw = raw.get("deployment") or {}
        dep_type = deployment_raw.get("type") or "local_attach"
        if dep_type == "local_attach":
            swerex_raw = deployment_raw.get("swerex") or {}
            auth_token = swerex_raw.get("auth_token") or os.environ.get("LOCAL_ATTACH_AUTH_TOKEN")
            if not auth_token:
                raise RuntimeError(
                    "deployment.swerex.auth_token is required for local_attach: set it in the "
                    "config file or via the LOCAL_ATTACH_AUTH_TOKEN env var (must match the "
                    "--auth-token passed to swerex.server inside the container)."
                )
            if "container" not in deployment_raw:
                raise RuntimeError("deployment.container is required for local_attach")
            deployment: DeploymentConfig = LocalAttachDeployment(
                container=deployment_raw["container"],
                swerex=SwerexConfig(
                    host=swerex_raw["host"],
                    port=int(swerex_raw["port"]),
                    auth_token=auth_token,
                ),
                post_setup_cmd=deployment_raw.get("post_setup_cmd", "cd /workspace"),
                timeout=float(deployment_raw.get("timeout", 300.0)),
                startup_timeout=float(deployment_raw.get("startup_timeout", 30.0)),
            )
            default_memory_dir = _DEFAULT_LOCAL_ATTACH_MEMORY_DIR
        elif dep_type == "local_native":
            tool_install_dir = Path(
                deployment_raw.get("tool_install_dir") or _DEFAULT_LOCAL_NATIVE_TOOL_INSTALL_DIR
            ).expanduser()
            deployment = LocalNativeDeployment(
                startup_timeout=float(deployment_raw.get("startup_timeout", 60.0)),
                tool_install_dir=tool_install_dir,
            )
            default_memory_dir = _DEFAULT_LOCAL_NATIVE_MEMORY_DIR
        else:
            raise ValueError(
                f"deployment.type={dep_type!r} is not supported; expected 'local_attach' or 'local_native'."
            )

        memory_raw = raw.get("memory_dir")
        if memory_raw is None:
            memory_dir = default_memory_dir
        else:
            memory_path = Path(memory_raw)
            # local_attach memory_dir is a container path; don't expand ~ to a host path.
            memory_dir = memory_path.expanduser() if isinstance(deployment, LocalNativeDeployment) else memory_path

        model_raw = raw.get("model") or {}
        api_key = model_raw.get("api_key") or os.environ.get("API_KEY") or "EMPTY"

        return cls(
            deployment=deployment,
            memory_dir=memory_dir,
            model=ModelConfig(
                base_url=model_raw["base_url"],
                name=model_raw["name"],
                api_key=api_key,
                sampling_params=dict(model_raw.get("sampling_params") or {}),
            ),
            tools=list(raw.get("tools") or []),
            skills_dir=Path(raw["skills_dir"]).expanduser(),
            transcripts_dir=Path(raw["transcripts_dir"]).expanduser(),
            agent=AgentLoopConfig(**(raw.get("agent") or {})),
        )

    def lark_cli_prefix(self) -> list[str]:
        """argv prefix wrapping ``lark-cli`` calls (``docker exec -i <container>``
        for ``local_attach``, empty for ``local_native``)."""
        if isinstance(self.deployment, LocalAttachDeployment):
            return ["docker", "exec", "-i", self.deployment.container]
        return []

    def build_env_config(self) -> AgentEnvConfig:
        if isinstance(self.deployment, LocalAttachDeployment):
            d = self.deployment
            return AgentEnvConfig(
                deployment={
                    "type": "local_attach",
                    "host": d.swerex.host,
                    "port": d.swerex.port,
                    "auth_token": d.swerex.auth_token,
                    "timeout": d.timeout,
                    "startup_timeout": d.startup_timeout,
                },
                env_variables={"NO_COLOR": "1", "TERM": "dumb"},
                post_setup_cmd=d.post_setup_cmd,
            )

        native = self.deployment
        return AgentEnvConfig(
            deployment={
                "type": "local_native",
                "startup_timeout": native.startup_timeout,
            },
            env_variables={"NO_COLOR": "1", "TERM": "dumb"},
            tool_install_dir=native.tool_install_dir,
        )


def _get_token_counter() -> Any:
    """``(text) -> int`` token counter; tiktoken ``cl100k_base`` when available,
    else ``len(text) // 4``. Cached on the function attribute."""
    cached = getattr(_get_token_counter, "_cached", None)
    if cached is not None:
        return cached
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        def count(text: str) -> int:
            return len(enc.encode(text or "", disallowed_special=()))
    except Exception:

        def count(text: str) -> int:
            return max(1, len(text or "") // 4)

    _get_token_counter._cached = count  # type: ignore[attr-defined]
    return count


def _message_tokens(msg: dict, count: Any) -> int:
    n = 0
    content = msg.get("content")
    if isinstance(content, str):
        n += count(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                n += count(part.get("text") or "")
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        n += count(fn.get("name") or "")
        n += count(fn.get("arguments") or "")
    if isinstance(msg.get("name"), str):
        n += count(msg["name"])
    return n + 4


def compact_history(
    messages: list[dict],
    *,
    max_tokens: int,
    target_tokens: int,
) -> list[dict]:
    """Hysteresis history compaction.

    Returns ``messages`` unchanged while total tokens fit under ``max_tokens``;
    otherwise trims non-system messages in user-anchored chunks down to
    ``target_tokens``. Chunks are atomic (so a ``role=tool`` is never separated
    from its parent ``role=assistant`` -- OpenAI rejects that with a 400 on
    ``tool_call_id`` linkage). The most-recent chunk is always kept.
    """
    if max_tokens <= 0 or target_tokens <= 0:
        return messages

    count = _get_token_counter()

    sys_msgs: list[dict] = []
    rest: list[dict] = []
    rest_costs: list[int] = []
    sys_cost = 0
    for m in messages:
        cost = _message_tokens(m, count)
        if m.get("role") == "system":
            sys_msgs.append(m)
            sys_cost += cost
        else:
            rest.append(m)
            rest_costs.append(cost)

    if sys_cost + sum(rest_costs) <= max_tokens:
        return messages

    user_idxs = [i for i, m in enumerate(rest) if m.get("role") == "user"]
    if not user_idxs:
        return messages

    chunks: list[tuple[list[dict], int]] = []
    for i, start in enumerate(user_idxs):
        end = user_idxs[i + 1] if i + 1 < len(user_idxs) else len(rest)
        chunks.append((rest[start:end], sum(rest_costs[start:end])))

    budget = max(0, target_tokens - sys_cost)
    kept_rev: list[list[dict]] = []
    used = 0
    for chunk, cost in reversed(chunks):
        if not kept_rev or used + cost <= budget:
            kept_rev.append(chunk)
            used += cost
        else:
            break

    return sys_msgs + [m for chunk in reversed(kept_rev) for m in chunk]


async def handle_one_message(
    event: dict,
    *,
    env: AgentEnv,
    model: OpenAICompatibleChatModel,
    tools_manager: ToolsManager,
    skills_manager: SkillsManager,
    store: TranscriptStore,
    config: LarkChatConfig,
) -> None:
    chat_id = event.get("chat_id")
    message_id = event.get("message_id")
    sender_id = event.get("sender_id")
    if not (chat_id and message_id and sender_id):
        print(f"⚠️  skipping malformed event (missing chat_id/message_id/sender_id): {event!r}")
        return

    chat_type = event.get("chat_type", "?")
    message_type = event.get("message_type", "?")
    content = event.get("content", "")
    create_time = event.get("create_time")

    print(f"\n{'━' * 70}")
    print(f"📨 [{chat_id}] msg={message_id} from={sender_id} {chat_type}/{message_type}")
    preview = content.strip().splitlines()[0] if content.strip() else "(empty)"
    print(f"   {preview[:120]}")

    persisted = store.load(chat_id)
    first_turn = not persisted

    if first_turn:
        messages: list[dict] = [
            {"role": "system", "content": prompts.build_system_prompt(memory_dir=config.memory_dir)}
        ]
    else:
        messages = compact_history(
            persisted,
            max_tokens=config.agent.history_max_tokens,
            target_tokens=config.agent.history_target_tokens,
        )

    messages.append(
        {
            "role": "user",
            "content": prompts.format_user_message(
                chat_id=chat_id,
                message_id=message_id,
                sender_id=sender_id,
                chat_type=chat_type,
                message_type=message_type,
                content=content,
                create_time=create_time,
            ),
        }
    )

    run_id = str(uuid.uuid4())
    interaction = AgentInteraction(
        run_id=run_id,
        env=env,
        model=model,
        tools_manager=tools_manager,
        messages=messages,
        skills_manager=skills_manager if first_turn else None,
        action_timeout=config.agent.action_timeout,
        max_turns=config.agent.max_steps_per_turn,
        chat_mode=True,
    )
    if first_turn:
        interaction.inject_skills_manifest()

    pre_run_len = len(messages)
    try:
        result = await interaction.run()
    except Exception:
        # interaction.messages is mutated in place; persist what we have
        store.save(chat_id, interaction.messages)
        raise

    store.save(chat_id, result["messages"])

    trajectory = result["trajectory"]
    new_asst_msgs = [m for m in result["messages"][pre_run_len:] if m.get("role") == "assistant"]
    asst_iter = iter(new_asst_msgs)
    for step in trajectory:
        # run()'s synthetic terminator (max_step_limit / unknown_error)
        # has no matching assistant message -- skip the iterator advance.
        is_terminator = (
            step.exit_reason in ("max_step_limit", "unknown_error") and not step.tool_results and not step.response
        )
        if is_terminator:
            print(f"   [step {step.step_idx}] exit={step.exit_reason} (loop terminator)")
            continue

        asst = next(asst_iter, None)
        attempted = asst.get("tool_calls", []) if asst else []
        # Anything in `attempted` missing from executed_status was
        # rejected by parse_structured_action (unknown name / bad args).
        executed_status = {tr.tool_call_id: tr.status for tr in step.tool_results}

        if attempted:
            for tc in attempted:
                name = tc["function"]["name"]
                status = executed_status.get(tc["id"], "rejected")
                print(f"   [step {step.step_idx}] tool={name}, status={status}")
        else:
            preview = (step.response or "").strip().splitlines()
            preview_str = preview[0][:120] if preview else "(empty)"
            print(f"   [step {step.step_idx}] no tool_call, exit={step.exit_reason or '?'} → {preview_str}")

    last_step = trajectory[-1] if trajectory else None
    if last_step is not None:
        print(f"   ✓ turn done in {len(trajectory)} step(s); exit={last_step.exit_reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-running Lark chat agent.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    config = LarkChatConfig.load(args.config)

    print("=" * 80)
    print("Lark chat agent (multi-turn, multi-tool)")
    print("=" * 80)
    print(f"config:        {args.config}")
    if isinstance(config.deployment, LocalAttachDeployment):
        print("deployment:    local_attach")
        print(f"container:     {config.deployment.container}")
        print(f"swerex:        {config.deployment.swerex.host}:{config.deployment.swerex.port}")
    else:
        print("deployment:    local_native (pexpect on host, no container)")
        print(f"tool install:  {config.deployment.tool_install_dir}")
    print(f"memory dir:    {config.memory_dir}  (inside runtime bash)")
    print(f"model:         {config.model.name} @ {config.model.base_url}")
    print(f"tools:         {config.tools}")
    print(f"skills dir:    {config.skills_dir} (exists={config.skills_dir.is_dir()})")
    print(f"transcripts:   {config.transcripts_dir}")

    lark_cli_prefix = config.lark_cli_prefix()
    prefix_str = " ".join(lark_cli_prefix) + " " if lark_cli_prefix else ""
    print(f"lark-cli via:  {prefix_str}lark-cli ...")

    print("\n[1/6] Resolving bot open_id via Lark Open API...")
    bot_open_id = await fetch_bot_open_id(command_prefix=lark_cli_prefix)
    print(f"  bot open_id: {bot_open_id}")

    print("\n[2/6] Starting sandbox env...")
    run_id = str(uuid.uuid4())
    env = AgentEnv(run_id=run_id, env_config=config.build_env_config())
    await env.start()
    print("  env started")

    print("\n[3/6] Installing tools + skills...")
    tools_manager = ToolsManager(
        ToolsManagerConfig(tools=[ToolConfig(name=name) for name in config.tools]),
    )
    await env.install_tools(tools_manager.tools)

    skills_manager = SkillsManager.from_config(SkillsManagerConfig(skills_dir=config.skills_dir))
    await env.install_skills(skills_manager)
    print(f"  {len(skills_manager.skills)} skill(s): {[s.name for s in skills_manager.skills]}")

    notes_dir = config.memory_dir / "notes"
    await env.communicate(f"mkdir -p {shlex.quote(str(notes_dir))}", check="raise")

    print("\n[4/6] Wiring model client...")
    model = OpenAICompatibleChatModel(
        base_url=config.model.base_url,
        api_key=config.model.api_key,
        model_name=config.model.name,
        sampling_params=config.model.sampling_params,
    )
    model.set_tools_schemas(tools_manager.tools_schemas)

    store = TranscriptStore(base_dir=config.transcripts_dir)
    print(f"  transcript store: {store.base_dir}")

    print("\n[5/6] Starting Lark event listener...")
    listener = LarkEventListener(
        event_key="im.message.receive_v1",
        as_identity="bot",
        jq=(f'select(.sender_id != "{bot_open_id}") | select(.message_type == "text" or .message_type == "post")'),
        command_prefix=lark_cli_prefix,
    )
    await listener.start()
    print("  listener ready")

    print("\n[6/6] Entering chat loop. Send a Lark message to the bot. Ctrl+C to stop.\n")

    try:
        async for event in listener:
            try:
                await handle_one_message(
                    event,
                    env=env,
                    model=model,
                    tools_manager=tools_manager,
                    skills_manager=skills_manager,
                    store=store,
                    config=config,
                )
            except Exception:
                print("✗ message handler failed:")
                print(traceback.format_exc())
                continue
    except KeyboardInterrupt:
        print("\n[shutdown] keyboard interrupt")
    finally:
        print("\n[shutdown] stopping listener and env...")
        try:
            await listener.stop()
        except Exception as e:
            print(f"  listener stop error: {e}")
        try:
            await env.close()
        except Exception as e:
            print(f"  env close error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
