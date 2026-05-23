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

import asyncio
import os
import sys
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.lark_chat import prompts  # noqa: E402
from app.lark_chat.conversation import ConversationStore  # noqa: E402
from app.lark_chat.listener import LarkEventListener, fetch_bot_open_id  # noqa: E402
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


@dataclass
class Settings:
    """Runtime config sourced from environment variables.

    Deployment is always ``local_attach`` -- the agent's bash session +
    its ``lark-cli`` both live inside a user-managed Docker container,
    so identity stays consistent between event subscription and reply.
    """

    container: str
    swerex_host: str
    swerex_port: int
    swerex_auth_token: str

    model_base_url: str
    model_api_key: str
    model_name: str
    sampling_params: dict

    skills_dir: Path
    history_dir: Path

    action_timeout: int = 60
    max_steps_per_turn: int = 20
    max_history_turns: int = 30

    @classmethod
    def from_env(cls) -> Settings:
        auth_token = os.environ.get("LOCAL_ATTACH_AUTH_TOKEN")
        if not auth_token:
            raise RuntimeError(
                "LOCAL_ATTACH_AUTH_TOKEN is required (matches the --auth-token "
                "you started swerex.server with inside the container)."
            )
        return cls(
            container=os.getenv("LOCAL_ATTACH_CONTAINER", "lark-chat-sandbox"),
            swerex_host=os.getenv("LOCAL_ATTACH_HOST", "http://127.0.0.1"),
            swerex_port=int(os.getenv("LOCAL_ATTACH_PORT", "18000")),
            swerex_auth_token=auth_token,
            model_base_url=os.getenv("BASE_URL", "http://localhost:8000/v1"),
            model_api_key=os.getenv("API_KEY", "EMPTY"),
            model_name=os.getenv("MODEL_NAME", "Qwen/Qwen3.6-35B-A3B"),
            sampling_params={
                "temperature": float(os.getenv("MODEL_TEMPERATURE", "1.0")),
                "top_p": float(os.getenv("MODEL_TOP_P", "0.95")),
                "presence_penalty": float(os.getenv("MODEL_PRESENCE_PENALTY", "1.5")),
                "top_k": int(os.getenv("MODEL_TOP_K", "20")),
                "repetition_penalty": float(os.getenv("MODEL_REPETITION_PENALTY", "1.0")),
            },
            skills_dir=Path(os.getenv("LARK_SKILLS_DIR", str(Path.home() / ".uni-agent" / "skills"))),
            history_dir=Path(
                os.getenv(
                    "LARK_CHAT_HISTORY_DIR",
                    str(Path.home() / ".uni-agent" / "app" / "lark_chat"),
                )
            ),
            action_timeout=int(os.getenv("LARK_CHAT_ACTION_TIMEOUT", "60")),
            max_steps_per_turn=int(os.getenv("LARK_CHAT_MAX_STEPS_PER_TURN", "20")),
            max_history_turns=int(os.getenv("LARK_CHAT_MAX_HISTORY_TURNS", "30")),
        )

    def lark_cli_prefix(self) -> list[str]:
        """argv prefix routing host-side ``lark-cli`` calls (listener +
        bot open_id lookup) into the same container the agent uses, so
        every lark-cli call shares one identity / one auth.
        """
        return ["docker", "exec", "-i", self.container]

    def build_env_config(self) -> AgentEnvConfig:
        return AgentEnvConfig(
            deployment={
                "type": "local_attach",
                "host": self.swerex_host,
                "port": self.swerex_port,
                "auth_token": self.swerex_auth_token,
                "timeout": 300.0,
                "startup_timeout": 30.0,
            },
            env_variables={"NO_COLOR": "1", "TERM": "dumb"},
            post_setup_cmd="cd /workspace",
        )


def trim_history(messages: list[dict], max_user_turns: int) -> list[dict]:
    """Keep the system message + the most-recent ``max_user_turns``
    user-anchored chunks. Trimming respects chunk boundaries so a
    ``role=tool`` is never separated from its parent ``role=assistant``
    (the OpenAI API rejects that with a 400 on ``tool_call_id`` linkage).
    """
    user_idxs = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_idxs) <= max_user_turns:
        return messages
    cutoff = user_idxs[-max_user_turns]
    head = [m for m in messages[:cutoff] if m.get("role") == "system"]
    return head + messages[cutoff:]


async def handle_one_message(
    event: dict,
    *,
    env: AgentEnv,
    model: OpenAICompatibleChatModel,
    tools_manager: ToolsManager,
    skills_manager: SkillsManager,
    store: ConversationStore,
    settings: Settings,
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
        messages: list[dict] = [{"role": "system", "content": prompts.SYSTEM_PROMPT}]
    else:
        messages = trim_history(persisted, max_user_turns=settings.max_history_turns)

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
        action_timeout=settings.action_timeout,
        max_turns=settings.max_steps_per_turn,
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


async def main() -> None:
    settings = Settings.from_env()

    print("=" * 80)
    print("Lark chat agent (multi-turn, multi-tool)")
    print("=" * 80)
    print(f"container:     {settings.container}")
    print(f"swerex:        {settings.swerex_host}:{settings.swerex_port}")
    print(f"model:         {settings.model_name} @ {settings.model_base_url}")
    print(f"skills dir:    {settings.skills_dir} (exists={settings.skills_dir.is_dir()})")
    print(f"history dir:   {settings.history_dir}")

    lark_cli_prefix = settings.lark_cli_prefix()
    print(f"lark-cli via:  {' '.join(lark_cli_prefix)} lark-cli ...")

    print("\n[1/6] Resolving bot open_id via Lark Open API...")
    bot_open_id = await fetch_bot_open_id(command_prefix=lark_cli_prefix)
    print(f"  bot open_id: {bot_open_id}")

    print("\n[2/6] Starting sandbox env...")
    run_id = str(uuid.uuid4())
    env = AgentEnv(run_id=run_id, env_config=settings.build_env_config())
    await env.start()
    print("  env started")

    print("\n[3/6] Installing tools + skills...")
    tools_manager = ToolsManager(
        ToolsManagerConfig(
            tools=[
                ToolConfig(name="execute_bash"),
                ToolConfig(name="lark-cli"),
                ToolConfig(name="str_replace_editor"),
                ToolConfig(name="finish"),
            ]
        )
    )
    await env.install_tools(tools_manager.tools)

    skills_manager = SkillsManager.from_config(SkillsManagerConfig(skills_dir=settings.skills_dir))
    await env.install_skills(skills_manager)
    print(f"  {len(skills_manager.skills)} skill(s): {[s.name for s in skills_manager.skills]}")

    await env.communicate("mkdir -p /workspace/history", check="raise")

    print("\n[4/6] Wiring model client...")
    model = OpenAICompatibleChatModel(
        base_url=settings.model_base_url,
        api_key=settings.model_api_key,
        model_name=settings.model_name,
        sampling_params=settings.sampling_params,
    )
    model.set_tools_schemas(tools_manager.tools_schemas)

    store = ConversationStore(base_dir=settings.history_dir / "conversations")
    print(f"  conversation store: {store.base_dir}")

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
                    settings=settings,
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
