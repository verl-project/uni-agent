"""deepseek_harness: a black-box agent that runs the real `dsh --profile headless`
inside the sandbox.

DeepSeek Harness is launched *in* the sandbox from a prebuilt tool image mounted
at ``/opt/dsh`` (Node CLI + node runtime + the baked ``minimal_patch.yml``).
This agent builds the ``dsh --profile headless`` shell command, points DSH's
built-in ``llm-deepseek`` route at the gateway tunnel, execs it in the sandbox,
and returns the result.

Unlike ``mini_swe_agent`` there is **no in-sandbox ``run_agent.py`` entrypoint**:
``dsh`` is itself the agent binary, so this agent builds and runs the command
directly. The unified ``uni_agent.framework.task_runner.run_task`` bridge owns the
gateway reverse tunnel (``upstream`` + ``model.base_url`` rewrite to
``http://127.0.0.1:<proxy_port>/v1``) and the reward, so this agent only owns the
launch.

Wire compat: DSH speaks OpenAI Chat Completions streaming with tool_calls, so it
uses the gateway's OpenAI adapter directly.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

# DSH writes session persistence + the auto-initialized headless profile here;
# must be writable inside the sandbox. Auto-init uses the templates shipped in
# the tool image, so it needs no network.
DSH_HOME_IN_SANDBOX = "/tmp/dsh-home"


def build_dsh_command(
    *,
    task: str,
    base_url: str,
    conda_env_path: str | None = None,
    tool_mode: str = "full",
    dsh_bin: str = "dsh",
    minimal_patch_path: str = "/opt/dsh/minimal_patch.yml",
    workdir: str | None = None,
    model_name: str | None = None,
    max_model_len: int | None = None,
    permission_mode: str = "danger-full-access",
    step_limit: int | None = None,
    disable_title_generation: bool = True,
    disable_auto_compaction: bool = True,
) -> str:
    """Build the shell command that runs DSH headless against the gateway tunnel.

    ``base_url`` is already the sandbox-internal tunnel address; ``run_task``
    rewrites ``agent.model.base_url`` before this agent runs.

    ``model_name`` (a runtime binding) drives an ``llm-deepseek`` overlay pointing
    DSH at the served model -- its own catalog model would 404 against a vLLM /
    OpenAI-compatible endpoint. The optional ``max_model_len`` advertises that
    context (DSH otherwise falls back to 131072) and caps the per-request
    ``maxTokens`` at 80% of it; harness-side only, since the Gateway clamps each
    request's ``max_tokens`` to the remaining trajectory capacity.

    ``dsh`` and its bundled ``node`` are exposed at ``/usr/bin`` via
    ``sandbox.executable_paths``, so nothing is added to PATH here.
    ``conda_env_path`` (optional, absolute) activates the task image's conda env
    for the shell commands DSH runs.

    ``tool_mode``: ``"minimal"`` restricts the surface to bash +
    str_replace_editor via the baked ``minimal_patch.yml``; ``"full"`` keeps the
    default 25-tool Code Mode surface.
    """
    # DSH's llm-deepseek route honors DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY. The
    # gateway has no auth path, so a dummy key is fine.
    env = {
        "DEEPSEEK_BASE_URL": base_url,
        "DEEPSEEK_API_KEY": "not-needed",
        "DSH_HOME": DSH_HOME_IN_SANDBOX,
        "IS_SANDBOX": "1",
        "DISABLE_AUTOUPDATER": "1",
        # Inside the openYuanrong outer sandbox DSH has no inner bash-sandbox
        # backend (bubblewrap/Landlock/approval channel); default to
        # danger-full-access so DSH's bash tool runs against the already-isolated
        # /testbed. Override per deployment via cfg.permission_mode.
        "DSH_PERMISSION_MODE": permission_mode,
    }
    env_assignments = [f"{key}={shlex.quote(value)}" for key, value in env.items()]

    if conda_env_path:
        # The task image keeps the repo's interpreter and deps in a conda env, and
        # DSH's bash tool inherits this process env, so activate it here. The conda
        # root is derived from the env path rather than hardcoding /opt/miniconda3.
        # `dsh` and `node` come from sandbox.executable_paths (/usr/bin), so nothing
        # else needs to be added to PATH.
        env_dir = PurePosixPath(conda_env_path)
        env_assignments.extend(
            [
                f"CONDA_DEFAULT_ENV={shlex.quote(env_dir.name)}",
                f"CONDA_PREFIX={shlex.quote(str(env_dir))}",
                f"PATH={shlex.quote(str(env_dir / 'bin'))}:{shlex.quote(str(env_dir.parent.parent / 'bin'))}:$PATH",
            ]
        )

    argv = [dsh_bin, "--profile", "headless"]
    if tool_mode == "minimal":
        argv += ["--patch", minimal_patch_path]
        logger.info("deepseek_harness: tool_mode=minimal -> %s (bash + str_replace_editor)", minimal_patch_path)
    elif tool_mode != "full":
        raise ValueError(f"deepseek_harness: unsupported tool_mode={tool_mode!r} (expected 'full' or 'minimal')")

    # Optional overlay that points DSH's llm-deepseek route at the served model
    # with a maxTokens cap compatible with the endpoint's max_model_len.
    # Written as a standalone command BEFORE the env-prefixed dsh command:
    # a heredoc delimiter must end its own line, so it cannot be spliced into
    # the space-joined env prefix (that would swallow the dsh argv into the
    # heredoc body and never run dsh). Each building block is its own line in
    # the produced shell script (no leading ';' on any line).
    blocks = [f"cd {shlex.quote(workdir or '/testbed')}; mkdir -p {DSH_HOME_IN_SANDBOX}"]
    if model_name:
        model_patch_path, model_patch_script = _build_model_overlay_patch(
            model_name=model_name,
            max_model_len=max_model_len,
            path="/tmp/dsh_model_patch.yml",
        )
        argv += ["--patch", model_patch_path]
        blocks.insert(0, model_patch_script)

    # Optional step watchdog: cap agent steps by counting DSH's plaintext session
    # log. DSH has no native step flag, so we (1) force the session-persistence
    # plugin to write uncompressed JSONL via a patch, then (2) launch dsh in the
    # background and run a watchdog beside it that kills that exact PID once
    # `step/start` events reach `step_limit` (see the launch tail below, which owns
    # the ordering). A killed dsh exits non-zero -> finished=False -> masked from
    # the loss (same semantics as mini_swe_agent.step_limit).
    # Normalized once so the launch tail and the watchdog helper agree; `None`
    # disables the watchdog, and the narrowed `int` is what the helper takes.
    step_cap = int(step_limit) if step_limit is not None and step_limit > 0 else None
    if step_cap is not None:
        session_patch_path, session_patch_script = _build_session_log_patch(
            path="/tmp/dsh_session_patch.yml",
        )
        argv += ["--patch", session_patch_path]
        blocks.append(session_patch_script)

    # Optional data-hygiene patch: disable DSH's auto session-title and auto
    # context-compaction so each rollout yields a single clean task trajectory
    # (no 1-turn title row, no compact-boundary split). Written under /tmp so it
    # does not depend on the mounted tool image being writable.
    hygiene_patch_path, hygiene_patch_script = _build_data_hygiene_patch(
        path="/tmp/dsh_data_hygiene.yml",
        disable_title=disable_title_generation,
        disable_auto_compaction=disable_auto_compaction,
    )
    if hygiene_patch_path:
        argv += ["--patch", hygiene_patch_path]
        blocks.append(hygiene_patch_script)

    # The task is the headless positional.
    argv.append(task)
    env_prefix = " ".join(env_assignments)
    dsh_run = f"{env_prefix} " + shlex.join(argv)
    if step_cap is not None:
        # dsh goes first so the watchdog can be handed its exact PID: $! is only
        # meaningful immediately after the background launch, and the subshell
        # forked below inherits $DSPID.
        dsh_run = (
            f"{dsh_run} &\n"
            "DSPID=$!\n"
            + _build_step_watchdog(step_limit=step_cap, session_dir=DSH_HOME_IN_SANDBOX)
            + "WPID=$!\n"
            + _watchdog_wait_block()
        )
    return "\n".join(blocks) + f"\n{dsh_run}"


def _build_model_overlay_patch(
    *,
    model_name: str,
    max_model_len: int | None,
    path: str,
) -> tuple[str, str]:
    """Return ``(patch_path, shell_script)`` writing a DSH ``llm-deepseek`` overlay.

    DSH's default catalog model 404s against a vLLM / OpenAI-compatible endpoint,
    so the patch replaces the catalog with ``model_name`` (``contextWindow`` and
    ``maxTokens`` capped under ``max_model_len`` when known) and points
    ``agent-default-model`` at it. Emitted as its own command block ahead of the
    env-prefixed dsh command; a heredoc delimiter must end its own line.
    """
    max_tokens = None
    if max_model_len is not None and max_model_len > 0:
        # Leave headroom for the prompt inside the endpoint's total context.
        max_tokens = int(max_model_len * 0.8)
    lines = ["- id: llm-deepseek", "  config:"]
    if max_tokens is not None:
        lines += [f"    maxTokens: {max_tokens}"]
    lines += ["    models:"]
    lines += [
        f"      - id: {model_name}",
        f"        name: {model_name}",
        f"        contextWindow: {max_model_len or 131072}",
    ]
    if max_tokens is not None:
        lines += [f"        maxTokens: {max_tokens}"]
    lines += [
        "- id: agent-default-model",
        "  config:",
        "    provider: deepseek-official",
        f"    model: {model_name}",
        "",
    ]
    body = "\n".join(lines)
    # The heredoc delimiter must terminate on its own line; the trailing
    # newline lets the caller append "; ..." on the next line.
    script = f"cat > {shlex.quote(path)} <<'DSH_MODEL_PATCH_EOF'\n{body}DSH_MODEL_PATCH_EOF\n"
    logger.info("deepseek_harness: model overlay patch (model_name=%s max_model_len=%s)", model_name, max_model_len)
    return path, script


def _build_data_hygiene_patch(*, path: str, disable_title: bool, disable_auto_compaction: bool) -> tuple[str, str]:
    """Return ``(patch_path, shell_script)`` for the runtime data-hygiene patch.

    ``disable_title`` disables DSH's auto session-title generation, which would
    otherwise surface as an extra 1-turn trajectory with no task signal;
    ``disable_auto_compaction`` sets ``compaction-basic.auto=false`` so a rollout
    never splits at a compaction boundary (over-length tasks fail and are masked
    instead). Written under the sandbox ``/tmp`` so it works even when the mounted
    tool image is read-only.
    """
    body = []
    if disable_title:
        body += [
            "- id: session-title",
            "  disabled: true",
            "- id: session-title-llm",
            "  disabled: true",
        ]
    if disable_auto_compaction:
        body += [
            "- id: compaction-basic",
            "  config:",
            "    auto: false",
        ]
    if not body:
        return "", ""
    body.append("")
    script = f"cat > {shlex.quote(path)} <<'DSH_HYGIENE_PATCH_EOF'\n{chr(10).join(body)}DSH_HYGIENE_PATCH_EOF\n"
    logger.info(
        "deepseek_harness: data-hygiene patch (disable_title=%s disable_auto_compaction=%s)",
        disable_title,
        disable_auto_compaction,
    )
    return path, script


def _build_session_log_patch(*, path: str) -> tuple[str, str]:
    """Return ``(patch_path, shell_script)`` forcing DSH's session log to plaintext.

    ``dsh-session-persistence-jsonl`` defaults to ``compression: zstd`` (writes
    ``session.jsonl.zstd``). The step watchdog needs to count ``step/start``
    events cheaply, so this overlay sets ``compression: none`` -> plaintext
    ``session.jsonl`` under ``$DSH_HOME/sessions/...``.
    """
    body = "\n".join(
        [
            "- id: session-persistence-jsonl",
            "  config:",
            "    root: !!js dshHomePath('sessions')",
            "    compression: none",
            "",
        ]
    )
    script = f"cat > {shlex.quote(path)} <<'DSH_SESSION_PATCH_EOF'\n{body}DSH_SESSION_PATCH_EOF\n"
    logger.info("deepseek_harness: session-persistence patched to plaintext JSONL (for step watchdog)")
    return path, script


def _build_step_watchdog(*, step_limit: int, session_dir: str) -> str:
    """Return a backgrounded snippet that kills this episode's dsh at ``step_limit``.

    Counts ``"type":"step/start"`` lines under ``session_dir`` every 0.5s and kills
    the PID captured in ``$DSPID`` once the count reaches the limit. The exact PID
    matters: the script runs as ``bash -c <script>``, so a ``pkill -f`` on the dsh
    path would also SIGKILL the shell running the episode and lose its exit status.
    """
    sessions_glob = f"{shlex.quote(session_dir)}/sessions/*/*/session.jsonl"
    return (
        "(\n"
        "  while :; do\n"
        f'    N=$(grep -ah \'"type":"step/start"\' {sessions_glob} 2>/dev/null | wc -l)\n'
        f'    if [ "${{N:-0}}" -ge {int(step_limit)} ]; then\n'
        '      kill -9 "$DSPID" 2>/dev/null\n'
        "      break\n"
        "    fi\n"
        "    sleep 0.5\n"
        "  done\n"
        ") &\n"
    )


def _watchdog_wait_block() -> str:
    """Return the shell tail that waits on dsh, stops the watchdog, and exits with
    dsh's real status (non-zero when the step cap fired, so the episode is
    reported unfinished).
    """
    return 'wait $DSPID\nRC=$?\nkill "$WPID" 2>/dev/null\nexit $RC\n'


class DeepSeekHarnessConfig(AgentConfig):
    """Black-box launch params for DeepSeek Harness (endpoint lives on :attr:`AgentConfig.model`)."""

    name: str = "deepseek_harness"
    run_timeout: float = Field(default=7200.0, description="Wallclock cap (s) on the dsh process.")
    conda_env_path: str | None = Field(
        default=None,
        description="Task-image path of the conda env used by the shell commands DSH runs "
        "(e.g. /opt/miniconda3/envs/testbed); leave unset for images without conda.",
    )
    tool_mode: str = Field(
        default="minimal",
        description="'minimal' (bash + str_replace_editor via --patch, the default) or 'full' (25-tool Code Mode).",
    )
    # Tool-image paths are bound to the prebuilt tool image's Dockerfile layout
    # (mounted at /opt/dsh). `dsh` and its bundled `node` are exposed at /usr/bin by
    # sandbox.executable_paths, so the CLI is invoked by name; the task config can
    # override either path if that layout changes.
    dsh_bin: str = Field(
        default="dsh",
        description="dsh CLI command, exposed at /usr/bin via sandbox.executable_paths "
        "(the tool image ships it at /opt/dsh/bin/dsh).",
    )
    minimal_patch_path: str = Field(
        default="/opt/dsh/minimal_patch.yml",
        description="Patch overlay baked into the tool image; applied when tool_mode=minimal.",
    )
    permission_mode: str = Field(
        default="danger-full-access",
        description="``DSH_PERMISSION_MODE`` for the headless run. Inside an outer isolation "
        "sandbox (openYuanrong), DSH's own inner bash sandbox has no backend (no bubblewrap/"
        "Landlock/approval channel), so ``bash`` tool calls fail unless set to "
        "``danger-full-access`` (letting DSH run unconfined because the outer sandbox already "
        "isolates it). Set ``workspace-write``/``read-only`` only when DSH runs directly on a host.",
    )
    max_model_len: int | None = Field(
        default=None,
        description="Endpoint context (tokens) advertised to DSH: sets the model overlay's "
        "contextWindow and caps its per-request maxTokens at 80% of this value. Harness-side "
        "only -- the Gateway already clamps every request's max_tokens to the remaining "
        "trajectory capacity (rollout.prompt_length + rollout.response_length); set this to "
        "that same sum so DSH's own context accounting matches the served endpoint.",
    )
    step_limit: int | None = Field(
        default=None,
        description="Max agent steps (DSh 'step' boundaries) before the episode is cut short. "
        "DSH has no native --max-steps flag, so a background watchdog inside the launch command "
        "counts ``step/start`` events in DSH's plaintext session log and kills the dsh process "
        "once the limit is reached. The killed dsh exits non-zero, so ``finished=False`` and the "
        "episode is masked from the loss via ``mask_unfinished_episode=True`` (same training "
        "semantics as mini_swe_agent.step_limit). None = no step cap (only run_timeout).",
    )
    disable_title_generation: bool = Field(
        default=True,
        description="Disable DSH's auto session-title generation (first-prompt LLM title call). "
        "Inside the training gateway that call materializes as a ~64-token extra trajectory on "
        "every rollout (0-indexed, carrying the task reward but no task signal). Default True to "
        "keep training batches clean.",
    )
    disable_auto_compaction: bool = Field(
        default=True,
        description="Disable DSH's auto context compaction (compaction-basic.auto=false). "
        "Stops step-pressure compaction and context-overflow auto-recovery, so a rollout never "
        "splits into compact-boundary segments. Tasks exceeding contextWindow will fail with a "
        "context-window-exceeded error and be masked from the loss. Default True (prefer clean "
        "single-trajectory rollouts over recovering over-length tasks).",
    )


@register_agent("deepseek_harness")
class DeepSeekHarnessAgent(Agent):
    """Black-box solver: launch `dsh --profile headless` in the sandbox against ``config.model``."""

    config_model = DeepSeekHarnessConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
        workdir: str | None = None,
    ) -> AgentResult:
        cfg: DeepSeekHarnessConfig = self.config  # type: ignore[assignment]
        if cfg.model.base_url is None:
            raise ValueError(
                "deepseek_harness: config.model.base_url is not set (run_task fills the gateway tunnel address)"
            )
        task = self._extract_task(messages)

        # run_task has already rewritten cfg.model.base_url to the sandbox-internal
        # tunnel (http://127.0.0.1:<proxy_port>/v1), so it passes through as-is.
        agent_cmd = build_dsh_command(
            task=task,
            base_url=cfg.model.base_url,
            conda_env_path=cfg.conda_env_path,
            tool_mode=cfg.tool_mode,
            dsh_bin=cfg.dsh_bin,
            minimal_patch_path=cfg.minimal_patch_path,
            workdir=workdir or "/testbed",
            model_name=cfg.model.model_name,
            max_model_len=cfg.max_model_len,
            permission_mode=cfg.permission_mode,
            step_limit=cfg.step_limit,
            disable_title_generation=cfg.disable_title_generation,
            disable_auto_compaction=cfg.disable_auto_compaction,
        )
        # DSH headless has no CLI --max-turns flag; step_limit (when set) is
        # enforced in-sandbox by a step-counting watchdog inside the launch
        # command (kills dsh at the limit -> non-zero exit -> finished=False).
        result = await sandbox.exec_shell(agent_cmd, timeout=cfg.run_timeout)

        # DSH headless exits 0 when the agent finishes normally; treat that as a
        # finished episode (non-zero => crash/timeout => masked from loss via
        # mask_unfinished_episode=True in the framework config).
        finished = result.exit_code == 0
        logger.info(
            "deepseek_harness: done rc=%s tool_mode=%s finished=%s",
            result.exit_code,
            cfg.tool_mode,
            finished,
        )
        return AgentResult(
            output={
                "dsh_exit_code": result.exit_code,
                "stdout_tail": (result.stdout or "")[-2000:],
                "tool_mode": cfg.tool_mode,
                "step_limit": cfg.step_limit,
            },
            transcript=list(messages),
            info={
                "tool_mode": cfg.tool_mode,
                "exit_code": result.exit_code,
                "step_limit": cfg.step_limit,
            },
            finished=finished,
        )

    @staticmethod
    def _extract_task(messages: list[dict[str, Any]]) -> str:
        if len(messages) > 2:
            raise ValueError(f"deepseek_harness accepts at most 2 messages (system?, user), got {len(messages)}")
        problem = next((m["content"] for m in messages if m.get("role") == "user"), None)
        # Must be plain text: an OpenAI content-parts list would reach shlex.join()
        # as a list element and fail there with an opaque TypeError.
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(
                "deepseek_harness requires a 'user' message with non-empty text content "
                "(content-parts lists are not supported)"
            )
        return problem
