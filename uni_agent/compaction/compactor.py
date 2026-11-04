"""Automatic conversation-trajectory compaction for agent loops.

Implements the checkpoint-style compaction used to keep long agentic
trajectories inside the model context window:

* **Trigger**: after each agent round, measure token usage (provider usage
  when available, heuristic estimate otherwise). Above
  ``threshold_ratio * max_context_tokens`` -> compact.
* **LLM checkpoint summary**: append a summary instruction *after* the
  full conversation (prefix reuse -> provider KV cache stays valid), let
  the model emit one structured checkpoint, then rebuild the conversation
  as ``[system, original user task, assistant(summary)]``.

  The system prompt and the original user task are kept VERBATIM in the
  rebuilt conversation, so the summary does NOT restate the task - it
  only carries process state (attempts / constraints / current code /
  last verify output / next step).

* **Multi-compaction**: the instruction merges any prior checkpoint into
  the new one (single consolidated summary, never stacked).
* **Safety**: the framed summary must be strictly smaller than the
  shadowed content, otherwise the compaction is rejected and the original
  conversation is kept (fallback). All events are logged and returned as
  stats for observability.

Designed for the inference path (``OpenAICompatibleChatModel``), where
``messages`` is passed wholesale to the API each call and the rollout
cache is stateless. Not wired into the verl token-id training path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from uni_agent.compaction.token_meter import TokenMeter, estimate_message

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# Summary instruction appended as the LAST user message of the
# conversation being compacted. Written for the Triton Ascend operator
# generation task; follows deepseek-harness's checkpoint structure with
# the retention priorities of docs/对话轨迹压缩方案.md. The task itself is
# NOT restated here - system prompt + original user task survive
# compaction verbatim.
COMPACTION_INSTRUCTION = """\
You are now acting as a compaction engine for this AI coding assistant.
Condense the conversation ABOVE into a structured checkpoint that lets you
resume the operator implementation work with no loss of essential context.
The system prompt and the original task message remain available after
compaction - do NOT restate them; capture only the work state.

Output EXACTLY the Markdown structure below: keep every section, in order.
Use terse bullets, not prose paragraphs. Write "(none)" for an empty
section - never drop a section. Do not wrap the output in code fences.

## Attempts History
- Each line: `尝试N-M: [方案] -> [结果]` (or `Attempt N-M: [approach] -> [outcome]`).
- Only record approach-level changes, not micro fixes. Merge repeated
  failures of the same kind into one line.

## Constraints Learned
- Rules discovered while working: AST-check bans, API limits, hardware
  constraints, verifier interface contracts (e.g. ModelNew signature).
- Deduplicate; these must survive every future compaction.

## Design Decisions
- Why the current implementation approach was chosen (one line each).

## Errors and Fixes
- error -> how it was resolved (or why it is still unresolved).

## Files and Code
- Exact file paths, and the CURRENT full content of the implementation
  file (final version only; drop intermediate snapshots). If the file is
  very long, keep the kernel signatures, class/forward structure, and the
  parts most relevant to the next fix.

## Current State
- Verification progress: ast_passed / compile_passed / correctness passed,
  pass_rate, speedup_vs_torch so far, number of verify runs.
- The COMPLETE output of the LAST run_verify (or "(none)" if never run).

## Next Step
- The single next action to continue the task.

Rules:
- Write concise engineering prose. Preserve exact file paths, commands,
  error strings, identifiers, numeric values, and function signatures.
- Completely DROP: skill/reference document contents read via tools (they
  live on disk and can be re-read), intermediate code snapshots, repeated
  skill listings, and verbose tool outputs not listed above.
- If the conversation already contains a prior checkpoint summary, it is a
  PRIOR compaction: do not copy it forward verbatim - preserve still-true
  facts, drop stale ones, and merge newer information into this single
  consolidated summary.
- Do NOT mention this summarization request or that the context was
  compacted.
- Wrap the ENTIRE checkpoint output in <compact_result>...</compact_result>
  tags. The content between these tags will be extracted as the compacted
  checkpoint; anything outside the tags (including any thinking, reasoning,
  or analysis) will be discarded and NOT passed to the next conversation turn.
  Output only the tags and the checkpoint, do not call any tool.
"""

# Framing wrapped around the LLM-generated summary when it replaces the
# conversation (kept minimal - the summary is stored as the model's own
# assistant message, so it must read as established background).
CHECKPOINT_PREAMBLE = (
    "[Compacted checkpoint of the earlier conversation. Treat it as "
    "established background and continue the task from here.]"
)


def _extract_compact_result(text: str) -> str:
    """Extract checkpoint content between ``<compact_result>...</compact_result>`` tags.

    The compaction instruction asks the model to wrap its checkpoint inside
    these tags. Anything outside (thinking, reasoning, analysis) is discarded
    and never reaches the next conversation turn. The raw response (including
    the stripped content) is preserved separately via
    ``last_summary_call["raw_response"]`` for the pre-compaction segment.

    Falls back to the original text when the tags are missing or malformed.
    """
    import re

    m = re.search(r"<compact_result>(.*?)</compact_result>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def frame_summary(summary_text: str) -> str:
    """Wrap the raw summary in the checkpoint preamble (and strip stray
    code fences some models add around the checkpoint)."""
    # Extract the tagged checkpoint content first (discards thinking /
    # reasoning outside <compact_result>...</compact_result>).
    summary_text = _extract_compact_result(summary_text)
    summary_text = summary_text.strip()
    if summary_text.startswith("```"):
        lines = summary_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        summary_text = "\n".join(lines).strip()
    return f"{CHECKPOINT_PREAMBLE}\n\n{summary_text}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CompactionConfig:
    """All knobs for the compaction service (env-overridable)."""

    enabled: bool = True
    # Compact when current tokens > threshold_ratio * max_context_tokens.
    threshold_ratio: float = 0.8
    # Model context window used to derive the trigger threshold.
    max_context_tokens: int = 40960
    # Minimum messages (beyond system + task) before compaction may run.
    min_messages: int = 6
    # After a failed/rejected compaction, wait this many steps to retry.
    failure_backoff_steps: int = 3
    # Extra calls to the summarizer allowed per compaction (retries when
    # the first summary is not smaller than the shadowed content).
    max_retries: int = 1

    @property
    def threshold_tokens(self) -> int:
        return int(self.max_context_tokens * self.threshold_ratio)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, max_model_len: int | None = None) -> "CompactionConfig":
        import os

        env = env if env is not None else dict(os.environ)

        def _bool(key: str, default: bool) -> bool:
            raw = env.get(key)
            if raw is None or raw == "":
                return default
            return raw.strip().lower() in ("1", "true", "yes", "on")

        def _int(key: str, default: int) -> int:
            try:
                return int(env.get(key, default))
            except (TypeError, ValueError):
                return default

        def _float(key: str, default: float) -> float:
            try:
                return float(env.get(key, default))
            except (TypeError, ValueError):
                return default

        # max_context_tokens comes from the model's real inference length.
        max_context_tokens = max_model_len if max_model_len is not None else 40960

        return cls(
            enabled=_bool("TRITON_COMPACTION_ENABLED", True),
            threshold_ratio=_float("TRITON_COMPACTION_THRESHOLD", 0.8),
            max_context_tokens=max_context_tokens,
            failure_backoff_steps=_int("TRITON_COMPACTION_BACKOFF", 3),
        )


# ---------------------------------------------------------------------------
# Compaction service
# ---------------------------------------------------------------------------


@dataclass
class CompactionEvent:
    """Observability record for one compaction attempt."""

    step: int
    triggered: bool = False
    stage: str = ""  # "summary" | "rejected" | "error"
    tokens_before: int = 0
    tokens_after: int = 0
    messages_before: int = 0
    messages_after: int = 0
    summary_chars: int = 0
    duration_s: float = 0.0
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "triggered": self.triggered,
            "stage": self.stage,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "messages_before": self.messages_before,
            "messages_after": self.messages_after,
            "summary_chars": self.summary_chars,
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
        }


@dataclass
class _RunState:
    """Book-keeping across compaction attempts within one agent run."""

    events: list[CompactionEvent] = field(default_factory=list)
    steps_since_failure: int = 0
    disabled_reason: str = ""


class ConversationCompactor:
    """Trigger + compaction over a live ``messages`` list.

    Usage in an agent loop, after each round::

        compactor.notify_step()
        if compactor.should_compact(interaction.messages, usage, step):
            new_messages = await compactor.compact(
                interaction.messages, chat_model, step_idx
            )
            if new_messages is not None:
                interaction.messages[:] = new_messages
    """

    def __init__(
        self,
        config: CompactionConfig,
        meter: TokenMeter | None = None,
        logger: Any = None,
    ):
        self.config = config
        self.meter = meter or TokenMeter()
        self.logger = logger
        self.state = _RunState()
        #: Populated after each successful summary compaction:
        #: ``{"prompt_messages": [...], "summary": str, "raw_response": str}``
        #: where ``prompt_messages`` is the EXACT input of the summarizer call
        #: (conversation + instruction as the last user message), ``summary``
        #: is the framed checkpoint text (thinking stripped), and
        #: ``raw_response`` is the model's original response including any
        #: thinking prefix (preserved for the pre-compaction segment).
        #: ``None`` after failed attempts.
        self.last_summary_call: dict[str, Any] | None = None

    # -- logging helper ---------------------------------------------------

    def _log(self, level: str, msg: str) -> None:
        if self.logger is not None:
            getattr(self.logger, level)(msg)

    # -- trigger -----------------------------------------------------------

    def should_compact(
        self,
        messages: list[dict[str, Any]],
        last_usage: tuple[int | None, int | None] | None = None,
    ) -> bool:
        if not self.config.enabled:
            return False
        if self.state.disabled_reason:
            return False
        if 0 < self.state.steps_since_failure <= self.config.failure_backoff_steps:
            return False
        if len(messages) < 2 + self.config.min_messages:
            return False
        current = self.meter.current_tokens(messages, last_usage)
        return current > self.config.threshold_tokens

    # -- LLM summary ---------------------------------------------------------

    async def _generate_summary(
        self,
        messages: list[dict[str, Any]],
        model: Any,
    ) -> str:
        """Call the model with the instruction appended after the full
        conversation (prefix reuse -> KV cache friendly). Returns the raw
        summary text.
        """
        instruction_msg = {"role": "user", "content": COMPACTION_INSTRUCTION}
        # Prefix is unchanged so the serving stack can reuse KV cache.
        # uni-agent's OpenAI client returns (text, tool_calls, generation_info).
        queried = await model.query([*messages, instruction_msg])
        if isinstance(queried, tuple):
            return str(queried[0] or "")
        return str(queried or "")

    # -- main entry ----------------------------------------------------------

    async def compact(
        self,
        messages: list[dict[str, Any]],
        model: Any,
        step: int = 0,
        last_usage: tuple[int | None, int | None] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Run one compaction attempt.

        Returns the new message list on success, ``None`` on failure (the
        caller keeps the original conversation - graceful fallback).
        """
        assert len(messages) >= 2, "need at least [system, user task]"
        cfg = self.config
        event = CompactionEvent(
            step=step,
            triggered=True,
            tokens_before=self.meter.current_tokens(messages, last_usage),
            messages_before=len(messages),
        )
        self.state.events.append(event)
        t0 = time.perf_counter()
        self.last_summary_call = None

        system_msg = messages[0]
        task_msg = messages[1]
        shadowed = messages[2:]
        shadowed_tokens = self.meter.estimate_messages(shadowed)

        try:
            # ---- LLM checkpoint summary ----
            summary_text = ""
            raw_response = ""  # preserved for segment_0 (includes thinking)
            attempts_allowed = 1 + cfg.max_retries
            for attempt in range(attempts_allowed):
                raw_summary = await self._generate_summary(messages, model)
                if not raw_summary or not raw_summary.strip():
                    event.stage = "rejected"
                    event.error = "empty summary"
                    self.state.steps_since_failure = 1
                    self._log("error", f"[compaction] step={step} rejected: empty summary")
                    return None
                summary_text = frame_summary(raw_summary)
                if estimate_message({"role": "assistant", "content": summary_text}) < shadowed_tokens:
                    raw_response = raw_summary
                    break
                self._log(
                    "warning",
                    f"[compaction] step={step} summary not smaller than shadowed "
                    f"content (attempt {attempt + 1}/{attempts_allowed})",
                )
            else:
                event.stage = "rejected"
                event.duration_s = time.perf_counter() - t0
                self.state.steps_since_failure = 1
                self._log(
                    "error",
                    f"[compaction] step={step} rejected: summary not smaller than "
                    f"shadowed content ({shadowed_tokens} est tokens); keeping original",
                )
                return None

            new_messages = [
                system_msg,
                task_msg,
                {"role": "assistant", "content": summary_text},
            ]
            self.last_summary_call = {
                # exact input of the summarizer call: full conversation +
                # instruction as the LAST user message
                "prompt_messages": [
                    *messages,
                    {"role": "user", "content": COMPACTION_INSTRUCTION},
                ],
                "summary": summary_text,
                "raw_response": raw_response,
            }
            event.stage = "summary"
            event.messages_after = len(new_messages)
            event.tokens_after = self.meter.estimate_messages(new_messages)
            event.summary_chars = len(summary_text)
            event.duration_s = time.perf_counter() - t0
            self.state.steps_since_failure = 0
            self._log(
                "info",
                f"[compaction] step={step} compacted: {event.tokens_before} -> "
                f"{event.tokens_after} est tokens, {event.messages_before} -> "
                f"{event.messages_after} messages, summary={event.summary_chars} chars, "
                f"{event.duration_s:.1f}s",
            )
            return new_messages
        except Exception as exc:  # malformed/oversized summaries can fall back; infra failures cannot
            from uni_agent.rollout_lifecycle import RolloutInfrastructureError

            if isinstance(exc, RolloutInfrastructureError):
                exc.messages = list(messages)
                raise
            event.stage = "error"
            event.error = f"{type(exc).__name__}: {exc}"
            event.duration_s = time.perf_counter() - t0
            self.state.steps_since_failure = 1
            self._log("error", f"[compaction] step={step} failed: {event.error}; keeping original")
            return None

    # -- observability -------------------------------------------------------

    def notify_step(self) -> None:
        """Call once per agent step so failure backoff can expire."""
        if self.state.steps_since_failure:
            self.state.steps_since_failure += 1

    def stats(self) -> dict[str, Any]:
        return {
            "events": [e.as_dict() for e in self.state.events],
            "num_compactions": sum(1 for e in self.state.events if e.stage == "summary"),
            "num_failures": sum(1 for e in self.state.events if e.stage in ("rejected", "error")),
        }
