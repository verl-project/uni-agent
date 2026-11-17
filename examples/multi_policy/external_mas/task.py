"""External-MAS Task definition and scoring."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from examples.multi_policy.external_mas.execution import (
    ExternalMASRequest,
    extract_user_task,
    run_external_mas,
)
from examples.multi_policy.external_mas.reward import (
    compute_score,
    extract_answer,
    extract_expected_answers,
)
from uni_agent.agents import AgentConfig
from uni_agent.sandbox import SandboxConfig
from uni_agent.tasks import Task, TaskConfig, TaskResult
from uni_agent.tasks.registry import register_task


class _LauncherSection(BaseModel):
    """Base for the ``launcher_kwargs`` sections: unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")


class ExternalMASCommandConfig(_LauncherSection):
    """How the external MAS command is built."""

    argv: list[str] | None = None
    template: str | None = None
    shell: bool = False
    work_dir: str | None = None
    env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_command_source(self) -> ExternalMASCommandConfig:
        if self.argv is not None and self.template is not None:
            raise ValueError("command.argv and command.template are mutually exclusive")
        if self.argv is None and self.template is None:
            raise ValueError("command.argv or command.template is required")
        if self.argv is not None and self.shell:
            raise ValueError("command.argv cannot be combined with shell=true")
        return self

    @field_validator("env", mode="before")
    @classmethod
    def _stringify_env(cls, value: Any) -> Any:
        """Accept YAML scalars (``yes``/``1``) and stringify them like the runner does."""
        if isinstance(value, Mapping):
            return {str(key): str(item) for key, item in value.items()}
        return value


class ExternalMASTemplateConfig(_LauncherSection):
    """Which MAS config template to copy, rewrite, and where to put the result."""

    #: Host-side path of the MAS config template (relative paths use the task cwd).
    template_path: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    injection: dict[str, Any] = Field(default_factory=dict)
    #: Sandbox directory receiving each rollout's rendered config.
    config_dir: str = "/tmp"

    @field_validator("template_path")
    @classmethod
    def _require_template_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("config.template_path must be non-empty")
        return value

    @field_validator("config_dir")
    @classmethod
    def _require_absolute_config_dir(cls, value: str) -> str:
        # The rendered config is written inside the sandbox, so a relative path
        # would resolve against the sandbox cwd and the MAS could miss the file.
        if not value.startswith("/"):
            raise ValueError("config.config_dir must be an absolute sandbox path")
        return value


class ExternalMASTimeoutConfig(_LauncherSection):
    """How long the external command may run."""

    seconds: float | None = None


class ExternalMASResultConfig(_LauncherSection):
    """How to read the external command's output."""

    mode: Literal["stdout_json", "stdout_text", "none"] = "stdout_json"
    field_mapping: dict[str, Any] = Field(default_factory=dict)


class ExternalMASLogsConfig(_LauncherSection):
    """Where the raw external-process stdout/stderr is kept."""

    directory: str | None = None


class ExternalMASLauncherConfig(_LauncherSection):
    """Typed ``launcher_kwargs``: the recipe-private block of the Task Config.

    Validating this block (instead of a free-form mapping) makes a misspelled
    key or an unknown backend fail while the Task Config loads, matching how the
    other recipes reject unknown top-level keys.
    """

    command: ExternalMASCommandConfig
    config: ExternalMASTemplateConfig
    timeout: ExternalMASTimeoutConfig | float | int | None = None
    result: ExternalMASResultConfig | None = None
    logs: ExternalMASLogsConfig | None = None


def _extract_final_answer(info: Mapping[str, Any]) -> str | None:
    """Resolve the answer field without substring-matching the full response."""
    final_answer = info.get("final_answer")
    if final_answer is not None:
        return str(final_answer)

    final_response = info.get("final_response")
    answer = extract_answer(final_response or "")
    if answer is not None:
        return answer

    final_result = info.get("final_result")
    if final_result is not None:
        return extract_answer(final_result) or str(final_result).strip()
    return None


class ExternalMASTaskConfig(TaskConfig):
    """Command settings plus authoritative rollout bindings supplied by the runner."""

    name: str = "multi_policy_external_mas"
    # The sandbox owns execution: every rollout runs its command inside it, and
    # the provider decides whether that is the host, a container, or a remote VM.
    sandbox: SandboxConfig
    launcher_kwargs: ExternalMASLauncherConfig = Field(default_factory=ExternalMASLauncherConfig)
    session_group_id: str
    role_urls: dict[str, str]
    ground_truth: list[str] = Field(default_factory=list)

    @field_validator("agent", mode="before")
    @classmethod
    def _resolve_agent(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and not value.get("name"):
            return AgentConfig(**value)
        return super()._resolve_agent(value)


@register_task("multi_policy_external_mas")
class ExternalMASTask(Task):
    config_model = ExternalMASTaskConfig

    def __init__(self, config: ExternalMASTaskConfig) -> None:
        super().__init__(config)
        self.config: ExternalMASTaskConfig = config

    async def run(self) -> TaskResult:
        config = self.config
        request = ExternalMASRequest(
            prompt=extract_user_task(config.prompt),
            role_urls=dict(config.role_urls),
            session_group_id=config.session_group_id,
        )
        raw_result = await self._run_in_sandbox(request)
        info = dict(raw_result.get("reward_info") or {})
        if raw_result.get("final_result") is not None:
            info.setdefault("final_result", raw_result["final_result"])

        final_answer = _extract_final_answer(info)
        ground_truths = list(config.ground_truth)
        if not ground_truths:
            ground_truths = extract_expected_answers(config.metadata)
        score = compute_score(final_answer, ground_truths)

        return TaskResult(
            reward=score,
            accuracy=score,
            finished=info.get("finished", raw_result.get("finished")),
            extra_info={
                **info,
                "final_answer": final_answer,
                "expected_answers": ground_truths,
            },
        )

    async def _run_in_sandbox(self, request: ExternalMASRequest) -> dict[str, Any]:
        """Run the external MAS command inside this rollout's sandbox.

        Only what the Task Config actually set is forwarded: defaults stay owned
        by ``run_external_mas`` so the execution layer is the single source of
        truth for them.
        """
        launcher_kwargs = self.config.launcher_kwargs.model_dump(exclude_unset=True, exclude_none=True)
        async with self.build_sandbox() as sandbox:
            return await run_external_mas(
                request=request,
                sandbox=sandbox,
                **launcher_kwargs,
            )
