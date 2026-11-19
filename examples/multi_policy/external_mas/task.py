"""External-MAS Task definition and scoring."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, field_validator

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
    # The external command owns execution; no Uni-Agent sandbox is constructed.
    sandbox: SandboxConfig | None = None
    launcher_kwargs: dict[str, Any] = Field(default_factory=dict)
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
        raw_result = await run_external_mas(request=request, **config.launcher_kwargs)
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
