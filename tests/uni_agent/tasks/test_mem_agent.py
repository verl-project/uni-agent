from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

import uni_agent.agents.mem_agent.agent as mem_agent_module
from uni_agent.agents import AgentResult
from uni_agent.agents.mem_agent import MemAgent, MemAgentConfig
from uni_agent.tasks.hotpotqa import HotpotQATask, HotpotQATaskConfig
from uni_agent.tasks.hotpotqa.preprocess import context_to_text, process_example, split_context_into_token_chunks
from uni_agent.tasks.hotpotqa.reward import compute_score, last_boxed_only_string, remove_boxed


class _FakeTokenizer:
    def encode(self, value: str, *, add_special_tokens: bool) -> list[str]:
        assert not add_special_tokens
        return value.split()

    def decode(self, tokens: list[str], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return " ".join(tokens)


class _FakeModel:
    def __init__(self, **_: Any) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    async def query(self, messages, *, sampling_params):
        del sampling_params
        self.calls.append(messages)
        return f"memory-{len(self.calls)}", [], {"prompt_tokens": 3, "completion_tokens": 2}

    async def aclose(self) -> None:
        pass


class _FakeSandbox:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None


class _FakeAgent:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return AgentResult(
            output={"response": "Final answer: \\boxed{alpha beta}"},
            info={"num_contexts": 3, "total_steps": 3},
        )


def test_context_management_methods_are_defined_on_mem_agent():
    assert "update_context" in MemAgent.__dict__
    assert "step" in MemAgent.__dict__


def test_mem_agent_config_does_not_accept_tools():
    assert "tools" not in MemAgentConfig.model_fields
    with pytest.raises(ValidationError):
        MemAgentConfig(tools=[])


def test_mem_agent_config_does_not_accept_tokenizer_or_chunk_size():
    assert "tokenizer_path" not in MemAgentConfig.model_fields
    assert "chunk_size" not in MemAgentConfig.model_fields
    with pytest.raises(ValidationError):
        MemAgentConfig(tokenizer_path="model", chunk_size=2)


@pytest.mark.asyncio
async def test_mem_agent_uses_each_chunk_as_a_new_context(monkeypatch):
    monkeypatch.setattr(mem_agent_module, "OpenAICompatibleChatModel", _FakeModel)
    agent = MemAgent(
        MemAgentConfig(
            max_chunks=2,
            max_steps=3,
            model={"base_url": "http://gateway.invalid/v1", "model_name": "policy"},
        )
    )

    result = await agent.run(
        sandbox=object(),
        messages=[{"role": "user", "content": "question"}],
        raw_data={"chunks": ["one two", "three four"]},
    )

    context_result = result.output["context_manager_result"]
    assert len(context_result.trajectory) == 3
    assert "memory-1" in context_result.trajectory[1].prompt_messages[0]["content"]
    assert result.output["response"] == "memory-3"


def test_mem_agent_reward_scores_last_boxed_answer():
    response = "Earlier: \\boxed{wrong}. Final answer: \\boxed{alpha beta}"

    assert compute_score(response, ["alpha beta"]) == 1.0


def test_mem_agent_reward_returns_zero_without_boxed_answer():
    assert compute_score("alpha beta", ["alpha beta"]) == 0.0


def test_boxed_helpers_preserve_nested_braces():
    boxed = last_boxed_only_string("answer: \\boxed{\\text{alpha beta}}")

    assert boxed == "\\boxed{\\text{alpha beta}}"
    assert remove_boxed(boxed) == "alpha beta"


def test_hotpotqa_preprocess_builds_standard_task_payload():
    example = {
        "id": "sample-id",
        "question": "question",
        "answer": "answer",
        "type": "bridge",
        "level": "hard",
        "context": {"title": ["Title"], "sentences": [["alpha", "beta"]]},
    }

    result = process_example(example, tokenizer=_FakeTokenizer(), chunk_size=2)

    assert context_to_text(example["context"]) == "Title\nalpha\nbeta"
    assert split_context_into_token_chunks(example["context"], tokenizer=_FakeTokenizer(), chunk_size=2) == [
        "Title alpha",
        "beta",
    ]
    assert result == {
        "data_source": "hotpotqa/hotpot_qa",
        "prompt": [{"role": "user", "content": "question"}],
        "extra_info": {
            "tools_kwargs": {
                "task": {
                    "name": "hotpotqa",
                    "prompt": [{"role": "user", "content": "question"}],
                    "ground_truth": ["answer"],
                    "metadata": {
                        "instance_id": "sample-id",
                        "type": "bridge",
                        "level": "hard",
                        "chunks": ["Title alpha", "beta"],
                    },
                }
            }
        },
    }


@pytest.mark.asyncio
async def test_hotpotqa_task_scores_final_response(monkeypatch):
    task = HotpotQATask(
        HotpotQATaskConfig(
            prompt=[{"role": "user", "content": "question"}],
            ground_truth=["alpha beta"],
            metadata={"chunks": ["long context"]},
            sandbox={"provider": "local"},
            agent=MemAgentConfig(),
        )
    )
    monkeypatch.setattr(task, "build_sandbox", lambda: _FakeSandbox())
    agent = _FakeAgent()
    monkeypatch.setattr(task, "build_agent", lambda: agent)

    result = await task.run()

    assert agent.calls[0]["messages"] == [{"role": "user", "content": "question"}]
    assert agent.calls[0]["raw_data"] == {"chunks": ["long context"]}
    assert result.reward == 1.0
    assert result.accuracy == 1.0
    assert result.extra_info == {
        "response": "Final answer: \\boxed{alpha beta}",
        "num_contexts": 3,
        "total_steps": 3,
    }
