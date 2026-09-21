import asyncio

from tests.uni_agent.support import FakeTokenizer, SequencedBackend
from uni_agent.gateway.adapters.openai import openai_to_internal
from uni_agent.gateway.annotation import infer_trajectory_annotations
from uni_agent.gateway.session import GatewaySession, MessageCodec, SessionHandle


def test_default_policy_recognizes_verified_harness_signals():
    annotations = infer_trajectory_annotations(
        headers={
            "x-claude-code-agent-id": "child-123",
            "x-codex-turn-metadata": '{"request_kind":"compaction"}',
            "x-deepseek-harness-compact": "1",
        },
        body={
            "messages": [
                {
                    "role": "system",
                    "content": "# Subagent Context\nSubagent spawned by main agent; one specific task.",
                },
                {
                    "role": "user",
                    "content": "The conversation history before this point was compacted into the following summary:",
                },
            ],
            "dsh_session_log": {"session": {"origin": "subagent"}},
        },
        protocol="openai_chat",
    )

    assert annotations == {
        "tags": [
            "context:compacted_history",
            "purpose:compaction_request",
            "role:subagent",
        ],
        "evidence": [
            "body:dsh_session_log.session.origin=subagent",
            "header:x-claude-code-agent-id:present",
            "header:x-codex-turn-metadata.request_kind=compaction",
            "header:x-deepseek-harness-compact=1",
            "prompt:openclaw-compacted-history-marker",
            "prompt:openclaw-subagent-marker",
        ],
    }


def test_default_policy_distinguishes_openclaw_branch_summary_and_unknown_role():
    annotations = infer_trajectory_annotations(
        headers={},
        body={
            "system": "You are a context summarization assistant.",
            "messages": [
                {
                    "role": "user",
                    "content": "The following is a summary of a branch that this conversation came back from:",
                }
            ],
        },
        protocol="anthropic_messages",
    )

    assert annotations == {
        "tags": ["context:branch_return_summary", "purpose:summary_generation", "role:unknown"],
        "evidence": [
            "prompt:openclaw-branch-summary-marker",
            "prompt:openclaw-summarizer-system",
        ],
    }


def test_default_policy_does_not_infer_main_agent_from_missing_signals():
    annotations = infer_trajectory_annotations(headers={}, body={"messages": []}, protocol="openai_chat")

    assert annotations == {"tags": ["role:unknown"], "evidence": []}


def test_prompt_markers_in_later_user_content_are_not_treated_as_harness_signals():
    annotations = infer_trajectory_annotations(
        headers={},
        body={
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Please quote [Subagent Context] You are running as a subagent"},
                {"role": "user", "content": "The conversation history before this point was compacted into"},
            ],
        },
        protocol="openai_chat",
    )

    assert annotations == {"tags": ["role:unknown"], "evidence": []}


def test_session_persists_one_annotation_per_successful_generation():
    session = GatewaySession(SessionHandle("annotation-session"), MessageCodec(FakeTokenizer()))
    request = openai_to_internal(
        {"model": "dummy", "messages": [{"role": "user", "content": "task"}]},
        base_sampling_params={},
        allowed_sampling_keys=frozenset({"max_tokens", "stop"}),
    )
    request["annotation_headers"] = {"x-openai-subagent": "collab_spawn"}
    request["annotation_body"] = dict(request)
    request["annotation_protocol"] = "openai_chat"

    asyncio.run(session.run_generation(request, SequencedBackend(["done"])))
    [trajectory] = asyncio.run(session.finalize())

    assert trajectory.extra_fields["trajectory_annotations"] == [
        {
            "tags": ["role:subagent"],
            "evidence": ["header:x-openai-subagent=collab_spawn"],
        }
    ]


def test_session_accepts_custom_annotation_policy():
    observed = {}

    def policy(headers, body, protocol):
        observed.update(headers=headers, body=body, protocol=protocol)
        return {"tags": ["role:custom", "role:custom"], "evidence": ["custom:marker"]}

    session = GatewaySession(
        SessionHandle("custom-annotation-session"),
        MessageCodec(FakeTokenizer()),
        annotation_policy=policy,
    )
    request = openai_to_internal(
        {"model": "dummy", "messages": [{"role": "user", "content": "task"}]},
        base_sampling_params={},
        allowed_sampling_keys=frozenset({"max_tokens", "stop"}),
    )
    request["annotation_headers"] = {"x-custom": "present"}
    request["annotation_body"] = {"messages": request["messages"]}
    request["annotation_protocol"] = "openai_chat"

    asyncio.run(session.run_generation(request, SequencedBackend(["done"])))
    [trajectory] = asyncio.run(session.finalize())

    assert observed == {
        "headers": {"x-custom": "present"},
        "body": {"messages": request["messages"]},
        "protocol": "openai_chat",
    }
    assert trajectory.extra_fields["trajectory_annotations"] == [
        {"tags": ["role:custom"], "evidence": ["custom:marker"]}
    ]
