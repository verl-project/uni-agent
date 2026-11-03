"""Custom trajectory selection and its discard audit trail.

Kept apart from ``test_generate_sequences_on_cpu`` (1900+ lines of rollout
pipeline tests) because selection is a separate contract. Fixtures come from
``conftest.py``, builders from ``support.py``.
"""

from __future__ import annotations

import json
import logging
import pathlib

import numpy as np
import pytest

from tests.uni_agent.framework.support import (
    _async_noop_runner,
    _build_framework_with_agent_runners,
    _build_prompts,
    _FakeGatewayManager,
    _inline_runner_config,
    _trajectory,
)
from uni_agent.framework.framework import GatewayAgentFramework, _select_session_trajectories
from uni_agent.gateway.session import Trajectory
from verl.utils import tensordict_utils as tu
from verl.utils.import_utils import load_extern_object

_SELECT_DOUBLES_MODULE = "tests.uni_agent.framework.support"


def _select_config(name: str) -> dict[str, object]:
    """Build the mapping form of trajectory_selection pointing at a fixture."""
    return {
        "custom_select_function": {
            "path": f"pkg://{_SELECT_DOUBLES_MODULE}",
            "name": name,
        }
    }


def _custom_runner_config(name: str, **kwargs: object) -> dict[str, object]:
    """Inline runner configured for custom trajectory selection."""
    return _inline_runner_config(
        _async_noop_runner,
        trajectory_selection=_select_config(name),
        **kwargs,
    )


def _dump_arrays(path):
    """Load a ``<stem>.npz`` dump into a plain dict."""
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def _read_dump(run_dir, stem):
    return json.loads((run_dir / f"{stem}.json").read_text())


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("trajectory_selection", "match"),
    [
        ("shortest", "Unknown trajectory selection"),
        ({"custom_select_function": {"path": "", "name": "select"}}, "non-empty 'path' and 'name'"),
        ({"custom_select_function": {"path": "pkg://x", "name": ""}}, "non-empty 'path' and 'name'"),
        ({}, "custom_select_function requires non-empty 'path' and 'name'"),
        (["all"], "trajectory_selection must be"),
        (42, "trajectory_selection must be"),
    ],
)
@pytest.mark.asyncio
async def test_framework_rejects_invalid_trajectory_selection_forms(fake_tq, trajectory_selection, match):
    """Every malformed form fails at start-up with a runner-scoped message."""
    with pytest.raises(ValueError, match=match):
        await _build_framework_with_agent_runners(
            agent_runners={
                "runner": _inline_runner_config(_async_noop_runner, trajectory_selection=trajectory_selection)
            },
            gateway_manager=_FakeGatewayManager({}),
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_framework_rejects_top_level_custom_select_function(fake_tq):
    """A custom_select_function outside trajectory_selection would be silently ignored."""
    with pytest.raises(ValueError, match="must be nested under 'trajectory_selection'"):
        await _build_framework_with_agent_runners(
            agent_runners={
                "runner": _inline_runner_config(
                    _async_noop_runner,
                    extra_config={"custom_select_function": {"path": "pkg://x", "name": "select"}},
                )
            },
            gateway_manager=_FakeGatewayManager({}),
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_framework_rejects_unloadable_or_non_callable_select_function(fake_tq):
    """A bad {path, name} must fail at construction, not on the first session."""
    with pytest.raises(ValueError, match="cannot load select function"):
        await _build_framework_with_agent_runners(
            agent_runners={
                "runner": _inline_runner_config(
                    _async_noop_runner,
                    trajectory_selection={
                        "custom_select_function": {"path": "pkg://no_such_module_xyz", "name": "select"}
                    },
                )
            },
            gateway_manager=_FakeGatewayManager({}),
        )

    with pytest.raises(TypeError, match="is not callable"):
        await _build_framework_with_agent_runners(
            agent_runners={"runner": _custom_runner_config("NOT_CALLABLE")},
            gateway_manager=_FakeGatewayManager({}),
        )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_custom_select_receives_session_and_trajectories(fake_tq):
    """The select function is called once per session with (session_id, trajectories)."""
    from tests.uni_agent.framework import support

    support.CALLS.clear()
    trajectories = [
        _trajectory(response_ids=[30, 31], chain_id=1),
        _trajectory(response_ids=[40, 41], chain_id=2),
    ]
    runtime = _FakeGatewayManager({"session-sample-0-rollout-0": trajectories})
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _custom_runner_config("record_calls")},
        gateway_manager=runtime,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=4))

    assert len(support.CALLS) == 1
    session_id, trajectory_ids = support.CALLS[0]
    assert session_id.startswith("session-sample-0-rollout-0")
    assert trajectory_ids == [id(trajectory) for trajectory in trajectories]
    assert fake_tq.batch_puts[0]["keys"] == ["uid-0_0_0", "uid-0_0_1"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_custom_select_keeps_and_drops_requested_trajectories(fake_tq):
    """Dropping the first trajectory removes exactly that TQ entry."""
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [
                _trajectory(response_ids=[30, 31], chain_id=1),
                _trajectory(response_ids=[40, 41], chain_id=2),
            ]
        }
    )
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _custom_runner_config("drop_first")},
        gateway_manager=runtime,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=4))

    assert len(fake_tq.batch_puts) == 1
    batch = fake_tq.batch_puts[0]
    assert batch["keys"] == ["uid-0_0_0"]
    assert batch["fields"]["responses"][0].tolist() == [40, 41]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_custom_select_preserves_user_order_without_reordering(fake_tq):
    """The framework never reorders: the user's return order is the TQ order."""
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [
                _trajectory(response_ids=[30, 31], chain_id=1),
                _trajectory(response_ids=[40, 41], chain_id=2),
            ]
        }
    )
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _custom_runner_config("reverse_order")},
        gateway_manager=runtime,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=4))

    batch = fake_tq.batch_puts[0]
    assert batch["keys"] == ["uid-0_0_0", "uid-0_0_1"]
    assert batch["fields"]["responses"][0].tolist() == [40, 41]
    assert batch["fields"]["responses"][1].tolist() == [30, 31]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_custom_select_supports_async_functions(fake_tq):
    """An async select function is awaited rather than used as a coroutine."""
    from tests.uni_agent.framework import support

    support.CALLS.clear()
    runtime = _FakeGatewayManager({"session-sample-0-rollout-0": [_trajectory(response_ids=[30, 31])]})
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _custom_runner_config("async_record_calls")},
        gateway_manager=runtime,
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=4))

    assert len(support.CALLS) == 1
    assert fake_tq.batch_puts[0]["keys"] == ["uid-0_0_0"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_custom_select_runs_even_for_single_trajectory(fake_tq):
    """The single-trajectory short-circuit applies to all/longest only."""
    from tests.uni_agent.framework import support

    support.CALLS.clear()
    runtime = _FakeGatewayManager({"session-sample-0-rollout-0": [_trajectory(response_ids=[30, 31])]})
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _custom_runner_config("drop_everything")},
        gateway_manager=runtime,
    )

    with pytest.raises(RuntimeError, match="All rollouts failed"):
        await framework.generate_sequences(_build_prompts(count=1, global_steps=4))

    assert support.CALLS and len(support.CALLS[0][1]) == 1
    assert fake_tq.batch_puts == []


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_custom_select_exception_falls_back_to_all(fake_tq, caplog):
    """A raising select function keeps every trajectory and logs the failure."""
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [
                _trajectory(response_ids=[30, 31], chain_id=1),
                _trajectory(response_ids=[40, 41], chain_id=2),
            ]
        }
    )
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _custom_runner_config("raise_boom")},
        gateway_manager=runtime,
    )

    with caplog.at_level(logging.ERROR, logger="uni_agent.framework.framework"):
        await framework.generate_sequences(_build_prompts(count=1, global_steps=4))

    assert fake_tq.batch_puts[0]["keys"] == ["uid-0_0_0", "uid-0_0_1"]
    assert any("falling back to 'all'" in record.getMessage() for record in caplog.records)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_custom_select_empty_result_uses_empty_status_path(fake_tq):
    """An intentional empty selection produces no TQ write at all."""
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [
                _trajectory(response_ids=[30, 31], chain_id=1),
                _trajectory(response_ids=[40, 41], chain_id=2),
            ]
        }
    )
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _custom_runner_config("drop_everything")},
        gateway_manager=runtime,
    )

    with pytest.raises(RuntimeError, match="All rollouts failed"):
        await framework.generate_sequences(_build_prompts(count=1, global_steps=4))

    assert fake_tq.batch_puts == []
    assert [put["tag"]["status"] for put in fake_tq.puts] == ["running", "failure"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_custom_select_releases_message_history_before_tq_and_scoring(fake_tq, monkeypatch):
    """Message text must not survive into reward scoring or the TQ fields."""
    seen_at_score = []

    async def fake_score(self, trajectories, sample_fields, task_result):
        seen_at_score.extend("message_history" in trajectory.extra_fields for trajectory in trajectories)
        return [(1.0, {}) for _ in trajectories]

    monkeypatch.setattr(GatewayAgentFramework, "_score_trajectories", fake_score)

    message_history = [{"role": "user", "content": "do the task"}]
    dropped = _trajectory(response_ids=[30, 31], chain_id=1, extra_fields={"message_history": message_history})
    kept = _trajectory(response_ids=[40, 41], chain_id=2, extra_fields={"message_history": message_history})
    runtime = _FakeGatewayManager({"session-sample-0-rollout-0": [dropped, kept]})
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _custom_runner_config("drop_first")},
        gateway_manager=runtime,
        reward_loop_worker_handles=["sentinel"],
        reward_config={"custom_reward_function": {"path": "pkg://custom_reward.py"}},
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=4))

    assert "message_history" not in dropped.extra_fields
    assert "message_history" not in kept.extra_fields
    assert seen_at_score == [False]
    for extra in tu.get(fake_tq.batch_puts[0]["fields"], "extra_fields"):
        assert "message_history" not in extra


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_custom_select_pass_through_flag_is_set_only_for_custom(fake_tq):
    """The gateway only copies message text for runners that actually read it."""
    runtime_custom = _FakeGatewayManager({"session-sample-0-rollout-0": [_trajectory()]})
    framework_custom = await _build_framework_with_agent_runners(
        agent_runners={"runner": _custom_runner_config("record_calls")},
        gateway_manager=runtime_custom,
    )
    await framework_custom.generate_sequences(_build_prompts(count=1, global_steps=4))
    assert runtime_custom.created_session_kwargs[0]["pass_through_message_history"] is True

    runtime_all = _FakeGatewayManager({"session-sample-0-rollout-0": [_trajectory()]})
    framework_all = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=runtime_all,
    )
    await framework_all.generate_sequences(_build_prompts(count=1, global_steps=4))
    assert runtime_all.created_session_kwargs[0]["pass_through_message_history"] is False


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_multiple_runners_use_their_own_select_function(fake_tq):
    """Two runners in one framework keep independent select functions."""
    from tests.uni_agent.framework import support

    support.CALLS.clear()
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [
                _trajectory(response_ids=[30, 31], chain_id=1),
                _trajectory(response_ids=[40, 41], chain_id=2),
            ],
            "session-sample-1-rollout-0": [
                _trajectory(response_ids=[50, 51], chain_id=1),
                _trajectory(response_ids=[60, 61], chain_id=2),
            ],
        }
    )
    prompts = _build_prompts(count=2, global_steps=4)
    prompts["agent_name"] = tu.get_tensordict(
        tensor_dict={"agent_name": ["drop", "keep"]},
        non_tensor_dict={},
    )["agent_name"]

    framework = await _build_framework_with_agent_runners(
        agent_runners={
            "drop": _custom_runner_config("drop_first"),
            "keep": _custom_runner_config("record_calls"),
        },
        gateway_manager=runtime,
    )

    await framework.generate_sequences(prompts)

    by_uid = {put["keys"][0].split("_")[0]: put for put in fake_tq.batch_puts}
    assert by_uid["uid-0"]["keys"] == ["uid-0_0_0"]
    assert by_uid["uid-0"]["fields"]["responses"][0].tolist() == [40, 41]
    assert by_uid["uid-1"]["keys"] == ["uid-1_0_0", "uid-1_0_1"]
    assert len(support.CALLS) == 2


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_mixed_custom_and_all_runners_in_one_framework(tmp_path, fake_tq):
    """Only the custom runner pays the text pass-through and writes an audit dump."""
    runtime = _FakeGatewayManager(
        {
            "session-sample-0-rollout-0": [
                _trajectory(chain_id=1, num_turns=5),
                _trajectory(chain_id=2, num_turns=2),
            ],
            "session-sample-1-rollout-0": [
                _trajectory(chain_id=1, num_turns=5),
                _trajectory(chain_id=2, num_turns=2),
            ],
        }
    )
    prompts = _build_prompts(count=2, global_steps=3)
    prompts["agent_name"] = tu.get_tensordict(
        tensor_dict={"agent_name": ["drop", "keepall"]},
        non_tensor_dict={},
    )["agent_name"]

    framework = await _build_framework_with_agent_runners(
        agent_runners={
            "drop": _custom_runner_config("drop_first"),
            "keepall": _inline_runner_config(_async_noop_runner),
        },
        gateway_manager=runtime,
        log_dir=str(tmp_path),
    )

    await framework.generate_sequences(prompts)

    assert [kwargs["pass_through_message_history"] for kwargs in runtime.created_session_kwargs] == [True, False]

    by_uid = {put["keys"][0].split("_")[0]: put for put in fake_tq.batch_puts}
    assert by_uid["uid-0"]["keys"] == ["uid-0_0_0"]
    assert by_uid["uid-1"]["keys"] == ["uid-1_0_0", "uid-1_0_1"]

    dumps = {}
    for session_dir in (tmp_path / "step_3").iterdir():
        dumps[session_dir.name] = sorted(path.name for path in session_dir.iterdir())
    audit_names = [name for name, entries in dumps.items() if "dropped_trajectory.json" in entries]
    plain_names = [name for name, entries in dumps.items() if "dropped_trajectory.json" not in entries]
    assert len(audit_names) == 1
    assert len(plain_names) == 1
    audited = _read_dump(tmp_path / "step_3" / audit_names[0], "dropped_trajectory")
    assert audited["num_trajectories"] == 1
    assert audited["trajectories"][0]["num_turns"] == 5


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_dropped_trajectories_are_dumped_like_kept_ones(tmp_path, fake_tq):
    """A discarded chain must stay inspectable: same files, same fields, same arrays."""
    # drop_first keeps everything but the FIRST trajectory, so list the discard first.
    discarded = _trajectory(prompt_ids=[50, 51], response_ids=[60, 61, 62], num_turns=7, chain_id=1)
    kept = _trajectory(prompt_ids=[10, 11], response_ids=[40, 41], num_turns=3, chain_id=2)
    runtime = _FakeGatewayManager({"session-sample-0-rollout-0": [discarded, kept]})
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _custom_runner_config("drop_first")},
        gateway_manager=runtime,
        log_dir=str(tmp_path),
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=5))

    session_dir = next((tmp_path / "step_5").iterdir())
    kept_dump = _read_dump(session_dir, "trajectory")
    dropped_dump = _read_dump(session_dir, "dropped_trajectory")

    assert kept_dump["num_trajectories"] == 1
    assert kept_dump["trajectories"][0]["num_turns"] == 3
    assert dropped_dump["num_trajectories"] == 1
    assert dropped_dump["trajectories"][0]["num_turns"] == 7
    assert set(kept_dump["trajectories"][0]) == set(dropped_dump["trajectories"][0])

    kept_arrays = _dump_arrays(session_dir / "trajectory.npz")
    dropped_arrays = _dump_arrays(session_dir / "dropped_trajectory.npz")
    assert kept_arrays["traj0_response_ids"].tolist() == [40, 41]
    assert dropped_arrays["traj0_response_ids"].tolist() == [60, 61, 62]
    assert dropped_arrays["traj0_prompt_ids"].tolist() == [50, 51]
    assert set(kept_arrays) == set(dropped_arrays)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_no_dropped_files_when_nothing_is_dropped(tmp_path, fake_tq):
    """A session that keeps everything must not gain empty audit files."""
    runtime = _FakeGatewayManager({"session-sample-0-rollout-0": [_trajectory()]})
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner)},
        gateway_manager=runtime,
        log_dir=str(tmp_path),
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=9))

    session_dir = next((tmp_path / "step_9").iterdir())
    assert (session_dir / "trajectory.json").exists()
    assert not (session_dir / "dropped_trajectory.json").exists()
    assert not (session_dir / "dropped_trajectory.npz").exists()


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_longest_selection_audits_its_discards(tmp_path, fake_tq):
    """Auditing is not custom-only: longest drops n-1 chains and those are recorded."""
    winner = _trajectory(response_ids=[30, 31, 32], response_mask=[1, 1, 1], num_turns=2, chain_id=1)
    loser = _trajectory(
        response_ids=[20, 21, 22, 23, 24, 25], response_mask=[1, 0, 0, 0, 0, 0], num_turns=10, chain_id=2
    )
    runtime = _FakeGatewayManager({"session-sample-0-rollout-0": [winner, loser]})
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _inline_runner_config(_async_noop_runner, trajectory_selection="longest")},
        gateway_manager=runtime,
        log_dir=str(tmp_path),
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=6))

    session_dir = next((tmp_path / "step_6").iterdir())
    kept_dump = _read_dump(session_dir, "trajectory")
    dropped_dump = _read_dump(session_dir, "dropped_trajectory")
    assert kept_dump["num_trajectories"] == 1
    assert kept_dump["trajectories"][0]["num_turns"] == 2
    assert dropped_dump["num_trajectories"] == 1
    assert dropped_dump["trajectories"][0]["num_turns"] == 10


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_discard_dump_survives_a_copied_kept_list(tmp_path, fake_tq):
    """A selector may return copies; the discards must still be reported correctly."""
    kept = _trajectory(chain_id=1, num_turns=3)
    discarded = _trajectory(chain_id=2, num_turns=7)
    runtime = _FakeGatewayManager({"session-sample-0-rollout-0": [kept, discarded]})
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _custom_runner_config("deepcopy_first")},
        gateway_manager=runtime,
        log_dir=str(tmp_path),
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=2))

    session_dir = next((tmp_path / "step_2").iterdir())
    dropped_dump = _read_dump(session_dir, "dropped_trajectory")
    assert dropped_dump["num_trajectories"] == 1
    assert dropped_dump["trajectories"][0]["num_turns"] == 7


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_empty_selection_still_audits_the_discards(tmp_path, fake_tq):
    """A selector that drops everything writes only the discard dump."""
    runtime = _FakeGatewayManager({"session-sample-0-rollout-0": [_trajectory(num_turns=4, chain_id=1)]})
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _custom_runner_config("drop_everything")},
        gateway_manager=runtime,
        log_dir=str(tmp_path),
    )

    with pytest.raises(RuntimeError, match="All rollouts failed"):
        await framework.generate_sequences(_build_prompts(count=1, global_steps=7))

    session_dir = next((tmp_path / "step_7").iterdir())
    assert not (session_dir / "trajectory.json").exists()
    assert not (session_dir / "trajectory.npz").exists()
    dropped_dump = _read_dump(session_dir, "dropped_trajectory")
    assert dropped_dump["num_trajectories"] == 1
    assert dropped_dump["trajectories"][0]["num_turns"] == 4


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_in_place_selection_still_audits_the_discards(tmp_path, fake_tq):
    """A selector may mutate the list it was given; the discard diff must survive."""
    discarded = _trajectory(prompt_ids=[50, 51], response_ids=[60], num_turns=5, chain_id=1)
    kept = _trajectory(prompt_ids=[10, 11], response_ids=[40, 41], num_turns=2, chain_id=2)
    runtime = _FakeGatewayManager({"session-sample-0-rollout-0": [discarded, kept]})
    framework = await _build_framework_with_agent_runners(
        agent_runners={"runner": _custom_runner_config("drop_first_in_place")},
        gateway_manager=runtime,
        log_dir=str(tmp_path),
    )

    await framework.generate_sequences(_build_prompts(count=1, global_steps=3))

    session_dir = next((tmp_path / "step_3").iterdir())
    assert _read_dump(session_dir, "trajectory")["trajectories"][0]["num_turns"] == 2
    dropped_dump = _read_dump(session_dir, "dropped_trajectory")
    assert dropped_dump["num_trajectories"] == 1
    assert dropped_dump["trajectories"][0]["num_turns"] == 5


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_in_place_mutation_then_raise_still_falls_back_to_everything():
    """A function that empties the list and raises must not lose trajectories."""

    first = _trajectory(chain_id=1)
    second = _trajectory(chain_id=2)
    trajectories = [first, second]

    def boom(session_id, items):
        items.clear()
        raise RuntimeError("boom")

    kept = await _select_session_trajectories("s", trajectories, "all", select_fn=boom)
    assert kept == [first, second]


# ---------------------------------------------------------------------------
# Sample selectors shipped in examples/quickstart/trajectory_selectors/
# ---------------------------------------------------------------------------

_SAMPLE_SELECTORS = (
    pathlib.Path(__file__).resolve().parents[3] / "examples" / "quickstart" / "trajectory_selectors" / "selectors.py"
)


def _sample_selector(name: str):
    """Load a sample selector the way the framework does: by file path."""
    return load_extern_object(module_path=str(_SAMPLE_SELECTORS), object_name=name)


def _sample_trajectory(chain_id: int, text: str, role: str = "user") -> Trajectory:
    return Trajectory(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        num_turns=1,
        chain_id=chain_id,
        extra_fields={"message_history": [{"role": role, "content": text}]},
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("name", "dropped_chain_id"),
    [
        ("drop_title_segments", 1),
        ("drop_compaction_segments", 2),
        ("drop_subagent_segments", 3),
        ("drop_non_work_segments", None),
    ],
)
def test_sample_selectors_drop_their_own_segment_only(name, dropped_chain_id):
    """Each sample drops the segment it is named for and keeps the working chain."""
    title = _sample_trajectory(1, "Create a concise title for this session", role="system")
    compaction = _sample_trajectory(2, "<state_snapshot> summarize the conversation so far")
    subagent = _sample_trajectory(3, "You are a subagent spawned by the main agent", role="system")
    work = _sample_trajectory(4, "def f(): return 1")
    trajectories = [title, compaction, subagent, work]

    kept = _sample_selector(name)("session", list(trajectories))
    kept_ids = [trajectory.chain_id for trajectory in kept]

    assert work.chain_id in kept_ids
    if dropped_chain_id is None:
        assert kept_ids == [4]
    else:
        assert dropped_chain_id not in kept_ids
        assert len(kept_ids) == 3


@pytest.mark.cpu
@pytest.mark.level0
def test_sample_selectors_preserve_the_order_they_were_given():
    """Reward scoring dispatches the last returned entry, so order must survive.

    The samples must not sort: a session with a subagent closes that chain first
    and the main chain last, and the main chain is the one that carries the task
    result. Reordering would hand the reward worker a subagent transcript.
    """
    selector = _sample_selector("drop_non_work_segments")
    # Creation order is title(1), subagent(2), work(3); close order is the reverse
    # for the first two, because the subagent finished before the main chain did.
    subagent = _sample_trajectory(2, "You are a subagent spawned by the main agent", role="system")
    title = _sample_trajectory(1, "Create a concise title for this session", role="system")
    suppressed = _sample_trajectory(4, "<state_snapshot> summarize the conversation so far")
    work = _sample_trajectory(3, "def f(): return 1")

    kept = selector("session", [subagent, title, suppressed, work])

    assert [trajectory.chain_id for trajectory in kept] == [3]
    assert kept[-1] is work


@pytest.mark.cpu
@pytest.mark.level0
def test_sample_selectors_keep_relative_order_of_the_survivors():
    """A filter must not reorder: survivors stay in the order they arrived."""
    selector = _sample_selector("drop_title_segments")
    work_late = _sample_trajectory(1, "second worker chain")
    title = _sample_trajectory(2, "Create a concise title for this session", role="system")
    work_first = _sample_trajectory(3, "first worker chain")

    kept = selector("session", [work_late, title, work_first])

    assert kept == [work_late, work_first]


@pytest.mark.cpu
@pytest.mark.level0
def test_sample_selector_keeps_first_work_segment_by_creation_order():
    """chain_id picks the earliest non-title chain, not the first one in the list."""
    selector = _sample_selector("keep_first_work_segment")
    title = _sample_trajectory(1, "Create a concise title for this session", role="system")
    earliest_work = _sample_trajectory(2, "first worker chain")
    later_work = _sample_trajectory(3, "second worker chain")

    # Close order is the reverse: the earliest chain finished last.
    kept = selector("session", [later_work, title, earliest_work])

    assert kept == [earliest_work]


@pytest.mark.cpu
@pytest.mark.level0
def test_sample_selector_keeps_first_work_segment_edge_cases():
    """No work at all returns empty; a missing chain_id must not raise."""
    selector = _sample_selector("keep_first_work_segment")
    title_only = [_sample_trajectory(1, "Create a concise title for this session", role="system")]

    assert selector("session", title_only) == []
    assert selector("session", []) == []

    no_chain_id = Trajectory(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        num_turns=1,
        chain_id=None,
        extra_fields={"message_history": [{"role": "user", "content": "work"}]},
    )
    assert selector("session", [no_chain_id]) == [no_chain_id]


@pytest.mark.cpu
@pytest.mark.level0
def test_sample_selectors_survive_empty_and_bare_input():
    """No trajectories, and a trajectory without message_history, must not raise."""
    selector = _sample_selector("drop_non_work_segments")

    assert selector("session", []) == []
    bare = Trajectory(prompt_ids=[1], response_ids=[2], response_mask=[1], num_turns=1, chain_id=1, extra_fields={})
    assert selector("session", [bare]) == [bare]


@pytest.mark.cpu
@pytest.mark.level0
def test_sample_selectors_read_multimodal_and_tool_call_content():
    """Text can sit in content parts or in tool calls; neither may crash the matcher."""
    selector = _sample_selector("drop_non_work_segments")
    multimodal = Trajectory(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        num_turns=1,
        chain_id=1,
        extra_fields={
            "message_history": [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image"}]}]
        },
    )
    tool_call = Trajectory(
        prompt_ids=[1],
        response_ids=[2],
        response_mask=[1],
        num_turns=1,
        chain_id=2,
        extra_fields={
            "message_history": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"name": "bash", "arguments": "ls"}}],
                }
            ]
        },
    )

    assert selector("session", [multimodal, tool_call]) == [multimodal, tool_call]


@pytest.mark.cpu
@pytest.mark.level0
def test_sample_selectors_export_every_documented_name():
    """The README table is the contract users copy from."""
    for name in (
        "drop_title_segments",
        "drop_compaction_segments",
        "drop_subagent_segments",
        "drop_non_work_segments",
        "is_title_segment",
        "is_compaction_segment",
        "is_subagent_segment",
    ):
        assert callable(_sample_selector(name)), name
